import copy
import functools
import inspect
import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F


def normalized_coefficients(prediction):
    return F.normalize(prediction.float(), dim=-1)


def router_alignment_loss(prediction, target):
    return (normalized_coefficients(prediction)
            - normalized_coefficients(target)).square().sum(dim=-1).mean()


def response_token_span(full_ids, empty_ids, prefix_ids):
    start = 0
    while start < min(len(full_ids), len(empty_ids)) and full_ids[start] == empty_ids[start]:
        start += 1
    stop, other_stop = len(full_ids), len(empty_ids)
    while stop > start and other_stop > start and full_ids[stop - 1] == empty_ids[other_stop - 1]:
        stop -= 1
        other_stop -= 1
    prefix_end = 0
    while prefix_end < min(len(full_ids), len(prefix_ids)) and full_ids[prefix_end] == prefix_ids[prefix_end]:
        prefix_end += 1
    start = max(start, prefix_end)
    if start >= stop:
        raise ValueError("A full Yes/No-conditioned description is required, not a bare answer")
    return start, stop


def mix_directions(mean_direction, basis, coefficients, routing_weight):
    if not 0 <= routing_weight <= 1:
        raise ValueError("routing_weight must be in [0, 1]")
    anchor = F.normalize(mean_direction.float(), dim=-1)
    dynamic = normalized_coefficients(coefficients) @ basis.float()
    return (1 - routing_weight) * anchor + routing_weight * dynamic


def js_divergence(logp, logq):
    logm = torch.logaddexp(logp, logq) - math.log(2)
    divergence = 0.5 * ((logp.exp() * (logp - logm)).sum(dim=-1)
                        + (logq.exp() * (logq - logm)).sum(dim=-1))
    return divergence.clamp(min=0, max=math.log(2))


def _log_distribution(hidden, final_norm, lm_head):
    norm_param = next(final_norm.parameters(), None)
    head_param = next(lm_head.parameters(), None)
    if norm_param is not None:
        hidden = hidden.to(norm_param)
    hidden = final_norm(hidden)
    if head_param is not None:
        hidden = hidden.to(head_param)
    return F.log_softmax(lm_head(hidden).float(), dim=-1)


@contextmanager
def same_token_jsd_gating(adapter, target_layers, beta=5.0, tau=0.1):
    if beta <= 0 or tau < 0:
        raise ValueError("beta must be positive and tau nonnegative")
    if getattr(adapter, '_stas_phase', None) is not None:
        raise RuntimeError("Nested STAS generation is not supported")
    layers = adapter._get_decoder_layers()
    targets = sorted(set(target_layers))
    if not targets or min(targets) < 0 or max(targets) >= len(layers):
        raise ValueError("Invalid STAS target layers")
    final_norm = adapter._get_final_norm()
    lm_head = adapter._get_lm_head()
    model = adapter.model
    original_forward = model.forward
    had_forward_override = 'forward' in model.__dict__
    signature = inspect.signature(original_forward)
    captured = {}
    handles = []
    sentinel = object()
    previous = {key: getattr(adapter, key, sentinel)
                for key in ('_stas_phase', '_stas_gates', '_stas_jsd')}

    def capture(layer_idx):
        def hook(module, inputs, output):
            if adapter._stas_phase == 'probe':
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = hidden[:, -1, :].detach().clone()
        return hook

    @functools.wraps(original_forward)
    def forward(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        mappings = [bound.arguments]
        for name, parameter in signature.parameters.items():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                bound.arguments[name] = dict(bound.arguments.get(name, {}))
                mappings.append(bound.arguments[name])
        for mapping in mappings:
            if mapping.get('past_key_values') is not None:
                mapping['past_key_values'] = copy.deepcopy(mapping['past_key_values'])
        captured.clear()
        adapter._stas_phase = 'probe'
        with torch.no_grad():
            probe = original_forward(*bound.args, **bound.kwargs)
            del probe
            logp_final = _log_distribution(captured[len(layers) - 1], final_norm, lm_head)
            gates, divergences = {}, {}
            for layer_idx in targets:
                logp = _log_distribution(captured[layer_idx], final_norm, lm_head)
                uncertainty = js_divergence(logp, logp_final)
                divergences[layer_idx] = uncertainty
                gates[layer_idx] = torch.sigmoid(beta * (uncertainty - tau))
        captured.clear()
        adapter._stas_gates = gates
        adapter._stas_jsd = divergences
        adapter._stas_phase = 'steer'
        return original_forward(*args, **kwargs)

    try:
        adapter._stas_phase = 'probe'
        adapter._stas_gates = {}
        adapter._stas_jsd = {}
        for layer_idx in sorted(set(targets + [len(layers) - 1])):
            handles.append(layers[layer_idx].register_forward_hook(capture(layer_idx)))
        model.forward = forward
        yield
    finally:
        if had_forward_override:
            model.forward = original_forward
        elif 'forward' in model.__dict__:
            del model.forward
        for handle in handles:
            handle.remove()
        for key, value in previous.items():
            if value is sentinel:
                if hasattr(adapter, key):
                    delattr(adapter, key)
            else:
                setattr(adapter, key, value)


PAPER_PROFILES = {
    'qwen3_vl': (18, 35, 0.1),
    'qwen3_vl_32b': (48, 63, 0.2),
    'llava_ov1_5': (18, 35, 0.15),
    'llava_ov1_5_8b': (18, 35, 0.15),
    'gemma3': (17, 33, 0.05),
    'gemma3_4b': (17, 33, 0.05),
    'gemma3_27b': (40, 59, 0.2),
}


def paper_profile(model_name):
    if model_name not in PAPER_PROFILES:
        raise ValueError(f"No paper profile for {model_name}; specify layers and tau explicitly")
    first, last, tau = PAPER_PROFILES[model_name]
    return list(range(first, last + 1)), tau
