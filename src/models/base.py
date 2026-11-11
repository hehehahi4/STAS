import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List
from PIL import Image
from src.stas import mix_directions, same_token_jsd_gating, response_token_span, paper_profile


class STASModelBase(ABC):

    def __init__(self, model_path: str, device: str = 'cuda', **kwargs):
        self.model_name = kwargs.get('model_name')
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self.config = None

        self.hidden_dim = None
        self.num_layers = None
        self.num_heads = None
        self.visual_dim = None

    @abstractmethod
    def load_model(self):
        pass

    def get_description_hidden_states(self, image, question, target_layers, answer):
        import re
        match = re.match(r'^\s*(yes|no)\b', answer, re.IGNORECASE)
        if match is None or not answer[match.end():].strip(' .,!:;\n'):
            raise ValueError("STAS needs full Yes/No-conditioned descriptions; regenerate the response pairs")
        inputs = self._prepare_inputs(image, question, answer=answer)
        empty = self._prepare_inputs(image, question, answer='')
        prefix = self._prepare_inputs(image, question, answer=answer[:match.end()])
        ids = inputs['input_ids'][0].tolist()
        start, stop = response_token_span(ids, empty['input_ids'][0].tolist(),
                                          prefix['input_ids'][0].tolist())
        del empty, prefix
        states, handles = {}, []
        layers = self._get_decoder_layers()

        def collect(layer_idx):
            def hook(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                shift = hidden.shape[1] - len(ids)
                if start + shift < 0 or stop + shift > hidden.shape[1]:
                    raise ValueError("Cannot align the response span with decoder outputs")
                states[layer_idx] = hidden[:, start + shift:stop + shift].float().mean(dim=1).detach()
            return hook

        try:
            for layer_idx in target_layers:
                handles.append(layers[layer_idx].register_forward_hook(collect(layer_idx)))
            with torch.no_grad():
                self.model(**inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return states

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 256,
        **kwargs
    ) -> str:
        pass

    def _get_decoder_layers(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _get_decoder_layers."
        )

    def _get_lm_head(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _get_lm_head."
        )

    def _get_final_norm(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _get_final_norm."
        )

    def get_router_token_features(self, image, question):
        from src.router import extract_router_tokens
        return extract_router_tokens(self, image, question)

    def freeze_model(self):
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()

    def generate_with_stas(
        self, image, question, pca_basis, mean_directions, target_layers,
        visual_router, base_alpha=2.0, routing_weight=0.5,
        beta=5.0, tau=None, return_routing_stats=False, **kwargs,
    ):
        if getattr(visual_router, 'router_type', None) != 'token_mlp':
            raise ValueError('This entry requires a token_mlp router')
        if tau is None:
            _, tau = paper_profile(self.model_name)
        targets = sorted(set(target_layers))
        if not targets or not set(targets).issubset(visual_router.target_layers):
            raise ValueError('Requested intervention layers are absent from the token router')
        layers = self._get_decoder_layers()
        if targets[0] < 0 or targets[-1] >= len(layers):
            raise ValueError('Invalid intervention layer index')
        if not set(targets).issubset(set(pca_basis) & set(mean_directions)):
            raise ValueError('Missing PCA basis or mean direction for an intervention layer')
        visual, text = self.get_router_token_features(image, question)
        parameter = next(visual_router.parameters())
        visual, text = visual.to(parameter), text.to(parameter)
        directions, routing_info = {}, {}
        with torch.no_grad():
            for layer_idx in targets:
                coefficients = visual_router.coefficients(layer_idx, visual, text)[0]
                basis = pca_basis[layer_idx].to(coefficients)
                mean = mean_directions[layer_idx].to(coefficients)
                directions[layer_idx] = mix_directions(mean, basis, coefficients, routing_weight)
                if return_routing_stats:
                    mass = coefficients.abs() / coefficients.abs().sum().clamp_min(1e-12)
                    routing_info[layer_idx] = {
                        'weights': coefficients.cpu().tolist(),
                        'top_component': coefficients.abs().argmax().item(),
                        'max_weight': coefficients.abs().max().item(),
                        'entropy': -(mass * mass.clamp_min(1e-12).log()).sum().item(),
                        'entropy_kind': 'absolute_coefficient_mass',
                        'cos_sim_with_mean': F.cosine_similarity(
                            (coefficients @ basis).unsqueeze(0), mean.unsqueeze(0)).item(),
                        'direction_norm': directions[layer_idx].norm().item(),
                    }
        stats = {layer_idx: [] for layer_idx in targets}

        def make_hook(layer_idx):
            def hook(module, inputs, output):
                if getattr(self, '_stas_phase', None) == 'probe':
                    return output
                hidden = output[0] if isinstance(output, tuple) else output
                gate = getattr(self, '_stas_gates', {}).get(layer_idx)
                steer = base_alpha * directions[layer_idx].to(hidden)
                if gate is not None:
                    steer = steer * gate.to(hidden).view(-1, 1, 1)
                edited = hidden.clone()
                edited[:, -1:, :] = hidden[:, -1:, :] + steer
                if return_routing_stats:
                    divergence = getattr(self, '_stas_jsd', {}).get(layer_idx)
                    stats[layer_idx].append({**routing_info[layer_idx],
                        'gamma': gate.mean().item() if gate is not None else 1.0,
                        'jsd': divergence.mean().item() if divergence is not None else None})
                return (edited,) + output[1:] if isinstance(output, tuple) else edited
            return hook

        handles = []
        context = same_token_jsd_gating(self, targets, beta, tau)
        try:
            for layer_idx in targets:
                handles.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
            with context, torch.no_grad():
                response = self.generate(image, question, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        return (response, stats) if return_routing_stats else response
