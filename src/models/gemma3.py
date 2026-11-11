import torch
import torch.nn as nn
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
from transformers.modeling_outputs import BaseModelOutputWithPast
import sys
import os
import types
import importlib
import inspect
import warnings
import functools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.base import STASModelBase


def _patch_transformers_compat():
    import sys
    import dataclasses

    from transformers import configuration_utils
    if not hasattr(configuration_utils, 'PreTrainedConfig'):
        configuration_utils.PreTrainedConfig = configuration_utils.PretrainedConfig

    if not hasattr(configuration_utils, 'layer_type_validation'):
        def layer_type_validation(layer_types):
            if layer_types is None:
                return
            valid_types = {"full_attention", "sliding_attention", "cross_attention"}
            for lt in layer_types:
                if lt not in valid_types:
                    raise ValueError(f"Unknown layer type: {lt}")
        configuration_utils.layer_type_validation = layer_type_validation

    from transformers import utils
    if not hasattr(utils, 'auto_docstring'):
        def auto_docstring(cls_or_fn=None, **kwargs):
            if cls_or_fn is not None:
                return cls_or_fn
            return lambda x: x
        utils.auto_docstring = auto_docstring

    try:
        import huggingface_hub.dataclasses
        def _patched_strict(cls=None, **kwargs):
            def decorator(cls):
                return cls
            if cls is not None:
                return decorator(cls)
            return decorator
        huggingface_hub.dataclasses.strict = _patched_strict
    except (ImportError, AttributeError):
        pass

    if 'transformers.masking_utils' not in sys.modules:
        try:
            from transformers import masking_utils
        except ImportError:
            _masking_utils = types.ModuleType('transformers.masking_utils')

            def _create_causal_mask(
                config, inputs_embeds, attention_mask=None, past_key_values=None,
                position_ids=None, **kwargs
            ):
                batch_size, seq_len, _ = inputs_embeds.shape
                past_len = 0
                if past_key_values is not None:
                    past_len = past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else 0
                total_len = past_len + seq_len

                mask = torch.triu(
                    torch.ones(seq_len, total_len, dtype=torch.bool, device=inputs_embeds.device),
                    diagonal=1 + past_len
                )
                mask = mask.unsqueeze(0).unsqueeze(0)

                if attention_mask is not None:
                    if attention_mask.dim() == 2:
                        attn_len = attention_mask.shape[-1]
                        if attn_len < total_len:
                            past_mask = torch.ones(batch_size, total_len - attn_len, dtype=attention_mask.dtype, device=attention_mask.device)
                            full_mask = torch.cat([past_mask, attention_mask], dim=-1)
                        elif attn_len > total_len:
                            full_mask = attention_mask[:, :total_len]
                        else:
                            full_mask = attention_mask
                        expanded_mask = full_mask.unsqueeze(1).unsqueeze(2)
                        if full_mask.dtype == torch.bool:
                            padding_mask = ~expanded_mask
                        else:
                            padding_mask = (expanded_mask == 0)
                        mask = mask.expand(batch_size, -1, -1, -1)
                        mask = mask | padding_mask
                    elif attention_mask.dim() == 4:
                        mask = mask | (attention_mask == 0)

                mask = mask.float().masked_fill(mask, float('-inf'))
                if mask.shape[0] == 1 and batch_size > 1:
                    mask = mask.expand(batch_size, -1, -1, -1)
                return mask.to(inputs_embeds.dtype)

            def _create_sliding_window_causal_mask(
                config, inputs_embeds, attention_mask=None, past_key_values=None,
                position_ids=None, sliding_window=None, or_mask_function=None, **kwargs
            ):
                batch_size, seq_len, _ = inputs_embeds.shape
                past_len = 0
                if past_key_values is not None:
                    past_len = past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else 0
                total_len = past_len + seq_len

                window = sliding_window
                if window is None:
                    window = getattr(config, 'sliding_window', None)
                if window is None:
                    window = total_len

                q_positions = torch.arange(past_len, past_len + seq_len, device=inputs_embeds.device)
                kv_positions = torch.arange(total_len, device=inputs_embeds.device)
                diff = q_positions.unsqueeze(1) - kv_positions.unsqueeze(0)
                causal_mask = diff < 0
                window_mask = diff >= window
                mask = causal_mask | window_mask
                mask = mask.unsqueeze(0).unsqueeze(0)

                if attention_mask is not None:
                    if attention_mask.dim() == 2:
                        attn_len = attention_mask.shape[-1]
                        if attn_len < total_len:
                            past_mask = torch.ones(batch_size, total_len - attn_len, dtype=attention_mask.dtype, device=attention_mask.device)
                            full_mask = torch.cat([past_mask, attention_mask], dim=-1)
                        elif attn_len > total_len:
                            full_mask = attention_mask[:, :total_len]
                        else:
                            full_mask = attention_mask
                        expanded_mask = full_mask.unsqueeze(1).unsqueeze(2)
                        if full_mask.dtype == torch.bool:
                            padding_mask = ~expanded_mask
                        else:
                            padding_mask = (expanded_mask == 0)
                        mask = mask.expand(batch_size, -1, -1, -1)
                        mask = mask | padding_mask
                    elif attention_mask.dim() == 4:
                        mask = mask | (attention_mask == 0)

                if or_mask_function is not None:
                    vision_group_ids = None
                    try:
                        vision_group_ids = or_mask_function.__closure__[1].cell_contents
                    except (AttributeError, IndexError, TypeError):
                        try:
                            for cell in (or_mask_function.__closure__ or []):
                                try:
                                    val = cell.cell_contents
                                    if isinstance(val, torch.Tensor) and val.dim() == 2:
                                        vision_group_ids = val
                                        break
                                except Exception:
                                    pass
                        except Exception:
                            pass

                    if vision_group_ids is not None:
                        q_groups = vision_group_ids[:, :seq_len]
                        kv_groups = torch.cat([
                            torch.full((batch_size, past_len), -1, dtype=vision_group_ids.dtype, device=vision_group_ids.device),
                            vision_group_ids[:, :seq_len]
                        ], dim=-1) if past_len > 0 else vision_group_ids[:, :total_len]
                        if kv_groups.shape[-1] < total_len:
                            pad_len = total_len - kv_groups.shape[-1]
                            kv_groups = torch.cat([kv_groups, torch.full((batch_size, pad_len), -1, dtype=vision_group_ids.dtype, device=vision_group_ids.device)], dim=-1)
                        elif kv_groups.shape[-1] > total_len:
                            kv_groups = kv_groups[:, :total_len]

                        same_group = (q_groups.unsqueeze(-1) == kv_groups.unsqueeze(-2)) & (q_groups.unsqueeze(-1) >= 0)
                        same_group_4d = same_group.unsqueeze(1)
                        mask = mask.expand(batch_size, -1, -1, -1) if mask.shape[0] == 1 else mask
                        mask = mask & ~same_group_4d
                    else:
                        _seq_len = seq_len
                        _total_len = total_len
                        has_or = False
                        for q in range(min(_seq_len, 5)):
                            for k in range(min(_total_len, 10)):
                                try:
                                    result = or_mask_function(0, 0, q, k)
                                    if isinstance(result, torch.Tensor) and result.item():
                                        has_or = True
                                        break
                                    elif result:
                                        has_or = True
                                        break
                                except Exception:
                                    pass
                            if has_or:
                                break
                        if has_or:
                            or_mask_full = torch.zeros(1, 1, _seq_len, _total_len, dtype=torch.bool, device=inputs_embeds.device)
                            for q in range(_seq_len):
                                for k in range(_total_len):
                                    try:
                                        result = or_mask_function(0, 0, q, k)
                                        if isinstance(result, torch.Tensor):
                                            or_mask_full[0, 0, q, k] = result.item()
                                        elif result:
                                            or_mask_full[0, 0, q, k] = True
                                    except Exception:
                                        pass
                            mask = mask.expand(batch_size, -1, -1, -1) if mask.shape[0] == 1 else mask
                            mask = mask & ~or_mask_full

                mask = mask.float().masked_fill(mask, float('-inf'))
                if mask.shape[0] == 1 and batch_size > 1:
                    mask = mask.expand(batch_size, -1, -1, -1)
                return mask.to(inputs_embeds.dtype)

            def _create_bidirectional_mask(
                config=None, inputs_embeds=None, attention_mask=None, past_key_values=None,
                position_ids=None, **kwargs
            ):
                if inputs_embeds is None:
                    return None
                batch_size, seq_len, _ = inputs_embeds.shape
                past_len = 0
                if past_key_values is not None:
                    past_len = past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else 0
                total_len = past_len + seq_len

                mask = torch.zeros(1, 1, seq_len, total_len, dtype=torch.float32, device=inputs_embeds.device)

                if attention_mask is not None and attention_mask.dim() == 2:
                    if past_len > 0:
                        past_mask = torch.ones(batch_size, past_len, dtype=attention_mask.dtype, device=attention_mask.device)
                        full_mask = torch.cat([past_mask, attention_mask], dim=-1)
                    else:
                        full_mask = attention_mask
                    expanded_mask = full_mask.float().unsqueeze(1).unsqueeze(2)
                    padding_mask = expanded_mask.masked_fill(expanded_mask == 0, float('-inf'))
                    padding_mask = padding_mask.masked_fill(expanded_mask != 0, 0.0)
                    mask = mask + padding_mask
                    if mask.shape[0] < batch_size:
                        mask = mask.expand(batch_size, -1, -1, -1)
                return mask.to(inputs_embeds.dtype)

            def _create_masks_for_generate(
                config, inputs_embeds, attention_mask=None, past_key_values=None,
                position_ids=None, **kwargs
            ):
                mask_kwargs = {
                    "config": config if not hasattr(config, 'get_text_config') else config.get_text_config(),
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "position_ids": position_ids,
                }
                return {
                    "full_attention": _create_causal_mask(**mask_kwargs),
                    "sliding_attention": _create_sliding_window_causal_mask(**mask_kwargs),
                }

            def _create_causal_mask_mapping(
                config, inputs_embeds, attention_mask=None, past_key_values=None,
                position_ids=None, mm_token_type_ids=None, token_type_ids=None,
                is_first_iteration=None, **kwargs
            ):
                mask_kwargs = {
                    "config": config.get_text_config() if hasattr(config, 'get_text_config') else config,
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "position_ids": position_ids,
                }
                sliding_mask_kwargs = mask_kwargs.copy()

                tids = token_type_ids or mm_token_type_ids
                if tids is not None:
                    is_vision = (tids == 1) | (tids == 2)
                    is_prev_vision = torch.roll(is_vision, shifts=1, dims=-1)
                    is_prev_vision[..., 0] = False
                    new_vision_starts = is_vision & ~is_prev_vision
                    vision_group_ids = torch.cumsum(new_vision_starts.int(), dim=1) - 1
                    vision_group_ids = torch.where(is_vision, vision_group_ids, -1)

                    def _or_mask_function(batch_idx, head_idx, q_idx, kv_idx):
                        seq_length = vision_group_ids.shape[-1]
                        q_idx_c = q_idx.clamp(max=seq_length - 1)
                        kv_idx_c = kv_idx.clamp(max=seq_length - 1)
                        q_group = vision_group_ids[batch_idx, q_idx_c]
                        kv_group = vision_group_ids[batch_idx, kv_idx_c]
                        q_group = torch.where(q_idx < seq_length, q_group, -1)
                        kv_group = torch.where(kv_idx < seq_length, kv_group, -1)
                        return (q_group == kv_group) & (q_group >= 0)

                    sliding_mask_kwargs["or_mask_function"] = _or_mask_function

                return {
                    "full_attention": _create_causal_mask(**mask_kwargs),
                    "sliding_attention": _create_sliding_window_causal_mask(**sliding_mask_kwargs),
                }

            _masking_utils.create_bidirectional_mask = _create_bidirectional_mask
            _masking_utils.create_causal_mask = _create_causal_mask
            _masking_utils.create_causal_mask_mapping = _create_causal_mask_mapping
            _masking_utils.create_masks_for_generate = _create_masks_for_generate
            _masking_utils.create_sliding_window_causal_mask = _create_sliding_window_causal_mask
            sys.modules['transformers.masking_utils'] = _masking_utils
            utils.masking_utils = _masking_utils

    if 'transformers.modeling_layers' not in sys.modules:
        try:
            from transformers import modeling_layers
        except ImportError:
            _modeling_layers = types.ModuleType('transformers.modeling_layers')
            from torch import nn as _nn
            class GradientCheckpointingLayer(_nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()
            _modeling_layers.GradientCheckpointingLayer = GradientCheckpointingLayer
            sys.modules['transformers.modeling_layers'] = _modeling_layers

    if 'transformers.utils.output_capturing' not in sys.modules:
        _oc = types.ModuleType('transformers.utils.output_capturing')
        class OutputRecorder:
            def __init__(self, layer_type, index=0):
                self.layer_type = layer_type
                self.index = index
        def capture_outputs(forward_fn):
            @functools.wraps(forward_fn)
            def wrapped_forward(self, *args, **kwargs):
                output_hidden_states = kwargs.pop('output_hidden_states', None)
                output_attentions = kwargs.pop('output_attentions', None)

                if not output_hidden_states and not output_attentions:
                    return forward_fn(self, *args, **kwargs)

                _can_record = getattr(self.__class__, '_can_record_outputs', None)
                if _can_record is None:
                    return forward_fn(self, *args, **kwargs)

                hooks = []
                hidden_states_list = []
                attentions_list = []

                if output_hidden_states and 'hidden_states' in _can_record:
                    layer_type = _can_record['hidden_states']
                    if isinstance(layer_type, OutputRecorder):
                        layer_type = layer_type.layer_type
                    for module in self.modules():
                        if isinstance(module, layer_type):
                            def _hs_hook(mod, inp, out, _lst=hidden_states_list):
                                if isinstance(out, tuple):
                                    _lst.append(out[0].detach() if isinstance(out[0], torch.Tensor) else out[0])
                                elif isinstance(out, torch.Tensor):
                                    _lst.append(out.detach())
                                else:
                                    _lst.append(out)
                            h = module.register_forward_hook(_hs_hook)
                            hooks.append(h)

                if output_attentions and 'attentions' in _can_record:
                    attn_type = _can_record['attentions']
                    if isinstance(attn_type, OutputRecorder):
                        attn_type = attn_type.layer_type
                    for module in self.modules():
                        if isinstance(module, attn_type):
                            def _attn_hook(mod, inp, out, _lst=attentions_list):
                                if isinstance(out, tuple):
                                    if len(out) > 1 and isinstance(out[1], tuple):
                                        _lst.append(tuple(a.detach() if isinstance(a, torch.Tensor) else a for a in out[1]))
                                    elif len(out) > 1 and isinstance(out[1], torch.Tensor):
                                        _lst.append(out[1].detach())
                                elif isinstance(out, torch.Tensor):
                                    _lst.append(out.detach())
                            h = module.register_forward_hook(_attn_hook)
                            hooks.append(h)

                try:
                    outputs = forward_fn(self, *args, **kwargs)
                finally:
                    for h in hooks:
                        h.remove()

                if isinstance(outputs, dict):
                    if output_hidden_states and hidden_states_list:
                        outputs['hidden_states'] = tuple(hidden_states_list)
                    if output_attentions and attentions_list:
                        outputs['attentions'] = tuple(attentions_list)
                elif hasattr(outputs, '__dict__'):
                    if output_hidden_states and hidden_states_list:
                        try:
                            outputs.hidden_states = tuple(hidden_states_list)
                        except Exception:
                            pass
                    if output_attentions and attentions_list:
                        try:
                            outputs.attentions = tuple(attentions_list)
                        except Exception:
                            pass

                return outputs

            return wrapped_forward
        _oc.OutputRecorder = OutputRecorder
        _oc.capture_outputs = capture_outputs
        sys.modules['transformers.utils.output_capturing'] = _oc

    if 'transformers.initialization' not in sys.modules:
        try:
            from transformers import initialization as _test_init
            if not hasattr(_test_init, 'constant_'):
                import torch.nn.init as _nn_init
                for _name in dir(_nn_init):
                    if not _name.startswith('_'):
                        setattr(_test_init, _name, getattr(_nn_init, _name))
        except ImportError:
            import torch.nn.init as _nn_init
            _init = types.ModuleType('transformers.initialization')
            for _name in dir(_nn_init):
                if not _name.startswith('_'):
                    setattr(_init, _name, getattr(_nn_init, _name))
            sys.modules['transformers.initialization'] = _init

    try:
        from transformers.integrations import use_experts_implementation as _uei, use_kernelized_func as _ukf
    except ImportError:
        _integrations = importlib.import_module('transformers.integrations')

        def _noop_decorator_cls(func_or_cls=None, **kw):
            if func_or_cls is not None:
                return func_or_cls
            return lambda x: x

        def _kernelized_func_decorator(kernel_fn=None, **kw):
            def class_decorator(cls):
                if kernel_fn is not None:
                    cls._kernelized_func = kernel_fn
                return cls
            return class_decorator

        _integrations.use_experts_implementation = _noop_decorator_cls
        _integrations.use_kernelized_func = _kernelized_func_decorator

    if 'transformers.utils.type_validators' not in sys.modules:
        _tv = types.ModuleType('transformers.utils.type_validators')
        def interval(min=0.0, max=1.0):
            class IntervalDescriptor:
                def __call__(self, default=None):
                    return default
            return IntervalDescriptor()
        _tv.interval = interval
        sys.modules['transformers.utils.type_validators'] = _tv
        utils.type_validators = _tv

    if not hasattr(utils, 'TransformersKwargs'):
        utils.TransformersKwargs = object

    if not hasattr(utils, 'torch_compilable_check'):
        utils.torch_compilable_check = lambda *a, **kw: True

    if not hasattr(utils.import_utils, 'requires'):
        def requires(backends=()):
            def decorator(cls):
                return cls
            return decorator
        utils.import_utils.requires = requires

    if 'transformers.video_utils' not in sys.modules:
        _video_utils = types.ModuleType('transformers.video_utils')
        class VideoInput:
            pass
        _video_utils.VideoInput = VideoInput
        sys.modules['transformers.video_utils'] = _video_utils

    if 'transformers.image_processing_backends' not in sys.modules:
        try:
            from transformers import image_processing_backends
        except ImportError:
            import numpy as _np
            from transformers.image_processing_utils import BaseImageProcessor
            from transformers.image_utils import (
                PILImageResampling, make_nested_list_of_images,
                to_numpy_array,
            )

            _ipb = types.ModuleType('transformers.image_processing_backends')

            class TorchvisionBackend(BaseImageProcessor):
                def rescale_and_normalize(self, image, do_rescale, rescale_factor, do_normalize, image_mean, image_std):
                    if do_rescale:
                        image = self.rescale(image, scale=rescale_factor)
                    if do_normalize:
                        image = self.normalize(image=image, mean=image_mean, std=image_std)
                    return image

            class PilBackend(BaseImageProcessor):
                def preprocess(self, images, **kwargs):
                    images = make_nested_list_of_images(images)
                    processed_images = []
                    for image_list in images:
                        for image in image_list:
                            if isinstance(image, Image.Image):
                                if getattr(self, 'do_convert_rgb', True):
                                    image = image.convert("RGB")
                                image = to_numpy_array(image)
                                if image.ndim == 3:
                                    image = image.transpose(2, 0, 1)
                            processed_images.append(image)

                    _preprocess_params = {}
                    sig = inspect.signature(self._preprocess)
                    for param_name, param in sig.parameters.items():
                        if param_name in ('self', 'images', 'kwargs'):
                            continue
                        if param_name in kwargs:
                            _preprocess_params[param_name] = kwargs[param_name]
                        elif hasattr(self, param_name):
                            _preprocess_params[param_name] = getattr(self, param_name)

                    return self._preprocess(processed_images, **_preprocess_params)

            _ipb.TorchvisionBackend = TorchvisionBackend
            _ipb.PilBackend = PilBackend
            sys.modules['transformers.image_processing_backends'] = _ipb

    _generic = importlib.import_module('transformers.utils.generic')
    _need_maybe_autocast_patch = True
    if hasattr(_generic, 'maybe_autocast'):
        import inspect as _inspect
        try:
            _sig = _inspect.signature(_generic.maybe_autocast)
            if 'device_type' in _sig.parameters:
                _need_maybe_autocast_patch = False
        except (ValueError, TypeError):
            pass
    if _need_maybe_autocast_patch:
        from contextlib import nullcontext
        def maybe_autocast(device_type=None, enabled=True):
            if device_type is not None and enabled:
                try:
                    return torch.autocast(device_type=device_type, enabled=False)
                except Exception:
                    pass
            return nullcontext()
        _generic.maybe_autocast = maybe_autocast
        utils.maybe_autocast = maybe_autocast
    if not hasattr(_generic, 'merge_with_config_defaults'):
        def merge_with_config_defaults(fn):
            return fn
        _generic.merge_with_config_defaults = merge_with_config_defaults
        utils.merge_with_config_defaults = merge_with_config_defaults

    from transformers import processing_utils
    if not hasattr(processing_utils, 'MultiModalData'):
        processing_utils.MultiModalData = object

    import jinja2
    try:
        major, minor = jinja2.__version__.split('.')[:2]
        version_tuple = (int(major), int(minor))
    except (ValueError, AttributeError):
        version_tuple = (0, 0)

    if version_tuple < (3, 0):
        if hasattr(jinja2, 'evalcontextfunction'):
            jinja2.pass_eval_context = jinja2.evalcontextfunction
        elif hasattr(jinja2, 'contextfunction'):
            jinja2.pass_eval_context = jinja2.contextfunction
        if hasattr(jinja2, 'contextfunction'):
            jinja2.pass_context = jinja2.contextfunction
    if version_tuple < (3, 1):
        jinja2.__version__ = '3.1.0'

    try:
        from transformers import modeling_flash_attention_utils
        if not hasattr(modeling_flash_attention_utils, 'flash_attn_varlen_func'):
            try:
                from flash_attn import flash_attn_varlen_func as _favf
                modeling_flash_attention_utils.flash_attn_varlen_func = _favf
            except ImportError:
                pass
    except Exception:
        pass

    try:
        from transformers.modeling_utils import SUPPORTED_TP_STYLES
        _new_styles = {'replicated_with_grad_allreduce', 'replicated', 'replicated_rowwise', 'packed_colwise', 'moe_tp_experts'}
        SUPPORTED_TP_STYLES.update(_new_styles)
    except (ImportError, AttributeError, TypeError):
        pass

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if not hasattr(ALL_ATTENTION_FUNCTIONS, 'get_interface'):
        def _get_interface(self, implementation, eager_fn):
            attn_fn = self.get(implementation)
            if attn_fn is not None:
                return attn_fn
            return eager_fn
        ALL_ATTENTION_FUNCTIONS.__class__.get_interface = _get_interface

    import transformers.modeling_rope_utils as _rope_utils
    _orig_dynamic_rope_update = _rope_utils.dynamic_rope_update
    from functools import wraps as _wraps

    def _patched_dynamic_rope_update(rope_forward):
        _orig_wrapper = _orig_dynamic_rope_update(rope_forward)

        @_wraps(rope_forward)
        def wrapper(self, x, position_ids, *args, **kwargs):
            layer_type = kwargs.get('layer_type')
            if layer_type is None and len(args) > 0:
                layer_type = args[0]

            rope_type = getattr(self, 'rope_type', None)
            if rope_type is not None and isinstance(rope_type, dict):
                cur_rope_type = rope_type.get(layer_type, 'default') if layer_type else 'default'
            else:
                cur_rope_type = getattr(self, 'rope_type', 'default')

            if 'dynamic' in str(cur_rope_type):
                seq_len = torch.max(position_ids) + 1
                if seq_len > self.max_seq_len_cached:
                    inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device=x.device, seq_len=seq_len)
                    self.register_buffer("inv_freq", inv_freq, persistent=False)
                    self.max_seq_len_cached = seq_len
                if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
                    self.original_inv_freq = self.original_inv_freq.to(x.device)
                    self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
                    self.max_seq_len_cached = self.original_max_seq_len
            elif cur_rope_type == 'longrope':
                seq_len = torch.max(position_ids) + 1
                if hasattr(self.config, "original_max_position_embeddings"):
                    original_max_position_embeddings = self.config.original_max_position_embeddings
                else:
                    original_max_position_embeddings = self.config.max_position_embeddings
                if seq_len > original_max_position_embeddings:
                    if not hasattr(self, "long_inv_freq"):
                        self.long_inv_freq, _ = self.rope_init_fn(self.config, x.device, seq_len=original_max_position_embeddings + 1)
                    self.register_buffer("inv_freq", self.long_inv_freq, persistent=False)
                else:
                    self.original_inv_freq = self.original_inv_freq.to(x.device)
                    self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)

            return rope_forward(self, x, position_ids, *args, **kwargs)

        return wrapper

    _rope_utils.dynamic_rope_update = _patched_dynamic_rope_update

    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        if 'proportional' not in ROPE_INIT_FUNCTIONS:
            def _compute_proportional_rope_parameters(
                config=None, device=None, seq_len=None, **rope_kwargs,
            ):
                if config is not None:
                    layer_type = rope_kwargs.get('layer_type', 'full_attention')
                    head_dim_key = rope_kwargs.get('head_dim_key', 'head_dim')
                    rope_params = config.rope_parameters.get(layer_type, {})
                    base = rope_params.get('rope_theta', getattr(config, 'rope_theta', 10000.0))
                    partial_rotary_factor = rope_params.get('partial_rotary_factor', 1.0)
                    head_dim = getattr(config, head_dim_key, None) or getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
                    dim = int(head_dim * partial_rotary_factor)
                else:
                    base = rope_kwargs.get('base', 10000.0)
                    dim = rope_kwargs.get('dim', 64)

                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim))
                return inv_freq, 1.0

            ROPE_INIT_FUNCTIONS['proportional'] = _compute_proportional_rope_parameters
    except (ImportError, AttributeError):
        pass

    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        if 'linear' not in ROPE_INIT_FUNCTIONS:
            if 'linear' not in ROPE_INIT_FUNCTIONS:
                def _compute_linear_rope_parameters(
                    config=None, device=None, seq_len=None, **rope_kwargs,
                ):
                    if config is not None:
                        factor = getattr(config, 'factor', getattr(config, 'rope_scaling', {}).get('factor', 1.0)) if not isinstance(config, dict) else config.get('factor', 1.0)
                        base = getattr(config, 'rope_theta', 10000.0)
                        head_dim = getattr(config, 'head_dim', None) or getattr(config, 'hidden_size', 2560) // getattr(config, 'num_attention_heads', 8)
                        dim = int(head_dim)
                    else:
                        base = rope_kwargs.get('base', 10000.0)
                        dim = rope_kwargs.get('dim', 256)
                        factor = rope_kwargs.get('factor', 1.0)

                    base = base / factor
                    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim))
                    attention_scaling = 1.0
                    return inv_freq, attention_scaling

                ROPE_INIT_FUNCTIONS['linear'] = _compute_linear_rope_parameters
    except (ImportError, AttributeError):
        pass

    from transformers.cache_utils import DynamicCache as _DynamicCache
    _dc_sig = inspect.signature(_DynamicCache.__init__)
    if 'config' not in _dc_sig.parameters:
        _orig_dc_init = _DynamicCache.__init__
        def _patched_dc_init(self, *args, **kwargs):
            kwargs.pop('config', None)
            _orig_dc_init(self, *args, **kwargs)
        _DynamicCache.__init__ = _patched_dc_init

    if 'transformers.modeling_layers' not in sys.modules:
        try:
            from transformers.modeling_layers import GenericForSequenceClassification
        except ImportError:
            pass
    try:
        from transformers.modeling_layers import GenericForSequenceClassification
    except ImportError:
        try:
            _ml = sys.modules.get('transformers.modeling_layers')
            if _ml is not None and not hasattr(_ml, 'GenericForSequenceClassification'):
                from transformers import PreTrainedModel
                class GenericForSequenceClassification(_nn.Module):
                    def __init__(self, config, *args, **kwargs):
                        super().__init__()
                _ml.GenericForSequenceClassification = GenericForSequenceClassification
        except Exception:
            pass

    if not hasattr(processing_utils, 'Unpack'):
        from typing import Any as _Any
        processing_utils.Unpack = _Any


_gemma3_registered = False


def _register_gemma3():
    global _gemma3_registered
    if _gemma3_registered:
        return

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    if 'gemma3' in CONFIG_MAPPING:
        _gemma3_registered = True
        return

    import importlib

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    gemma3_dir = os.path.join(project_root, 'models', 'gemma3')

    transformers_pkg = importlib.import_module('transformers')
    models_pkg = importlib.import_module('transformers.models')

    gemma3_pkg = types.ModuleType('transformers.models.gemma3')
    gemma3_pkg.__path__ = [gemma3_dir]
    gemma3_pkg.__package__ = 'transformers.models.gemma3'
    gemma3_pkg.__file__ = os.path.join(gemma3_dir, '__init__.py')
    sys.modules['transformers.models.gemma3'] = gemma3_pkg
    models_pkg.gemma3 = gemma3_pkg

    if 'transformers.models.siglip' not in sys.modules:
        try:
            import transformers.models.siglip as _siglip
        except ImportError:
            pass

    from transformers.models.gemma3.configuration_gemma3 import (
        Gemma3Config, Gemma3TextConfig
    )

    from transformers.models.gemma3.modeling_gemma3 import (
        Gemma3ForConditionalGeneration, Gemma3TextModel,
    )

    try:
        from transformers.models.gemma3.processing_gemma3 import Gemma3Processor
    except ImportError as e:
        print(f"Warning: Could not import Gemma3Processor: {e}")
        Gemma3Processor = None

    from transformers import AutoModelForCausalLM, AutoModel, AutoConfig
    from transformers.models.auto import modeling_auto, processing_auto

    CONFIG_MAPPING._mapping['gemma3'] = 'Gemma3Config'
    CONFIG_MAPPING._extra_content['gemma3'] = Gemma3Config

    CONFIG_MAPPING._extra_content['gemma3_text'] = Gemma3TextConfig

    modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES['gemma3'] = 'Gemma3ForConditionalGeneration'

    AutoModel._model_mapping.register(Gemma3TextConfig, Gemma3TextModel, exist_ok=True)

    if not hasattr(Gemma3ForConditionalGeneration, 'config_class') or Gemma3ForConditionalGeneration.config_class is None:
        Gemma3ForConditionalGeneration.config_class = Gemma3Config

    try:
        AutoModelForCausalLM.register(Gemma3Config, Gemma3ForConditionalGeneration)
    except (AttributeError, Exception) as e:
        print(f"Warning: Could not register with AutoModelForCausalLM: {e}")

    if Gemma3Processor is not None:
        processing_auto.PROCESSOR_MAPPING_NAMES['gemma3'] = 'Gemma3Processor'

    _gemma3_registered = True
    print("Gemma3 registered with transformers Auto classes")


_patch_transformers_compat()


class Gemma3Model(STASModelBase):

    DEFAULT_PATH = "/mnt/r-contentsecurity-p/common/datas_yl/chizhu/models/gemma-3-4b-it"

    def __init__(self, model_path: str = None, device: str = 'cuda', **kwargs):
        super().__init__(model_path or self.DEFAULT_PATH, device, **kwargs)

        self.hidden_dim = 2560
        self.num_layers = 34
        self.num_heads = 8
        self.visual_dim = 2560

        self.load_model()

    def load_model(self):
        print(f"Loading Gemma3 from: {self.model_path}")

        _patch_transformers_compat()

        _register_gemma3()

        from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration
        from transformers.models.gemma3.configuration_gemma3 import Gemma3Config, Gemma3TextConfig

        import json as _json
        config_path = os.path.join(self.model_path, 'config.json')
        with open(config_path) as f:
            config_dict = _json.load(f)

        from transformers.models.siglip import SiglipVisionConfig
        text_config = Gemma3TextConfig(**config_dict.get('text_config', {}))
        vision_config = SiglipVisionConfig(**config_dict.get('vision_config', {})) if config_dict.get('vision_config') else None

        config = Gemma3Config(
            text_config=text_config,
            vision_config=vision_config,
        )
        for key in ['mm_tokens_per_image', 'boi_token_index', 'eoi_token_index',
                     'image_token_index', 'initializer_range', 'tie_word_embeddings']:
            if key in config_dict:
                setattr(config, key, config_dict[key])

        self.hidden_dim = text_config.hidden_size
        self.num_layers = text_config.num_hidden_layers
        self.num_heads = text_config.num_attention_heads
        self.visual_dim = self.hidden_dim

        print(f"Gemma3 config: hidden_dim={self.hidden_dim}, "
              f"num_layers={self.num_layers}, num_heads={self.num_heads}, "
              f"visual_dim={self.visual_dim}")

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        ).eval()

        if config.text_config.tie_word_embeddings:
            self.model.language_model.lm_head.weight = self.model.language_model.model.embed_tokens.weight

        self._load_processor()

        self.freeze_model()

    def _load_processor(self):
        import json as _json

        tokenizer_config_path = os.path.join(self.model_path, 'tokenizer_config.json')
        extra_special_tokens_backup = None
        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, 'r') as f:
                tokenizer_config = _json.load(f)
            if 'extra_special_tokens' in tokenizer_config:
                est = tokenizer_config['extra_special_tokens']
                if isinstance(est, list):
                    extra_special_tokens_backup = est
                    tokenizer_config['extra_special_tokens'] = {}
                    with open(tokenizer_config_path, 'w') as f:
                        _json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except Exception as e:
            if extra_special_tokens_backup is not None:
                tokenizer_config['extra_special_tokens'] = extra_special_tokens_backup
                with open(tokenizer_config_path, 'w') as f:
                    _json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
            raise RuntimeError(f"Failed to load tokenizer: {e}")

        if extra_special_tokens_backup is not None:
            tokenizer_config['extra_special_tokens'] = extra_special_tokens_backup
            with open(tokenizer_config_path, 'w') as f:
                _json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

        self.image_processor = None

        preprocessor_config_path = os.path.join(self.model_path, 'preprocessor_config.json')
        ip_config = {}
        if os.path.exists(preprocessor_config_path):
            with open(preprocessor_config_path, 'r') as f:
                ip_config = _json.load(f)
            ip_config.pop('image_processor_type', None)
            ip_config.pop('processor_class', None)

        try:
            from transformers.models.gemma3.image_processing_pil_gemma3 import Gemma3ImageProcessorPil
            self.image_processor = Gemma3ImageProcessorPil(**ip_config) if ip_config else Gemma3ImageProcessorPil()
            print(f"Image processor loaded: Gemma3ImageProcessorPil")
        except Exception as e:
            print(f"Warning: Failed to load Gemma3ImageProcessorPil: {e}")
            try:
                from transformers.models.gemma3.image_processing_gemma3 import Gemma3ImageProcessor
                self.image_processor = Gemma3ImageProcessor(**ip_config) if ip_config else Gemma3ImageProcessor()
                print(f"Image processor loaded: Gemma3ImageProcessor (torchvision)")
            except Exception as e2:
                print(f"Warning: Failed to load image processor: {e2}")

        self.processor = None

        self.freeze_model()

    def _patch_gemma3_prepare_inputs_for_generation(self):
        import functools
        import inspect

        if not hasattr(self.model, 'prepare_inputs_for_generation'):
            return

        if hasattr(self.model, '_gemma3_prepare_call_count'):
            self.model._gemma3_prepare_call_count = 0
            return

        orig_prepare = self.model.prepare_inputs_for_generation
        sig = inspect.signature(orig_prepare)

        if 'is_first_iteration' not in sig.parameters:
            return

        self.model._gemma3_prepare_call_count = 0

        @functools.wraps(orig_prepare)
        def patched_prepare(*args, **kwargs):
            self.model._gemma3_prepare_call_count += 1
            kwargs['is_first_iteration'] = (self.model._gemma3_prepare_call_count == 1)
            return orig_prepare(*args, **kwargs)

        self.model.prepare_inputs_for_generation = patched_prepare

    def _prepare_inputs(self, image: Image.Image, text: str, answer: str = None) -> Dict:
        has_image = image is not None and not isinstance(image, str)

        boi_token_id = self.model.config.boi_token_index
        eoi_token_id = self.model.config.eoi_token_index
        image_token_id = self.model.config.image_token_index
        mm_tokens_per_image = self.model.config.mm_tokens_per_image

        pixel_values = None
        if has_image and self.image_processor is not None:
            image_inputs = self.image_processor(images=[image], return_tensors="pt")
            pixel_values = image_inputs["pixel_values"]

        if has_image:
            prompt_before = "<start_of_turn>user\n"
            prompt_after = f"{text}<end_of_turn>\n<start_of_turn>model\n"

            before_ids = self.tokenizer(prompt_before, add_special_tokens=True, return_tensors="pt")["input_ids"]
            after_ids = self.tokenizer(prompt_after, add_special_tokens=False, return_tensors="pt")["input_ids"]

            img_tokens = torch.full((1, mm_tokens_per_image), image_token_id, dtype=torch.long)
            input_ids = torch.cat([
                before_ids,
                torch.tensor([[boi_token_id]], dtype=torch.long),
                img_tokens,
                torch.tensor([[eoi_token_id]], dtype=torch.long),
                after_ids,
            ], dim=-1)
            attention_mask = torch.ones_like(input_ids)

            token_type_ids = torch.zeros_like(input_ids)
            boi_pos = before_ids.shape[-1]
            eoi_end = boi_pos + 1 + mm_tokens_per_image + 1
            token_type_ids[0, boi_pos:eoi_end] = 1
        else:
            prompt = f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
            text_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = text_inputs["input_ids"]
            attention_mask = text_inputs["attention_mask"]
            token_type_ids = None

        if answer is not None:
            answer_ids = self.tokenizer(answer, add_special_tokens=False, return_tensors="pt")["input_ids"]
            input_ids = torch.cat([input_ids, answer_ids], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(answer_ids)], dim=-1)
            if token_type_ids is not None:
                token_type_ids = torch.cat(
                    [token_type_ids, torch.zeros_like(answer_ids)],
                    dim=-1
                )

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids

        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values

        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        return inputs


    def generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 256,
        **kwargs
    ) -> str:
        inputs = self._prepare_inputs(image, question)

        self._patch_gemma3_prepare_inputs_for_generation()

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        response = self.tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return response


    def _get_decoder_layers(self):
        try:
            return self.model.model.language_model.layers
        except AttributeError:
            try:
                return self.model.language_model.model.layers
            except AttributeError:
                try:
                    return self.model.language_model.layers
                except AttributeError:
                    raise AttributeError("Cannot find Gemma3 decoder layers")

    def _get_lm_head(self):
        try:
            return self.model.language_model.lm_head
        except AttributeError:
            return self.model.lm_head

    def _get_final_norm(self):
        try:
            return self.model.language_model.model.norm
        except AttributeError:
            try:
                return self.model.model.language_model.norm
            except AttributeError:
                return self.model.model.norm
