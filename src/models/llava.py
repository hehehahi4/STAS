import torch
import torch.nn as nn
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
import sys
import os
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.base import STASModelBase


def _patch_jinja2_compat():
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


def _patch_transformers_flash_attention():
    try:
        from transformers import modeling_flash_attention_utils

        if not hasattr(modeling_flash_attention_utils, 'flash_attn_varlen_func'):
            try:
                from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
                modeling_flash_attention_utils.flash_attn_varlen_func = _flash_attn_varlen_func
            except ImportError:
                def _flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                           dropout_p=0.0, softmax_scale=None, causal=False,
                                           window_size=(-1, -1), alibi_slopes=None):
                    if softmax_scale is None:
                        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)
                    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
                    attn_weights = torch.softmax(attn_weights, dim=-1)
                    output = torch.matmul(attn_weights, v)
                    return output

                modeling_flash_attention_utils.flash_attn_varlen_func = _flash_attn_varlen_func
                warnings.warn("Using compatibility layer for flash_attn_varlen_func")
    except Exception as e:
        warnings.warn(f"Could not patch flash attention: {e}")


_patch_jinja2_compat()
_patch_transformers_flash_attention()


class LlavaOV1_5Model(STASModelBase):

    DEFAULT_PATH = "/mnt/r-contentsecurity-p/common/datas_yl/chizhu/models/LLaVA-OneVision-1.5-8B-Instruct"

    def __init__(self, model_path: str = None, device: str = 'cuda', **kwargs):
        super().__init__(model_path or self.DEFAULT_PATH, device, **kwargs)

        self.hidden_dim = 4096
        self.num_layers = 36
        self.num_heads = 32
        self.visual_dim = 1024

        self.load_model()

    @staticmethod
    def _remap_checkpoint_keys(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.layers.') or key.startswith('model.embed_tokens.') or key.startswith('model.norm.'):
                new_key = key.replace('model.', 'model.language_model.', 1)
                new_state_dict[new_key] = value
            elif key.startswith('visual.'):
                new_key = 'model.' + key
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        return new_state_dict

    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
        from safetensors.torch import load_file
        import glob

        print(f"Loading LLaVA-OneVision-1.5 from: {self.model_path}")

        self._patch_transformers_compat()

        config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        if hasattr(config, 'text_config'):
            self.hidden_dim = config.text_config.hidden_size
            self.num_layers = config.text_config.num_hidden_layers
            self.num_heads = config.text_config.num_attention_heads

        if hasattr(config, 'vision_config'):
            self.visual_dim = config.text_config.hidden_size

        self.model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            attn_implementation='eager',
            torch_dtype=torch.float16,
        ).to(self.device)

        print("Loading and remapping checkpoint weights...")
        checkpoint_state_dict = {}
        for shard_file in sorted(glob.glob(os.path.join(self.model_path, "*.safetensors"))):
            shard = load_file(shard_file)
            checkpoint_state_dict.update(shard)

        remapped_state_dict = self._remap_checkpoint_keys(checkpoint_state_dict)

        load_result = self.model.load_state_dict(remapped_state_dict, strict=False)
        if load_result.missing_keys:
            print(f"Warning: Missing keys ({len(load_result.missing_keys)}): {load_result.missing_keys[:5]}...")
        if load_result.unexpected_keys:
            print(f"Warning: Unexpected keys ({len(load_result.unexpected_keys)}): {load_result.unexpected_keys[:5]}...")
        print(f"Weights loaded: {len(remapped_state_dict)} keys remapped from {len(checkpoint_state_dict)} checkpoint keys")

        self._fix_visual_nan()

        self.model = self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        try:
            from models.qwen2_5_vl_inference import process_vision_info
            self.process_vision_info = process_vision_info
        except ImportError:
            try:
                from models.qwen3_vl_inference import process_vision_info
                self.process_vision_info = process_vision_info
            except ImportError:
                self.process_vision_info = None

        print(f"LLaVA-OV-1.5 config: hidden_dim={self.hidden_dim}, "
              f"num_layers={self.num_layers}, num_heads={self.num_heads}, "
              f"visual_dim={self.visual_dim}")

        self.freeze_model()

    def _fix_visual_nan(self):
        nan_fixed = 0
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                with torch.no_grad():
                    param.zero_()
                    param.normal_(mean=0.0, std=0.02)
                nan_fixed += 1
        if nan_fixed > 0:
            print(f"Fixed {nan_fixed} NaN parameter tensors with random initialization")

    @staticmethod
    def _patch_transformers_compat():
        import sys

        from transformers import configuration_utils
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
            def auto_docstring(cls_or_fn=None):
                if cls_or_fn is not None:
                    return cls_or_fn
                return lambda x: x
            utils.auto_docstring = auto_docstring

        keys_to_remove = [
            k for k in sys.modules
            if 'LLaVA-OneVision-1.5' in k or 'llavaonevision' in k.lower()
        ]
        for k in keys_to_remove:
            del sys.modules[k]

    def _prepare_inputs(self, image: Image.Image, text: str, answer: str = None) -> Dict:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text}
                ]
            }
        ]

        if answer is not None:
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            })

        add_generation_prompt = (answer is None)

        try:
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        except (AttributeError, Exception) as e:
            print(f"Warning: apply_chat_template failed ({e}), using fallback prompt")
            prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" + text + "<|im_end|>\n<|im_start|>assistant\n"
            if answer is not None:
                prompt += answer

        if self.process_vision_info is not None:
            image_inputs, video_inputs = self.process_vision_info(messages)
        else:
            image_inputs, video_inputs = [image], None

        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True
        )

        return inputs.to(self.device)


    def generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 256,
        **kwargs
    ) -> str:
        inputs = self._prepare_inputs(image, question)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return response


    def _get_decoder_layers(self):
        try:
            return self.model.model.language_model.layers
        except AttributeError:
            return self.model.model.layers

    def _get_lm_head(self):
        return self.model.lm_head

    def _get_final_norm(self):
        try:
            return self.model.model.language_model.norm
        except AttributeError:
            return self.model.model.norm
