import torch
import torch.nn as nn
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.base import STASModelBase


class Qwen3VLModel(STASModelBase):

    DEFAULT_PATH = "/mnt/r-contentsecurity-p/common/checkpoints/opensources/Qwen3-VL-8B-Instruct"

    def __init__(self, model_path: str = None, device: str = 'cuda', **kwargs):
        super().__init__(model_path or self.DEFAULT_PATH, device, **kwargs)

        self.hidden_dim = 4096
        self.num_layers = 36
        self.num_heads = 32
        self.visual_dim = 4096

        self.load_model()

    def load_model(self):
        self._patch_transformers_compat()
        self._patch_maybe_autocast()

        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError:
            from models.qwen3_vl import Qwen3VLForConditionalGeneration

        self._use_qwen3_processor = False
        try:
            from transformers import AutoTokenizer, Qwen2VLImageProcessorFast
            from models.qwen3_vl import Qwen3VLProcessor
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            image_processor = Qwen2VLImageProcessorFast.from_pretrained(self.model_path)
            self.processor = Qwen3VLProcessor(
                image_processor=image_processor, tokenizer=tokenizer
            )
            self._use_qwen3_processor = True
            print("Using Qwen3VLProcessor (native mm_token_type_ids support)")
        except (ImportError, Exception) as e:
            print(f"Qwen3VLProcessor not available ({e}), using Qwen2_5_VLProcessor + manual mm_token_type_ids")
            from transformers import Qwen2_5_VLProcessor
            self.processor = Qwen2_5_VLProcessor.from_pretrained(self.model_path)

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map='auto'
        ).eval()

        if hasattr(self.model.config, 'text_config'):
            self.hidden_dim = self.model.config.text_config.hidden_size
            self.num_layers = self.model.config.text_config.num_hidden_layers
            self.num_heads = self.model.config.text_config.num_attention_heads
        else:
            self.hidden_dim = self.model.config.hidden_size
            self.num_layers = self.model.config.num_hidden_layers
            self.num_heads = self.model.config.num_attention_heads

        if hasattr(self.model.config, 'vision_config') and hasattr(self.model.config.vision_config, 'out_hidden_size'):
            self.visual_dim = self.model.config.vision_config.out_hidden_size


        print(f"Qwen3-VL config: hidden_dim={self.hidden_dim}, num_layers={self.num_layers}, "
              f"num_heads={self.num_heads}, visual_dim={self.visual_dim}")

        if not self._use_qwen3_processor:
            if 'mm_token_type_ids' not in getattr(self.model, '_model_input_names', []):
                self.model._model_input_names = list(
                    getattr(self.model, '_model_input_names', [])
                ) + ['mm_token_type_ids']

        self._patch_prepare_inputs_for_generation()

        self.freeze_model()

    def _prepare_inputs(self, image: Image.Image, text: str, answer: str = None, system_prompt: str = None) -> Dict:
        if system_prompt is None:
            system_prompt = "/no_think"
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image, "min_pixels": 65536, "max_pixels": 16777216},
                {"type": "text", "text": text}
            ]
        })

        if answer is not None:
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            })

        add_generation_prompt = (answer is None)

        try:
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        except (AttributeError, Exception) as e:
            print(f"Warning: apply_chat_template failed ({e}), using fallback prompt")
            text_prompt = (
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                + text + "<|im_end|>\n<|im_start|>assistant\n"
            )
            if answer is not None:
                text_prompt += answer

        if self._use_qwen3_processor:
            self.processor.image_processor.size["longest_edge"] = 16777216
            inputs = self.processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
        else:
            try:
                from models.qwen3_vl_inference import process_vision_info
            except ImportError:
                try:
                    from models.qwen2_5_vl_inference import process_vision_info
                except ImportError:
                    process_vision_info = None

            if process_vision_info is not None:
                vision_messages = []
                for msg in messages:
                    if isinstance(msg.get("content"), list):
                        vision_messages.append(msg)
                image_inputs, video_inputs = process_vision_info(vision_messages) if vision_messages else ([image], None)
            else:
                image_inputs, video_inputs = [image], None

            self.processor.image_processor.size["longest_edge"] = 16777216
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )

            if 'mm_token_type_ids' not in inputs and 'image_grid_thw' in inputs:
                input_ids = inputs['input_ids']
                vision_start_id = self.processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
                vision_end_id = self.processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
                image_pad_id = self.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
                video_pad_id = self.processor.tokenizer.convert_tokens_to_ids('<|video_pad|>')

                batch_mm_token_type_ids = []
                for batch_idx in range(input_ids.shape[0]):
                    ids = input_ids[batch_idx].tolist()
                    mm_token_type_ids = [0] * len(ids)
                    in_vision = False
                    for i, token_id in enumerate(ids):
                        if token_id == vision_start_id:
                            in_vision = True
                        elif token_id == vision_end_id:
                            in_vision = False
                        elif in_vision:
                            if token_id == image_pad_id:
                                mm_token_type_ids[i] = 1
                            elif token_id == video_pad_id:
                                mm_token_type_ids[i] = 2
                    batch_mm_token_type_ids.append(mm_token_type_ids)

                inputs['mm_token_type_ids'] = torch.tensor(batch_mm_token_type_ids, dtype=torch.int32)

        return inputs.to(self.device)

    @staticmethod
    def _patch_maybe_autocast():
        from contextlib import nullcontext
        import inspect as _inspect

        try:
            from transformers.utils import generic as _generic
        except ImportError:
            return

        if hasattr(_generic, 'maybe_autocast'):
            try:
                sig = _inspect.signature(_generic.maybe_autocast)
                if 'device_type' in sig.parameters:
                    return
            except (ValueError, TypeError):
                pass

        def maybe_autocast(device_type=None, enabled=True):
            if device_type is not None and enabled:
                try:
                    return torch.autocast(device_type=device_type, enabled=False)
                except Exception:
                    pass
            return nullcontext()

        _generic.maybe_autocast = maybe_autocast
        try:
            import transformers.utils as _utils
            _utils.maybe_autocast = maybe_autocast
        except Exception:
            pass

    def _patch_prepare_inputs_for_generation(self):
        import functools
        import inspect

        orig_prepare = self.model.prepare_inputs_for_generation
        sig = inspect.signature(orig_prepare)

        if 'is_first_iteration' not in sig.parameters:
            return

        self.model._qwen3_prepare_call_count = 0

        @functools.wraps(orig_prepare)
        def patched_prepare(*args, **kwargs):
            self.model._qwen3_prepare_call_count += 1
            kwargs['is_first_iteration'] = (self.model._qwen3_prepare_call_count == 1)
            return orig_prepare(*args, **kwargs)

        self.model.prepare_inputs_for_generation = patched_prepare

    @staticmethod
    def _patch_transformers_compat():
        import sys
        import inspect as _inspect

        from transformers import utils as _utils
        def auto_docstring(cls_or_fn=None, **kwargs):
            if cls_or_fn is not None:
                return cls_or_fn
            return lambda x: x
        _utils.auto_docstring = auto_docstring

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

        keys_to_remove = [
            k for k in sys.modules
            if 'LLaVA-OneVision-1.5' in k or 'llavaonevision' in k.lower() or 'qwen3_vl' in k.lower()
        ]
        for k in keys_to_remove:
            del sys.modules[k]


    def generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 256,
        system_prompt: str = None,
        **kwargs
    ) -> str:
        inputs = self._prepare_inputs(image, question, system_prompt=system_prompt)

        self.model._qwen3_prepare_call_count = 0

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        response = self._strip_thinking(response)

        return response

    @staticmethod
    def _strip_thinking(text: str) -> str:
        import re
        stripped = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return stripped.strip()


    def _get_decoder_layers(self):
        return self.model.model.language_model.layers

    def _get_lm_head(self):
        return self.model.lm_head

    def _get_final_norm(self):
        return self.model.model.language_model.norm
