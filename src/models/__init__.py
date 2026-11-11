from importlib import import_module
from src.config import get_model_config


MODEL_CLASSES = {
    'qwen3_vl': ('qwen3', 'Qwen3VLModel'),
    'qwen3_vl_32b': ('qwen3', 'Qwen3VLModel'),
    'llava_ov1_5_8b': ('llava', 'LlavaOV1_5Model'),
    'gemma3_4b': ('gemma3', 'Gemma3Model'),
    'gemma3_27b': ('gemma3', 'Gemma3Model'),
}


def load_model(model_name, model_path=None, device='cuda', **kwargs):
    config = get_model_config(model_name)
    import transformers.processing_utils
    module_name, class_name = MODEL_CLASSES[config.name]
    model_class = getattr(import_module(f'src.models.{module_name}'), class_name)
    return model_class(model_path=model_path or config.model_path, device=device,
                       model_name=config.name, **kwargs)
