import os
from dataclasses import dataclass
from typing import List
from src.stas import paper_profile


@dataclass
class ModelConfig:
    name: str
    model_path: str
    hidden_dim: int
    num_layers: int
    num_heads: int
    visual_dim: int
    default_target_layers: List[int]
    description: str = ""


MODEL_CONFIGS = {
    'qwen3_vl': ModelConfig(name='qwen3_vl', model_path='/mnt/r-contentsecurity-p/common/checkpoints/opensources/Qwen3-VL-8B-Instruct', hidden_dim=4096, num_layers=36, num_heads=32, visual_dim=4096, default_target_layers=paper_profile('qwen3_vl')[0], description='Qwen3-VL 8B Instruct'),
    'llava_ov1_5_8b': ModelConfig(name='llava_ov1_5_8b', model_path='/mnt/r-contentsecurity-p/common/datas_yl/chizhu/models/LLaVA-OneVision-1.5-8B-Instruct', hidden_dim=4096, num_layers=36, num_heads=32, visual_dim=4096, default_target_layers=paper_profile('llava_ov1_5_8b')[0], description='LLaVA-OneVision-1.5 8B'),
    'gemma3_4b': ModelConfig(name='gemma3_4b', model_path='/mnt/r-contentsecurity-p/common/datas_yl/chizhu/models/gemma-3-4b-it', hidden_dim=2560, num_layers=34, num_heads=8, visual_dim=2560, default_target_layers=paper_profile('gemma3_4b')[0], description='Gemma3 4B IT'),
    'qwen3_vl_32b': ModelConfig(name='qwen3_vl_32b', model_path='/mnt/r-contentsecurity-p/common/checkpoints/opensources/Qwen3-VL-32B-Instruct', hidden_dim=5120, num_layers=64, num_heads=64, visual_dim=5120, default_target_layers=paper_profile('qwen3_vl_32b')[0], description='Qwen3-VL 32B Instruct'),
    'gemma3_27b': ModelConfig(name='gemma3_27b', model_path='/mnt/r-contentsecurity-p/common/checkpoints/opensources/gemma-3-27b-it', hidden_dim=5376, num_layers=62, num_heads=32, visual_dim=5376, default_target_layers=paper_profile('gemma3_27b')[0], description='Gemma3 27B IT')
}

MODEL_ALIASES = {'llava_ov1_5': 'llava_ov1_5_8b', 'gemma3': 'gemma3_4b'}


def get_model_config(model_name):
    name = model_name.lower().replace('-', '_').replace('.', '_')
    name = MODEL_ALIASES.get(name, name)
    if name not in MODEL_CONFIGS:
        raise ValueError(f'Unsupported paper model: {model_name}. Choose from {list(MODEL_CONFIGS)}')
    return MODEL_CONFIGS[name]


def get_directories(model_name, base_dir=None):
    name = get_model_config(model_name).name
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {'checkpoint_dir': os.path.join(base_dir, 'checkpoints', name),
            'result_dir': os.path.join(base_dir, 'results', name)}
