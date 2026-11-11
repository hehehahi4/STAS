import base64
import hashlib
import io
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

from src.stas import normalized_coefficients

TOKEN_FEATURE_PROTOCOL = 'native_visual_question_tokens_v1'


class MultiLayerTokenRouter(nn.Module):
    def __init__(self, input_dim, target_layers, num_components=10, mlp_hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.target_layers = sorted(set(target_layers))
        self.num_components = num_components
        self.mlp_hidden_dim = mlp_hidden_dim
        self.router_type = 'token_mlp'
        self.feature_protocol = TOKEN_FEATURE_PROTOCOL
        if not self.target_layers or min(self.target_layers) < 0:
            raise ValueError('Router requires nonnegative target layers')
        self.routers = nn.ModuleDict({str(layer): nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim), nn.GELU(),
            nn.Linear(mlp_hidden_dim, num_components),
        ) for layer in self.target_layers})

    def forward(self, layer_idx, visual_tokens, question_tokens,
                visual_mask=None, question_mask=None):
        if visual_tokens.ndim != 3 or question_tokens.ndim != 3:
            raise ValueError('Expected unpooled token tensors with shape (batch, tokens, dim)')
        if visual_tokens.shape[0] != question_tokens.shape[0]:
            raise ValueError('Visual and question batch sizes differ')
        if visual_tokens.shape[-1] != self.input_dim or question_tokens.shape[-1] != self.input_dim:
            raise ValueError('Native visual and text features must share the router input dimension')
        if str(layer_idx) not in self.routers:
            raise ValueError(f'Router has no layer {layer_idx}')
        masks = []
        for tokens, mask in ((visual_tokens, visual_mask), (question_tokens, question_mask)):
            if mask is None:
                mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
            if mask.shape != tokens.shape[:2]:
                raise ValueError('Token mask shape does not match its sequence')
            mask = mask.to(device=tokens.device, dtype=torch.bool)
            if not mask.any(dim=1).all():
                raise ValueError('Each sample needs valid tokens from both modalities')
            masks.append(mask)
        tokens = torch.cat((visual_tokens, question_tokens), dim=1)
        mask = torch.cat(masks, dim=1)
        tokens = tokens.masked_fill(~mask.unsqueeze(-1), 0)
        logits = self.routers[str(layer_idx)](tokens)
        logits = logits.masked_fill(~mask.unsqueeze(-1), 0)
        return logits.sum(dim=1) / mask.sum(dim=1, keepdim=True).to(logits.dtype)

    def coefficients(self, *args, **kwargs):
        return normalized_coefficients(self(*args, **kwargs))

    def config(self):
        return {'type': self.router_type, 'input_dim': self.input_dim,
                'target_layers': self.target_layers, 'num_components': self.num_components,
                'mlp_hidden_dim': self.mlp_hidden_dim,
                'feature_protocol': self.feature_protocol,
                'coefficient_normalization': 'signed_l2'}


def _primary_visual_sequence(output):
    if hasattr(output, 'pooler_output') and output.pooler_output is not None:
        output = output.pooler_output
    elif isinstance(output, tuple):
        output = output[0]
    if isinstance(output, list):
        return torch.cat([_primary_visual_sequence(value) for value in output], dim=0)
    if not torch.is_tensor(output) or output.ndim not in (2, 3):
        raise ValueError('Native vision output is not a patch-token sequence')
    return output.reshape(-1, output.shape[-1])


def extract_router_tokens(adapter, image, question):
    model = adapter.model
    vision = None
    for path in ('model.visual', 'visual', 'model.multi_modal_projector',
                 'multi_modal_projector', 'model.embed_vision', 'embed_vision'):
        try:
            vision = model.get_submodule(path)
            break
        except AttributeError:
            pass
    if vision is None:
        raise NotImplementedError('No native visual token output found for this model adapter')
    tokenizer = getattr(getattr(adapter, 'processor', None), 'tokenizer', None) or getattr(adapter, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError('A text tokenizer is required for question-only embeddings')
    inputs = adapter._prepare_inputs(image, question)
    captured = []

    class FeaturesReady(Exception):
        pass

    def capture(module, args, output):
        captured.append(_primary_visual_sequence(output).detach().clone())

    def stop_before_decoder(module, args, kwargs):
        raise FeaturesReady()

    handles = []
    try:
        handles.append(vision.register_forward_hook(capture))
        handles.append(adapter._get_decoder_layers()[0].register_forward_pre_hook(
            stop_before_decoder, with_kwargs=True))
        with torch.no_grad():
            try:
                model(**{**inputs, 'use_cache': False})
            except FeaturesReady:
                pass
            if not captured:
                raise ValueError('The native forward did not produce visual tokens')
            visual = torch.cat(captured, dim=0).unsqueeze(0).float()
            embedding = model.get_input_embeddings()
            encoded = tokenizer(question, add_special_tokens=False, return_tensors='pt')
            ids = encoded['input_ids'].to(embedding.weight.device)
            if ids.shape[1] == 0:
                raise ValueError('Question must contain at least one token')
            text = embedding(ids).detach().float()
            if visual.shape[-1] != text.shape[-1]:
                raise ValueError('Native projected vision and question embedding dimensions differ')
            if not torch.isfinite(visual).all() or not torch.isfinite(text).all():
                raise ValueError('Non-finite native router features')
            return visual, text
    finally:
        for handle in handles:
            handle.remove()


def collate_router_tokens(samples):
    result = {}
    for name in ('visual', 'question'):
        sequences = [sample[name] for sample in samples]
        result[name] = pad_sequence(sequences, batch_first=True)
        lengths = torch.tensor([len(seq) for seq in sequences])
        result[name + '_mask'] = torch.arange(result[name].shape[1])[None, :] < lengths[:, None]
    return result


def prepare_router_token_cache(model_name, pca_dir, device='cuda', use_cache=True):
    pca_dir = Path(pca_dir)
    metadata_bytes = (pca_dir / 'stas_metadata.json').read_bytes()
    metadata = json.loads(metadata_bytes)
    source_bytes = Path(metadata['data_path']).read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != metadata['data_sha256']:
        raise ValueError('Source data changed since PCA extraction')
    cache_id = hashlib.sha256(metadata_bytes + TOKEN_FEATURE_PROTOCOL.encode()).hexdigest()[:16]
    cache_dir = pca_dir / 'router_tokens' / cache_id
    manifest_path = cache_dir / 'manifest.json'
    paths = [cache_dir / f'{i:06d}.pt' for i in range(len(metadata['sample_indices']))]
    if use_cache and manifest_path.exists() and all(path.exists() for path in paths):
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['protocol'] == TOKEN_FEATURE_PROTOCOL and manifest['model'] == model_name:
            return paths, manifest['input_dim']
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.unlink(missing_ok=True)
    from src.models import load_model
    model = load_model(model_name, device=device)
    data = json.loads(source_bytes)
    input_dim = None
    try:
        for path, source_index in zip(paths, metadata['sample_indices']):
            item = data[source_index]
            if item.get('image_bytes'):
                raw = item['image_bytes']
                raw = base64.b64decode(raw) if isinstance(raw, str) else raw
                image = Image.open(io.BytesIO(raw)).convert('RGB')
            else:
                image = Image.open(item['image_path']).convert('RGB')
            visual, question = extract_router_tokens(model, image, item['question'])
            if input_dim is not None and input_dim != visual.shape[-1]:
                raise ValueError('Inconsistent token feature dimensions in source samples')
            input_dim = visual.shape[-1]
            torch.save({'visual': visual[0].cpu(), 'question': question[0].cpu()}, path)
    finally:
        model.cleanup()
    if input_dim is None:
        raise ValueError('No PCA samples available for token extraction')
    manifest_path.write_text(json.dumps({'protocol': TOKEN_FEATURE_PROTOCOL,
        'model': model_name, 'input_dim': input_dim, 'sample_count': len(paths)}, indent=2), encoding='utf-8')
    return paths, input_dim
