#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.stas import paper_profile, router_alignment_loss
from src.router import MultiLayerTokenRouter, TOKEN_FEATURE_PROTOCOL, prepare_router_token_cache, collate_router_tokens
from src.config import get_model_config, get_directories


def train_visual_router(model_name, num_components=10, epochs=200, lr=1e-3,
                        batch_size=32, device='cuda', mlp_hidden_dim=256,
                        max_samples=3000, use_cache=True):
    config = get_model_config(model_name)
    model_name = config.name
    target_layers, _ = paper_profile(model_name)
    if epochs < 1 or batch_size < 1 or max_samples < 1:
        raise ValueError('epochs, batch_size and max_samples must be positive')
    pca_dir = Path(get_directories(model_name)['checkpoint_dir']) / 'pca_directions'
    metadata_bytes = (pca_dir / 'stas_metadata.json').read_bytes()
    metadata = json.loads(metadata_bytes)
    if metadata.get('representation') != 'decoder_output_continuation_mean' or metadata.get('model') != model_name:
        raise ValueError('PCA metadata does not match the paper representation and selected model')
    pca_coeffs = torch.load(pca_dir / 'pca_coeffs.pt', map_location='cpu')
    token_paths, input_dim = prepare_router_token_cache(model_name, pca_dir, device=device, use_cache=use_cache)
    if input_dim != config.hidden_dim:
        raise ValueError('Native token dimension does not match the selected VLM')
    if len(token_paths) < max_samples:
        raise ValueError(f'Need {max_samples} paired samples, found {len(token_paths)}')
    if not set(target_layers).issubset(pca_coeffs):
        raise ValueError('PCA cache is missing paper intervention layers')
    if any(pca_coeffs[layer].shape != (len(token_paths), num_components) for layer in target_layers):
        raise ValueError('PCA coefficients do not match token-cache rows or num_components')
    router = MultiLayerTokenRouter(input_dim, target_layers, num_components, mlp_hidden_dim).to(device)
    optimizer = torch.optim.Adam(router.parameters(), lr=lr)
    best_loss, best_state, wait, patience = float('inf'), None, 0, 20
    print(f'Training {model_name} token router: samples={max_samples}, batch_size={batch_size}')
    for epoch in range(epochs):
        router.train()
        order = torch.randperm(max_samples)
        total_loss = 0.0
        for start in range(0, max_samples, batch_size):
            indices = order[start:start + batch_size]
            samples = [torch.load(token_paths[i], map_location='cpu') for i in indices.tolist()]
            tokens = {key: value.to(device) for key, value in collate_router_tokens(samples).items()}
            loss = sum(router_alignment_loss(
                router(layer, tokens['visual'], tokens['question'], tokens['visual_mask'], tokens['question_mask']),
                pca_coeffs[layer][indices].to(device)) for layer in target_layers) / len(target_layers)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        average_loss = total_loss / max_samples
        print(f'Epoch {epoch + 1}: loss={average_loss:.6f}')
        if average_loss < best_loss:
            best_loss, wait = average_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in router.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise ValueError('Training did not produce a finite loss')
    router.load_state_dict(best_state)
    router.eval()
    torch.save({'state_dict': best_state, 'config': {
        **router.config(), 'pca_metadata_sha256': hashlib.sha256(metadata_bytes).hexdigest(),
        'training': {'samples': max_samples, 'batch_size': batch_size, 'epochs': epochs,
                     'lr': lr, 'patience': patience, 'best_loss': best_loss},
    }}, pca_dir / 'router_token_mlp.pt')
    print(f'Saved router to {pca_dir / "router_token_mlp.pt"}')
    return router


def load_visual_router(model_name, device='cuda'):
    config = get_model_config(model_name)
    pca_dir = Path(get_directories(model_name)['checkpoint_dir']) / 'pca_directions'
    path = pca_dir / 'router_token_mlp.pt'
    if not path.is_file():
        raise FileNotFoundError(f'Missing STAS token router: {path}. No automatic training is performed.')
    checkpoint = torch.load(path, map_location=device)
    saved = checkpoint['config']
    if saved.get('type') != 'token_mlp' or saved.get('feature_protocol') != TOKEN_FEATURE_PROTOCOL:
        raise ValueError('Checkpoint is not a paper token router')
    metadata_hash = hashlib.sha256((pca_dir / 'stas_metadata.json').read_bytes()).hexdigest()
    if saved.get('coefficient_normalization') != 'signed_l2' or saved.get('pca_metadata_sha256') != metadata_hash:
        raise ValueError('Router does not match the current PCA cache')
    if saved['input_dim'] != config.hidden_dim:
        raise ValueError('Token router dimension does not match the selected VLM')
    router = MultiLayerTokenRouter(saved['input_dim'], saved['target_layers'],
                                  saved['num_components'], saved['mlp_hidden_dim']).to(device)
    router.load_state_dict(checkpoint['state_dict'])
    return router.eval()


def main():
    parser = argparse.ArgumentParser(description='Train the STAS token router')
    parser.add_argument('--model', default='qwen3_vl')
    parser.add_argument('--num_components', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_samples', type=int, default=3000)
    parser.add_argument('--mlp_hidden_dim', type=int, default=256)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no_cache', action='store_true')
    args = vars(parser.parse_args())
    args['model_name'] = args.pop('model')
    args['use_cache'] = not args.pop('no_cache')
    train_visual_router(**args)


if __name__ == '__main__':
    main()
