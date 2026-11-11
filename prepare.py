#!/usr/bin/env python3
import os, sys, json, torch, io, base64, hashlib
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models import load_model
from src.config import get_model_config, get_directories
from src.stas import paper_profile


def compute_pca_directions(model_name, data_path, max_samples=3000, num_components=10, device='cuda'):
    print(f"\n{'='*60}")
    print(f"Computing PCA contrastive directions: {model_name}")
    print(f"  max_samples={max_samples}, num_components={num_components}")
    print(f"{'='*60}")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)[:max_samples]
    print(f"Using {len(data)} samples from {data_path}")

    config = get_model_config(model_name)
    model_name = config.name
    target_layers, _ = paper_profile(model_name)
    if len(data) < max_samples:
        raise ValueError(f'Need {max_samples} paired samples, found {len(data)}')
    model = load_model(model_name, device=device)
    print(f'Target layers: {target_layers}')

    layer_pos = {l: [] for l in target_layers}
    layer_neg = {l: [] for l in target_layers}

    sample_indices = []

    try:
        for i, item in enumerate(tqdm(data, desc=f"Collecting {model_name}")):
            if 'image_bytes' in item and item['image_bytes']:
                img_bytes = item['image_bytes']
                if isinstance(img_bytes, str):
                    img_bytes = base64.b64decode(img_bytes)
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            elif 'image_path' in item and item['image_path']:
                image = Image.open(item['image_path']).convert('RGB')
            else:
                raise ValueError(f'Sample {i} is missing its image')

            question = item['question']
            pos_answer = item['positive_answer']
            neg_answer = item['negative_answer']

            with torch.no_grad():
                hs_pos = model.get_description_hidden_states(image, question, target_layers, answer=pos_answer)
                hs_neg = model.get_description_hidden_states(image, question, target_layers, answer=neg_answer)

                sample_indices.append(i)

            if not set(target_layers).issubset(set(hs_pos) & set(hs_neg)):
                raise ValueError(f'Sample {i} is missing decoder-layer states')
            for l in target_layers:
                layer_pos[l].append(hs_pos[l].cpu().float())
                layer_neg[l].append(hs_neg[l].cpu().float())

            if i % 50 == 0:
                torch.cuda.empty_cache()
    finally:
        model.cleanup()
        torch.cuda.empty_cache()

    if len(sample_indices) <= num_components:
        raise ValueError("Centered PCA needs more samples than num_components")
    print(f"Collected {len(sample_indices)} paired descriptions")

    pca_basis = {}
    pca_coeffs = {}
    mean_directions_raw = {}
    pca_stats = {}

    print(f"\nComputing SVD per layer...")
    for l in tqdm(target_layers):
        if not layer_pos[l]:
            continue

        pos_tensor = torch.cat(layer_pos[l], dim=0)
        neg_tensor = torch.cat(layer_neg[l], dim=0)
        delta_h = pos_tensor - neg_tensor

        n_samples = delta_h.shape[0]
        K = min(num_components, n_samples - 1, delta_h.shape[1])

        mean_dir = delta_h.mean(dim=0)
        mean_directions_raw[l] = mean_dir

        delta_h_centered = delta_h - mean_dir.unsqueeze(0)

        delta_h_gpu = delta_h_centered.to(device)
        U, S, Vt = torch.linalg.svd(delta_h_gpu, full_matrices=False)

        basis = Vt[:K].cpu()
        pca_basis[l] = basis

        coeffs = (delta_h_gpu @ Vt[:K].T).cpu()
        pca_coeffs[l] = coeffs

        S_cpu = S[:K].cpu()
        total_var = (S ** 2).sum().item()
        explained_var_ratio = [(s.item() ** 2 / max(total_var, 1e-12)) for s in S_cpu]

        del delta_h_gpu, U, S, Vt

        pca_stats[l] = {
            'explained_variance_ratio': explained_var_ratio,
            'singular_values': [s.item() for s in S_cpu],
            'n_samples': n_samples,
            'total_variance': total_var,
        }

    save_dir = os.path.join(get_directories(model_name)['checkpoint_dir'], 'pca_directions')
    os.makedirs(save_dir, exist_ok=True)

    torch.save(pca_basis, os.path.join(save_dir, 'pca_basis.pt'))
    torch.save(pca_coeffs, os.path.join(save_dir, 'pca_coeffs.pt'))
    torch.save(mean_directions_raw, os.path.join(save_dir, 'mean_directions_raw.pt'))
    with open(os.path.join(save_dir, 'pca_basis.pt'), 'rb') as basis_file:
        basis_hash = hashlib.sha256(basis_file.read()).hexdigest()
    with open(os.path.join(save_dir, 'mean_directions_raw.pt'), 'rb') as mean_file:
        mean_hash = hashlib.sha256(mean_file.read()).hexdigest()
    with open(data_path, 'rb') as source:
        data_hash = hashlib.sha256(source.read()).hexdigest()
    with open(os.path.join(save_dir, 'stas_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump({'version': 1, 'model': model_name,
                   'data_path': os.path.abspath(data_path), 'data_sha256': data_hash,
                   'basis_sha256': basis_hash, 'mean_sha256': mean_hash,
                   'sample_indices': sample_indices, 'num_components': num_components,
                   'representation': 'decoder_output_continuation_mean',
                   'direction': 'faithful_minus_hallucinated'}, f, indent=2)

    with open(os.path.join(save_dir, 'pca_stats.json'), 'w') as f:
        json.dump(pca_stats, f, indent=2)

    print(f"\nPCA Direction Stats for {model_name}:")
    print(f"{'Layer':<6} {'PC1%':>6} {'PC1-3%':>7} {'PC1-5%':>8} {'PC1-10%':>8} {'1-PC1%':>7} {'N':>4}")
    print("-" * 60)

    for l in target_layers:
        if l not in pca_stats:
            continue
        s = pca_stats[l]
        evr = s['explained_variance_ratio']
        total_evr_full = sum(evr)
        pc1 = evr[0] * 100
        pc3 = sum(evr[:min(3, len(evr))]) * 100
        pc5 = sum(evr[:min(5, len(evr))]) * 100
        pc10 = sum(evr[:min(10, len(evr))]) * 100
        dyn = 100 - pc1

        if l in [0, 5, 10, 15, 17, 20, 25, 30, 35]:
            print(f"{l:<6} {pc1:>5.1f}% {pc3:>6.1f}% {pc5:>7.1f}% {pc10:>7.1f}% {dyn:>6.1f}% {s['n_samples']:>4}")

    print(f"\nSaved to {save_dir}")
    return pca_basis, pca_coeffs, pca_stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Construct the STAS subspace from full paired descriptions')
    parser.add_argument('--model', default='qwen3_vl')
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--max_samples', type=int, default=3000)
    parser.add_argument('--num_components', type=int, default=10)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    compute_pca_directions(args.model, args.data_path, args.max_samples, args.num_components, args.device)


if __name__ == '__main__':
    main()
