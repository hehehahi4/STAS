<div align="center">
  
  <h1>STAS: Subspace- and Token-Adaptive Steering for Mitigating Hallucinations in Large Vision-Language Models</h1>

</div>

[![arXiv](https://img.shields.io/badge/arXiv-2611-red.svg)]()

## 🐘 Introduction
Despite demonstrating impressive multimodal capabilities, Large Vision-Language Models (LVLMs) frequently suffer from hallucinations. Activation steering is an inference-time intervention technique that mitigates hallucinations by identifying their direction in the hidden states and manipulating the model to shift away from that direction. However, existing methods typically rely on either a static global direction or an entangled monolithic vector with a uniform steering strength across all tokens, which is imprecise and inflexible when confronting diverse hallucinations. To address these limitations, we propose Subspace- and Token-Adaptive Steering (STAS), a framework that dynamically synthesizes specific steering directions and scales steering strength. Specifically, STAS first constructs a low-dimensional orthogonal subspace to disentangle diverse hallucination patterns. A lightweight cross-modal router is then trained to predict projection coefficients and synthesize input-specific steering directions within this subspace. Furthermore, we introduce a token-adaptive strength scaling mechanism by measuring uncertainty inside the model. This ensures that strong interventions are selectively applied to high-risk semantic tokens without disrupting the syntactic fluency of low-risk tokens. Extensive experiments demonstrate that STAS outperforms existing baselines across multiple architectures and benchmarks.


## 🎭 Getting Started

### 🌲 Env
Create conda virtual environment and install requirements:
```
python -m pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng wordnet
```


### 📔 Data

Download [POPE](https://huggingface.co/datasets/lmms-lab-encoder/POPE/tree/main/Full), [MME](https://huggingface.co/datasets/lmms-lab-encoder/MME/tree/main/data), [AMBER](https://github.com/junyangwang0410/AMBER), and [COCO 2014](https://cocodataset.org/#download) with the [CHAIR metadata](https://github.com/LisaAnne/Hallucination). Arrange the files as follows:

```text
data/
  rah_pairs.json
  POPE/Full/*.parquet
  MME/data/*.parquet
  AMBER/data/
  AMBER/image/
  CHIAR/val2014/
  CHIAR/annotations_trainval2014/annotations/
  CHIAR/Hallucination-master/data/synonyms.txt
```

The current loader expects the folder spelling `CHIAR`.

For training, prepare 3,000 [RAH-Bench](https://arxiv.org/abs/2311.16479) response pairs in `data/rah_pairs.json`. Each JSON record needs `image_path`, `question`, `positive_answer`, and `negative_answer`. Answers must contain full model-generated descriptions after forced correct/incorrect Yes/No prefixes. Pair generation is not included in this repository.

### 💪🏾 Train

Run from the project directory. Download the model weights and set their paths in `src/config.py`. The commands below use Qwen3-VL-8B.

```bash
python prepare.py --model qwen3_vl --data_path data/rah_pairs.json
python train.py --model qwen3_vl
```

### ✒️ Evaluate

```bash
python evaluate.py --model qwen3_vl --dataset POPE
python evaluate.py --model qwen3_vl --dataset MME
python evaluate.py --model qwen3_vl --dataset AMBER
python evaluate.py --model qwen3_vl --dataset CHAIR
```

By default, evaluation runs both the unsteered model and STAS. Add `--mode stas` to run STAS only. Results are saved to `results/qwen3_vl/stas_token_mlp/`.


## 🙏 Acknowledgements


## 🙏 Citation
If you think our work helpful, please cite our paper. Thank you very much!
