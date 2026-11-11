# STAS

## 1. Download data

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

## 2. Train

Run from the project directory. Download the model weights and set their paths in `src/config.py`. The commands below use Qwen3-VL-8B.

```bash
python -m pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng wordnet
python prepare.py --model qwen3_vl --data_path data/rah_pairs.json
python train.py --model qwen3_vl
```

## 3. Evaluate

```bash
python evaluate.py --model qwen3_vl --dataset POPE
python evaluate.py --model qwen3_vl --dataset MME
python evaluate.py --model qwen3_vl --dataset AMBER
python evaluate.py --model qwen3_vl --dataset CHAIR
```

By default, evaluation runs both the unsteered model and STAS. Add `--mode stas` to run STAS only. Results are saved to `results/qwen3_vl/stas_token_mlp/`.
