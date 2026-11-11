#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.data import load_pope_data, load_mme_data, load_amber_disc_data, parse_yes_no, MME_PERCEPTION_CATEGORIES, MME_COGNITION_CATEGORIES
from src.models import load_model
from src.stas import paper_profile
from src.config import get_model_config, get_directories
from train import load_visual_router

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _generate_stas(model, *args, **kwargs):
    return model.generate_with_stas(*args, **getattr(model, '_stas_options', {}), **kwargs)


def extract_nouns(text):
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        nouns = [lemmatizer.lemmatize(word.lower()) for word, pos in tagged if pos.startswith('NN')]
        return nouns
    except Exception:
        pass

    stop_words = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
        'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
        'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
        'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
        'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
        'about', 'against', 'between', 'through', 'during', 'before', 'after', 'above',
        'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
        't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're',
        've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
        'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
        'won', 'wouldn', 'also', 'there', 'here', 'yes', 'no', 'image', 'picture',
        'photo', 'photograph', 'visible', 'shown', 'appears', 'appearing', 'seen',
        'look', 'looks', 'looking', 'see', 'sees', 'show', 'shows', 'showing',
        'appear', 'would', 'could', 'might', 'shall', 'may', 'must', 'need',
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'many', 'much', 'several', 'every', 'another', 'whole', 'part', 'side',
        'top', 'bottom', 'left', 'right', 'middle', 'front', 'back', 'center',
        'background', 'foreground', 'overall', 'entire', 'entirely',
        'there', 'nothing', 'anything', 'something', 'everything',
    }
    common_verbs = {
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'can', 'shall', 'must', 'need', 'going', 'making', 'taking', 'giving',
        'getting', 'putting', 'setting', 'standing', 'sitting', 'walking',
        'running', 'moving', 'holding', 'carrying', 'wearing', 'looking',
        'appearing', 'seems', 'seem', 'feels', 'feel', 'makes', 'made',
        'sitting', 'lying', 'laying', 'placed', 'hangs', 'hanging',
    }

    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    nouns = []
    for token in tokens:
        if token in stop_words or token in common_verbs or len(token) <= 2:
            continue
        lemma = token
        if lemma.endswith('ies') and len(lemma) > 4:
            lemma = lemma[:-3] + 'y'
        elif lemma.endswith('es') and len(lemma) > 3:
            lemma = lemma[:-2]
        elif lemma.endswith('s') and not lemma.endswith('ss') and len(lemma) > 3:
            lemma = lemma[:-1]
        nouns.append(lemma)
    return nouns


def compute_amber_gen_metrics(results, annotations, relation_dict, safe_words_list):
    chair_score = 0
    chair_num = 0
    safe_cover_score = 0
    safe_cover_num = 0
    hallu_cover_score = 0
    hallu_cover_num = 0
    non_hallu_count = 0
    non_hallu_total = 0

    hallucination_words = set()
    for w1 in relation_dict:
        hallucination_words.add(w1.lower())
        for w2 in relation_dict[w1]:
            hallucination_words.add(w2.lower())

    global_safe = set(w.lower() for w in safe_words_list)

    for r in results:
        idx = r['id']
        annotation = annotations[idx - 1]
        if annotation.get('type') != 'generative':
            continue

        nouns = extract_nouns(r['response'])
        filtered_nouns = [n for n in nouns if n.lower() in hallucination_words]

        truth_words = [w.lower() for w in annotation.get('truth', [])]
        hallu_words = [w.lower() for w in annotation.get('hallu', [])]

        safe_expanded = list(truth_words)
        safe_flags = [0] * len(truth_words)
        for w in truth_words:
            if w in relation_dict:
                safe_expanded.extend(relation_dict[w])
                safe_flags.extend([0] * len(relation_dict[w]))

        hallu_expanded = list(hallu_words)
        hallu_flags = [0] * len(hallu_words)
        for w in hallu_words:
            if w in relation_dict:
                hallu_expanded.extend(relation_dict[w])
                hallu_flags.extend([0] * len(relation_dict[w]))

        safe_flag_list = [0] * len(filtered_nouns)
        for i, noun in enumerate(filtered_nouns):
            noun_lower = noun.lower()

            if noun_lower in global_safe:
                continue

            if noun_lower in safe_expanded:
                for j, sw in enumerate(safe_expanded):
                    if noun_lower == sw and j < len(safe_flags):
                        safe_flags[j] = 1
                        break
                continue

            if noun_lower in hallu_expanded:
                for j, hw in enumerate(hallu_expanded):
                    if noun_lower == hw and j < len(hallu_flags):
                        hallu_flags[j] = 1
                        break

            safe_flag_list[i] = 1

        chair_score += sum(safe_flag_list)
        chair_num += len(safe_flag_list)
        safe_cover_score += sum(safe_flags[:len(truth_words)])
        safe_cover_num += len(truth_words)
        hallu_cover_score += sum(hallu_flags[:len(hallu_words)])
        hallu_cover_num += len(hallu_words)

        if sum(safe_flag_list) == 0:
            non_hallu_count += 1
        non_hallu_total += 1

    eps = 1e-6
    chair = round(chair_score / max(chair_num, eps) * 100, 2)
    cover = round(safe_cover_score / max(safe_cover_num, eps) * 100, 2)
    hal = round(100 - non_hallu_count / max(non_hallu_total, eps) * 100, 2)
    cog = round(hallu_cover_score / max(hallu_cover_num, eps) * 100, 2)

    return {
        'CHAIR': chair,
        'Cover': cover,
        'Hal': hal,
        'Cog': cog,
        'total_samples': non_hallu_total
    }


def load_amber_gen_data(num_samples=None):
    amber_data_dir = os.path.join(DATA_ROOT, 'AMBER')
    amber_image_dir = os.path.join(amber_data_dir, 'image')
    query_file = os.path.join(amber_data_dir, 'data', 'query', 'query_generative.json')

    if not os.path.exists(query_file):
        print(f"Warning: AMBER generative query file not found: {query_file}")
        return []

    queries = json.load(open(query_file))
    queries = [q for q in queries if q['id'] < 1005]
    if num_samples:
        queries = queries[:num_samples]

    all_data = []
    for query in queries:
        image_file = query['image']
        question = query['query']
        image_path = os.path.join(amber_image_dir, image_file)

        if not os.path.exists(image_path):
            continue

        all_data.append({
            'id': query['id'],
            'image_path': image_path,
            'question': question,
        })

    return all_data


def load_chair_data(num_samples=None):
    coco_ann_dir = os.path.join(DATA_ROOT, 'CHIAR', 'annotations_trainval2014', 'annotations')
    coco_image_dir = os.path.join(DATA_ROOT, 'CHIAR', 'val2014')
    synonyms_file = os.path.join(DATA_ROOT, 'CHIAR', 'Hallucination-master', 'data', 'synonyms.txt')

    from PIL import Image

    with open(os.path.join(coco_ann_dir, 'instances_val2014.json'), 'r') as f:
        coco_instances = json.load(f)

    id_to_name = {c['id']: c['name'] for c in coco_instances['categories']}
    id_to_filename = {img['id']: img['file_name'] for img in coco_instances['images']}

    img_to_gt_objects = defaultdict(set)
    for ann in coco_instances['annotations']:
        cat_name = id_to_name.get(ann['category_id'])
        if cat_name:
            img_to_gt_objects[ann['image_id']].add(cat_name)

    captions_file = os.path.join(coco_ann_dir, 'captions_val2014.json')
    if os.path.exists(captions_file):
        with open(captions_file, 'r') as f:
            coco_captions = json.load(f)
        mscoco_objects_set, inverse_synonym_dict = _load_chair_synonyms(synonyms_file)
        coco_double_words, animal_words, vehicle_words = _get_chair_double_words()
        double_word_dict = _build_double_word_dict(coco_double_words, animal_words, vehicle_words)

        for ann in coco_captions['annotations']:
            imid = ann['image_id']
            caption_objects = _caption_to_objects(
                ann['caption'], mscoco_objects_set, inverse_synonym_dict,
                double_word_dict
            )
            img_to_gt_objects[imid].update(caption_objects)

    mscoco_objects_set, inverse_synonym_dict = _load_chair_synonyms(synonyms_file)
    coco_double_words, animal_words, vehicle_words = _get_chair_double_words()
    double_word_dict = _build_double_word_dict(coco_double_words, animal_words, vehicle_words)

    all_image_ids = sorted(img_to_gt_objects.keys())
    if num_samples:
        all_image_ids = all_image_ids[:num_samples]

    prompt = "Describe this image in detail."

    data = []
    for img_id in all_image_ids:
        filename = id_to_filename.get(img_id)
        if filename is None:
            continue
        img_path = os.path.join(coco_image_dir, filename)
        if not os.path.exists(img_path):
            continue

        data.append({
            'image_id': img_id,
            'image_path': img_path,
            'question': prompt,
            'gt_objects': img_to_gt_objects[img_id],
        })

    gt_objects = {img_id: img_to_gt_objects[img_id] for img_id in all_image_ids}

    chair_helpers = {
        'mscoco_objects_set': mscoco_objects_set,
        'inverse_synonym_dict': inverse_synonym_dict,
        'double_word_dict': double_word_dict,
    }

    return data, gt_objects, chair_helpers


def _load_chair_synonyms(synonyms_file):
    mscoco_objects = []
    inverse_synonym_dict = {}
    with open(synonyms_file, 'r') as f:
        for line in f:
            synonyms = line.strip().split(', ')
            mscoco_objects.extend(synonyms)
            for s in synonyms:
                inverse_synonym_dict[s] = synonyms[0]
    return set(mscoco_objects), inverse_synonym_dict


def _get_chair_double_words():
    coco_double_words = [
        'motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light',
        'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter',
        'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket',
        'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear',
        'hair drier', 'potted plant', 'bow tie', 'laptop computer', 'stove top oven',
        'home plate', 'train track'
    ]
    animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']
    vehicle_words = ['jet', 'train']
    return coco_double_words, animal_words, vehicle_words


def _build_double_word_dict(coco_double_words, animal_words, vehicle_words):
    double_word_dict = {}
    for dw in coco_double_words:
        double_word_dict[dw] = dw
    for aw in animal_words:
        double_word_dict[f'baby {aw}'] = aw
        double_word_dict[f'adult {aw}'] = aw
    for vw in vehicle_words:
        double_word_dict[f'passenger {vw}'] = vw
    double_word_dict['bow tie'] = 'tie'
    double_word_dict['toilet seat'] = 'toilet'
    double_word_dict['wine glas'] = 'wine glass'
    return double_word_dict


def _caption_to_objects(caption, mscoco_objects_set, inverse_synonym_dict, double_word_dict):
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(caption.lower())
        words = [lemmatizer.lemmatize(w) for w in tokens]
    except Exception:
        words = re.findall(r'\b[a-zA-Z]+\b', caption.lower())

    i = 0
    processed_words = []
    while i < len(words):
        double_word = ' '.join(words[i:i+2]) if i + 1 < len(words) else ''
        if double_word in double_word_dict:
            processed_words.append(double_word_dict[double_word])
            i += 2
        else:
            processed_words.append(words[i])
            i += 1

    if 'toilet' in processed_words and 'seat' in processed_words:
        processed_words = [w for w in processed_words if w != 'seat']

    node_words = []
    for w in processed_words:
        if w in mscoco_objects_set:
            node_words.append(inverse_synonym_dict.get(w, w))

    return node_words


def evaluate_pope_stas(model, data, pca_basis, mean_directions, target_layers,
                         visual_router,
                         base_alpha=2.0, routing_weight=0.5,
                         mode='stas', max_new_tokens=10,
                         collect_routing_stats=False):
    if mode not in ('base', 'stas'):
        raise ValueError(f'Unknown mode: {mode}')
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    predictions = []
    labels = []
    all_routing_stats = {l: [] for l in target_layers if l in pca_basis}

    for item in data:
        image = item['image']
        question = item['question']
        ground_truth = item['ground_truth']
        labels.append(1 if ground_truth.lower() == 'yes' else 0)

        if mode == 'base':
            response = model.generate(image, question, max_new_tokens=max_new_tokens)
        elif mode == 'stas':
            response, routing_stats = _generate_stas(model,
                image, question, pca_basis, mean_directions, target_layers,
                visual_router,
                base_alpha=base_alpha, routing_weight=routing_weight,
                max_new_tokens=max_new_tokens,
                return_routing_stats=True
            )
            if collect_routing_stats:
                for l in routing_stats:
                    all_routing_stats[l].extend(routing_stats[l])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        pred = 1 if parse_yes_no(response) == 'yes' else 0
        predictions.append(pred)

    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    yes_count = sum(predictions)

    result = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'yes_count': yes_count,
        'total': len(labels),
    }

    if collect_routing_stats and mode == 'stas':
        result['routing_stats'] = all_routing_stats

    return result


def evaluate_mme_stas(model, data, pca_basis, mean_directions, target_layers,
                        visual_router,
                        base_alpha=2.0, routing_weight=0.5,
                        mode='stas', max_new_tokens=10,
                        collect_routing_stats=False):
    if mode not in ('base', 'stas'):
        raise ValueError(f'Unknown mode: {mode}')
    results = []

    for item in data:
        image = item['image']
        question = item['question']
        gt = item['ground_truth']
        category = item['category']
        question_id = item['question_id']

        if mode == 'base':
            response = model.generate(image, question, max_new_tokens=max_new_tokens)
        elif mode == 'stas':
            response, _ = _generate_stas(model,
                image, question, pca_basis, mean_directions, target_layers,
                visual_router,
                base_alpha=base_alpha, routing_weight=routing_weight,
                max_new_tokens=max_new_tokens,
                return_routing_stats=True
            )

        pred = parse_yes_no(response)
        results.append({
            'ground_truth': gt, 'prediction': pred,
            'category': category, 'question_id': question_id
        })

    by_category = {}
    categories = set(r['category'] for r in results)
    from sklearn.metrics import accuracy_score
    for cat in sorted(categories):
        cat_results = [r for r in results if r['category'] == cat]
        gt_bin = [1 if r['ground_truth'] == 'yes' else 0 for r in cat_results]
        pred_bin = [1 if r['prediction'] == 'yes' else 0 for r in cat_results]
        accuracy = accuracy_score(gt_bin, pred_bin)

        image_groups = defaultdict(list)
        for r in cat_results:
            image_groups[r['question_id']].append(r)
        both_correct = sum(
            1 for group in image_groups.values()
            if len(group) >= 2 and all(r['prediction'] == r['ground_truth'] for r in group[:2])
        )
        acc_plus = both_correct / len(image_groups) if image_groups else 0
        score = (accuracy + acc_plus) * 100

        by_category[cat] = {
            'accuracy': round(accuracy, 4),
            'acc_plus': round(acc_plus, 4),
            'score': round(score, 2),
            'total': len(cat_results)
        }

    perception_cats = [c for c in categories if c in MME_PERCEPTION_CATEGORIES]
    cognition_cats = [c for c in categories if c in MME_COGNITION_CATEGORIES]
    perception_score = sum(by_category[c]['score'] for c in perception_cats if c in by_category)
    cognition_score = sum(by_category[c]['score'] for c in cognition_cats if c in by_category)

    overall = {
        'perception_score': round(perception_score, 2),
        'cognition_score': round(cognition_score, 2),
        'mme_score': round(perception_score + cognition_score, 2),
    }
    return {'by_category': by_category, 'overall': overall}


def evaluate_amber_disc_stas(model, data, pca_basis, mean_directions, target_layers,
                                visual_router,
                                base_alpha=2.0, routing_weight=0.5,
                                mode='stas', max_new_tokens=10,
                                collect_routing_stats=False):
    if mode not in ('base', 'stas'):
        raise ValueError(f'Unknown mode: {mode}')
    results = []

    for item in data:
        image = item['image']
        question = item['question']
        truth = item['truth']
        annotation_type = item['annotation_type']

        if mode == 'base':
            response = model.generate(image, question, max_new_tokens=max_new_tokens)
        elif mode == 'stas':
            response, _ = _generate_stas(model,
                image, question, pca_basis, mean_directions, target_layers,
                visual_router,
                base_alpha=base_alpha, routing_weight=routing_weight,
                max_new_tokens=max_new_tokens,
                return_routing_stats=True
            )

        parsed = parse_yes_no(response)
        results.append({
            'truth': truth, 'parsed': parsed,
            'annotation_type': annotation_type
        })

    def _calc_metrics(res_list):
        if not res_list:
            return {}
        correct = ans_no_correct = 0
        truth_no_correct = 0
        ans_no_num = truth_no_num = 0
        for r in res_list:
            pred = r['parsed']
            tr = r['truth']
            if pred == tr:
                correct += 1
            if pred == 'no':
                ans_no_num += 1
                if tr == 'no':
                    ans_no_correct += 1
            if tr == 'no':
                truth_no_num += 1
                if pred == 'no':
                    truth_no_correct += 1
        total = len(res_list)
        acc = correct / total * 100 if total else 0
        prec = ans_no_correct / ans_no_num * 100 if ans_no_num else 0
        rec = truth_no_correct / truth_no_num * 100 if truth_no_num else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        return {'accuracy': round(acc, 2), 'precision': round(prec, 2),
                'recall': round(rec, 2), 'f1': round(f1, 2), 'total': total}

    total_metrics = _calc_metrics(results)
    by_subtype = {}
    types = set(r['annotation_type'] for r in results)
    for t in sorted(types):
        by_subtype[t] = _calc_metrics([r for r in results if r['annotation_type'] == t])

    return {'total': total_metrics, 'by_subtype': by_subtype}


def evaluate_amber_gen_stas(model, data, pca_basis, mean_directions, target_layers,
                               visual_router,
                               base_alpha=2.0, routing_weight=0.5,
                               mode='stas', max_new_tokens=512):
    if mode not in ('base', 'stas'):
        raise ValueError(f'Unknown mode: {mode}')
    from PIL import Image
    results = []

    for item in data:
        if 'image' in item:
            image = item['image']
        elif 'image_path' in item:
            image = Image.open(item['image_path']).convert('RGB')
        else:
            continue
        question = item['question']
        item_id = item['id']

        if mode == 'base':
            response = model.generate(image, question, max_new_tokens=max_new_tokens)
        elif mode == 'stas':
            response, _ = _generate_stas(model,
                image, question, pca_basis, mean_directions, target_layers,
                visual_router,
                base_alpha=base_alpha, routing_weight=routing_weight,
                max_new_tokens=max_new_tokens,
                return_routing_stats=True
            )

        results.append({'id': item_id, 'response': response})

    return results


def evaluate_chair_stas(model, data, gt_objects, chair_helpers,
                          pca_basis, mean_directions, target_layers,
                          visual_router,
                          base_alpha=2.0, routing_weight=0.5,
                          mode='stas', max_new_tokens=256):
    if mode not in ('base', 'stas'):
        raise ValueError(f'Unknown mode: {mode}')
    mscoco_objects_set = chair_helpers['mscoco_objects_set']
    inverse_synonym_dict = chair_helpers['inverse_synonym_dict']
    double_word_dict = chair_helpers['double_word_dict']

    total_coco_words = 0
    total_halluc_words = 0
    total_caps = 0
    halluc_caps = 0
    total_gt_objects = 0
    total_recall_objects = 0
    total_caption_length = 0

    from PIL import Image

    for item in data:
        if 'image' in item:
            image = item['image']
        elif 'image_path' in item:
            image = Image.open(item['image_path']).convert('RGB')
        else:
            continue
        question = item['question']
        img_id = item['image_id']

        if mode == 'base':
            caption = model.generate(image, question, max_new_tokens=max_new_tokens)
        elif mode == 'stas':
            caption, _ = _generate_stas(model,
                image, question, pca_basis, mean_directions, target_layers,
                visual_router,
                base_alpha=base_alpha, routing_weight=routing_weight,
                max_new_tokens=max_new_tokens,
                return_routing_stats=True
            )

        generated_objects = _caption_to_objects(caption, mscoco_objects_set, inverse_synonym_dict, double_word_dict)
        gt_objs = gt_objects.get(img_id, set())

        hallucinated_words = [obj for obj in generated_objects if obj not in gt_objs]

        recall_objects = set(obj for obj in generated_objects if obj in gt_objs)

        total_caps += 1
        total_coco_words += len(generated_objects)
        total_halluc_words += len(hallucinated_words)
        if hallucinated_words:
            halluc_caps += 1
        total_gt_objects += len(gt_objs)
        total_recall_objects += len(recall_objects)
        total_caption_length += len(caption.split())

    eps = 1e-6
    chair_s = halluc_caps / max(total_caps, eps) * 100
    chair_i = total_halluc_words / max(total_coco_words, eps) * 100
    recall = total_recall_objects / max(total_gt_objects, eps) * 100
    avg_len = 0.01 * total_caption_length / max(total_caps, eps) * 100

    return {
        'CHAIRs': round(chair_s, 2),
        'CHAIRi': round(chair_i, 2),
        'Recall': round(recall, 2),
        'Length': round(avg_len, 4),
        'total_captions': total_caps,
    }


def evaluate_dataset(dataset_name, model, data, pca_basis, mean_directions, target_layers,
                     visual_router, base_alpha=2.0, routing_weight=0.5, mode='stas',
                     amber_gen_data=None, amber_annotations=None, amber_relation_dict=None,
                     amber_safe_words=None, chair_gt_objects=None, chair_helpers=None):
    arguments = (model, data, pca_basis, mean_directions, target_layers, visual_router)
    options = {'base_alpha': base_alpha, 'routing_weight': routing_weight, 'mode': mode}
    if dataset_name == 'POPE':
        return evaluate_pope_stas(*arguments, **options)
    if dataset_name == 'MME':
        return evaluate_mme_stas(*arguments, **options)
    if dataset_name == 'AMBER':
        disc = evaluate_amber_disc_stas(*arguments, **options)
        responses = evaluate_amber_gen_stas(model, amber_gen_data, pca_basis, mean_directions,
                                            target_layers, visual_router, **options)
        gen = compute_amber_gen_metrics(responses, amber_annotations, amber_relation_dict, amber_safe_words)
        return {'discriminative': disc, 'generative': gen, 'responses': responses}
    if dataset_name == 'CHAIR':
        return evaluate_chair_stas(model, data, chair_gt_objects, chair_helpers,
                                   pca_basis, mean_directions, target_layers, visual_router, **options)
    raise ValueError(f'Unknown main-experiment dataset: {dataset_name}')


def main():
    parser = argparse.ArgumentParser(description='STAS main experiments: POPE, MME, AMBER and CHAIR')
    parser.add_argument('--model', default='qwen3_vl')
    parser.add_argument('--dataset', choices=['POPE', 'MME', 'AMBER', 'CHAIR'], default='POPE')
    parser.add_argument('--mode', choices=['base', 'stas', 'both'], default='both')
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--routing_weight', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=5.0)
    parser.add_argument('--tau', type=float)
    parser.add_argument('--num_components', type=int, default=10)
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model_path')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = get_model_config(args.model)
    model_name = config.name
    target_layers, paper_tau = paper_profile(model_name)
    tau = paper_tau if args.tau is None else args.tau
    if args.alpha < 0 or not 0 <= args.routing_weight <= 1 or args.beta <= 0 or tau < 0:
        raise ValueError('Invalid STAS hyperparameters')
    if args.num_samples is not None and args.num_samples < 1:
        raise ValueError('num_samples must be positive')
    paths = get_directories(model_name)
    pca_basis, mean_directions, router, metadata = {}, {}, None, None
    if args.mode != 'base':
        pca_dir = Path(paths['checkpoint_dir']) / 'pca_directions'
        metadata = json.loads((pca_dir / 'stas_metadata.json').read_text(encoding='utf-8'))
        if metadata.get('representation') != 'decoder_output_continuation_mean' or metadata.get('model') != model_name:
            raise ValueError('PCA metadata does not match the selected paper model')
        for filename, key in [('pca_basis.pt', 'basis_sha256'), ('mean_directions_raw.pt', 'mean_sha256')]:
            if hashlib.sha256((pca_dir / filename).read_bytes()).hexdigest() != metadata.get(key):
                raise ValueError(f'PCA artifact changed since extraction: {filename}')
        pca_basis = torch.load(pca_dir / 'pca_basis.pt', map_location='cpu')
        raw_directions = torch.load(pca_dir / 'mean_directions_raw.pt', map_location='cpu')
        if not set(target_layers).issubset(set(pca_basis) & set(raw_directions)):
            raise ValueError('PCA cache is missing paper intervention layers')
        if any(pca_basis[layer].shape[0] != args.num_components for layer in target_layers):
            raise ValueError('num_components does not match the PCA cache')
        mean_directions = {layer: torch.nn.functional.normalize(raw_directions[layer].float(), dim=-1)
                           for layer in target_layers}
        router = load_visual_router(model_name, device=args.device)
        if not set(target_layers).issubset(router.target_layers) or router.num_components != args.num_components:
            raise ValueError('Router does not match the paper layers or PCA dimension')
    extras = {}
    if args.dataset == 'POPE':
        data = load_pope_data(num_samples_per_cat=args.num_samples)
    elif args.dataset == 'MME':
        data = load_mme_data(num_samples_per_cat=args.num_samples)
    elif args.dataset == 'AMBER':
        data = load_amber_disc_data(num_samples=args.num_samples)
        extras['amber_gen_data'] = load_amber_gen_data(num_samples=args.num_samples)
        amber_dir = Path(DATA_ROOT) / 'AMBER' / 'data'
        extras['amber_annotations'] = json.loads((amber_dir / 'annotations.json').read_text(encoding='utf-8'))
        extras['amber_relation_dict'] = json.loads((amber_dir / 'relation.json').read_text(encoding='utf-8'))
        extras['amber_safe_words'] = (amber_dir / 'safe_words.txt').read_text(encoding='utf-8').splitlines()
        if not extras['amber_gen_data']:
            raise ValueError('No AMBER generative evaluation samples were loaded')
    else:
        data, extras['chair_gt_objects'], extras['chair_helpers'] = load_chair_data(num_samples=args.num_samples)
    if not data:
        raise ValueError(f'No {args.dataset} evaluation samples were loaded')
    output_dir = Path(paths['result_dir']) / 'stas_token_mlp' / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / 'config.json'
    result_path = output_dir / 'results.json'
    run_config = {key: value for key, value in vars(args).items() if key != 'resume'}
    run_config.update(model=model_name, target_layers=target_layers, tau=tau, pca_metadata=metadata)
    if args.resume and config_path.exists():
        if json.loads(config_path.read_text(encoding='utf-8')) != run_config:
            raise ValueError('Resume configuration differs from the saved run')
    results = json.loads(result_path.read_text(encoding='utf-8')) if args.resume and result_path.exists() else {}
    config_path.write_text(json.dumps(run_config, indent=2), encoding='utf-8')
    modes = ['base', 'stas'] if args.mode == 'both' else [args.mode]
    pending = [mode for mode in modes if mode not in results]
    if not pending:
        print(f'All requested results already exist: {result_path}')
        return
    model = load_model(model_name, model_path=args.model_path, device=args.device)
    model._stas_options = {'beta': args.beta, 'tau': tau}
    try:
        for mode in pending:
            result = evaluate_dataset(args.dataset, model, data, pca_basis, mean_directions, target_layers,
                                      router, args.alpha, args.routing_weight, mode, **extras)
            results[mode] = result
            result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
            metrics = {key: value for key, value in result.items() if key != 'responses'}
            print(f'{mode}: {json.dumps(metrics, ensure_ascii=False)}')
    finally:
        model.cleanup()
    print(f'Results saved to {result_path}')


if __name__ == '__main__':
    main()
