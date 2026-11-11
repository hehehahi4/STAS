import os
import io
import base64
import pandas as pd
from PIL import Image

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


MME_PERCEPTION_CATEGORIES = [
    'existence', 'count', 'position', 'color',
    'posters', 'celebrity', 'scene', 'landmark', 'artwork', 'OCR'
]

MME_COGNITION_CATEGORIES = [
    'commonsense_reasoning', 'numerical_calculation',
    'text_translation', 'code_reasoning'
]

def load_pope_data(categories=None, num_samples_per_cat=None):
    if categories is None:
        categories = ['adversarial', 'popular', 'random']
    pope_dir = os.path.join(DATA_ROOT, 'POPE', 'Full')
    all_data = []
    for cat in categories:
        path = os.path.join(pope_dir, f'{cat}-00000-of-00001.parquet')
        if not os.path.exists(path):
            print(f"Warning: POPE file not found: {path}")
            continue
        df = pd.read_parquet(path)
        if num_samples_per_cat:
            df = df.head(num_samples_per_cat)
        for _, row in df.iterrows():
            img_bytes = row['image']['bytes']
            if isinstance(img_bytes, str):
                img_bytes = base64.b64decode(img_bytes)
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            question = row['question']
            gt = row.get('answer', row.get('label', '')).lower().strip()
            all_data.append({
                'image': image,
                'question': question,
                'ground_truth': gt,
                'category': cat
            })
    return all_data

def load_mme_data(num_samples_per_cat=None):
    mme_dir = os.path.join(DATA_ROOT, 'MME', 'data')
    files = sorted([f for f in os.listdir(mme_dir) if f.endswith('.parquet')])
    all_data = []
    cat_counts = {}
    for fname in files:
        df = pd.read_parquet(os.path.join(mme_dir, fname))
        for _, row in df.iterrows():
            category = row.get('category', 'unknown')
            if num_samples_per_cat:
                cat_counts[category] = cat_counts.get(category, 0) + 1
                if cat_counts[category] > num_samples_per_cat:
                    continue
            img_bytes = row['image']['bytes']
            if isinstance(img_bytes, str):
                img_bytes = base64.b64decode(img_bytes)
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            question = row.get('question', row.get('prompt', ''))
            gt = row.get('answer', row.get('label', '')).lower().strip()
            question_id = row.get('question_id', row.get('image_id', ''))
            all_data.append({
                'image': image,
                'question': question,
                'ground_truth': gt,
                'category': category,
                'question_id': question_id
            })
    return all_data

def load_amber_disc_data(num_samples=None):
    amber_data_dir = os.path.join(DATA_ROOT, 'AMBER')
    amber_image_dir = os.path.join(amber_data_dir, 'image')
    query_file = os.path.join(amber_data_dir, 'data', 'query', 'query_discriminative.json')
    annotation_file = os.path.join(amber_data_dir, 'data', 'annotations.json')

    if not os.path.exists(query_file):
        print(f"Warning: AMBER query file not found: {query_file}")
        return []

    import json as _json
    queries = _json.load(open(query_file))
    queries = [q for q in queries if q['id'] >= 1005]
    if num_samples:
        queries = queries[:num_samples]

    annotations_by_id = {}
    if os.path.exists(annotation_file):
        annotations_list = _json.load(open(annotation_file))
        for ann in annotations_list:
            annotations_by_id[ann['id']] = ann

    all_data = []
    for query in queries:
        image_file = query['image']
        question = query['query']
        image_path = os.path.join(amber_image_dir, image_file)

        if not os.path.exists(image_path):
            continue

        image = Image.open(image_path).convert('RGB')

        qid = query['id']
        truth = 'yes'
        annotation_type = 'unknown'
        if qid in annotations_by_id:
            ann = annotations_by_id[qid]
            truth = str(ann.get('truth', 'yes')).lower().strip()
            annotation_type = ann.get('type', 'unknown')

        all_data.append({
            'image': image,
            'question': question,
            'truth': truth,
            'annotation_type': annotation_type
        })

    return all_data

def strip_thinking(text):
    lines = text.strip().split('\n')
    answer_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() == 'thought':
            continue
        if stripped.lower().startswith('thought//thought'):
            after = stripped[len('thought//thought'):].strip()
            if after:
                answer_lines.append(after)
            continue
        if stripped.startswith('//'):
            content = stripped[2:].strip()
            content_lower = content.lower()
            if (content_lower.startswith('yes') or content_lower.startswith('no')):
                answer_lines.append(content)
            continue
        if stripped.lower().startswith('thought//'):
            after = stripped[len('thought//'):].strip()
            if after and after.lower() != 'thought':
                answer_lines.append(after)
            continue
        if stripped.lower().startswith('thought '):
            after = stripped[len('thought'):].strip()
            after_lower = after.lower()
            if after_lower.startswith('yes') or after_lower.startswith('no') or after_lower.startswith('answer'):
                answer_lines.append(after)
            continue
        answer_lines.append(stripped)
    if answer_lines:
        return '\n'.join(answer_lines)
    return text

def parse_yes_no(text):
    clean = strip_thinking(text).lower().strip()
    first_sentence = clean
    for sep in ['. ', '.\n', '\n']:
        if sep in first_sentence:
            first_sentence = first_sentence.split(sep)[0]
            break
    first_sentence = first_sentence.replace(',', '').replace('.', '').strip()
    words = first_sentence.split()
    if words:
        first_word = words[0].rstrip('.,!?;:')
        if first_word == 'yes': return 'yes'
        if first_word == 'no': return 'no'
    if 'no' in words or 'not ' in first_sentence or "n't " in first_sentence:
        return 'no'
    return 'yes'
