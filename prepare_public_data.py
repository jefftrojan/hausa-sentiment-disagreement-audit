"""Import a checked-out official AfriSenti release; audit without training.

Usage: python3 prepare_public_data.py --source /path/to/afrisent-semeval-2023
Outputs are generated artifacts, not a newly collected HausaCS corpus.
"""
import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import subprocess
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalized(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    text = re.sub(r'https?://\S+', '<url>', text)
    return ' '.join(text.split())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'data')
    args = parser.parse_args()
    root = args.source.resolve()
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    out = args.output.resolve()
    raw = out / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {'repository': 'https://github.com/afrisenti-semeval/afrisent-semeval-2023',
                'revision': revision, 'files': [],
                'license_note': 'Official GitHub README states CC BY 4.0; HF NaijaSenti card states CC BY-NC-SA 4.0. Preserve release-specific terms; resolve before redistribution.',
                'annotation_status': 'Sentiment supplied by source. Token LID and pragmatics not annotated here.'}
    paths = [f'data/hau/{split}.tsv' for split in ('train', 'dev', 'test')]
    paths += ['data_with_annotators_labels/hausa_individual_labels.csv',
              'README.md', 'data/README.txt', 'data_with_annotators_labels/README.txt']
    for rel in paths:
        source = root / rel
        dest = raw / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        manifest['files'].append({'path': rel, 'sha256': digest(source.read_bytes()),
                                  'url': f"{manifest['repository']}/blob/{revision}/{rel}"})
    records = {}
    groups = defaultdict(list)
    summary = {'splits': {}, 'cross_split_normalized_groups': []}
    for split in ('train', 'dev', 'test'):
        with (raw / f'data/hau/{split}.tsv').open(newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
        counts = Counter()
        exact = Counter()
        built = []
        for i, row in enumerate(rows, 1):
            text, label = row['tweet'], row['label'].strip().lower()
            if not text.strip() or label not in {'negative', 'neutral', 'positive'}:
                raise ValueError(f'Invalid row {split}:{i}')
            key = digest(normalized(text).encode())
            rec = {'id': f'afrisenti-hau-{split}-{i:06d}', 'text': text,
                   'sentiment': label, 'source_split': split, 'normalized_text_hash': key}
            groups[key].append({'id': rec['id'], 'split': split, 'label': label})
            counts[label] += 1
            exact[text] += 1
            built.append(rec)
        records[split] = built
        summary['splits'][split] = {'rows': len(built), 'labels': dict(counts),
                                    'exact_duplicate_excess': sum(n-1 for n in exact.values())}
        with (out / f'{split}.jsonl').open('w', encoding='utf-8') as f:
            for rec in built:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    for key, members in groups.items():
        if len({r['split'] for r in members}) > 1:
            summary['cross_split_normalized_groups'].append({'hash': key, 'members': members})
    summary['normalized_conflicting_label_groups'] = sum(len({r['label'] for r in members}) > 1 for members in groups.values())
    with (raw / 'data_with_annotators_labels/hausa_individual_labels.csv').open(newline='', encoding='utf-8-sig') as f:
        annotations = list(csv.DictReader(f))
    summary['source_annotation_rows'] = len(annotations)
    # Balanced training-only screening pilot: no evaluation texts, no gold LID
    # claims. Duplicates and any normalized train/dev/test overlap are excluded.
    rng = random.Random(42)
    candidates = defaultdict(list)
    seen = set()
    for rec in records['train']:
        key = rec['normalized_text_hash']
        if key in seen or len({r['split'] for r in groups[key]}) > 1:
            continue
        seen.add(key)
        candidates[rec['sentiment']].append(rec)
    pilot = []
    for label in sorted(candidates):
        rng.shuffle(candidates[label])
        pilot.extend(candidates[label][:50])
    rng.shuffle(pilot)
    fields = ['id', 'text', 'annotator_id', 'sentence_language', 'token_annotations_json',
              'sentiment_review', 'pragmatic_tags', 'context_sufficient', 'notes']
    with (out / 'annotation_pilot.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in pilot:
            writer.writerow({'id': rec['id'], 'text': rec['text']})
    summary['pilot_rows'] = len(pilot)
    (out / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({**summary, 'cross_split_normalized_groups': len(summary['cross_split_normalized_groups'])}, indent=2))


if __name__ == '__main__':
    main()
