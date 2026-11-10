"""Exploratory post-review sensitivity checks; no fitting or label replacement.

Run after analyze_disagreement.py. Subsets were chosen after primary results.
Outputs contain aggregate scores only. CIs are not computed for these checks.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).parent
LABELS = ['negative', 'neutral', 'positive']


def main():
    joined_path = ROOT / 'data/disagreement/eligible_join.jsonl'
    joined = [json.loads(s) for s in joined_path.read_text().splitlines()]
    by_id = {r['id']: r for r in joined}
    paths = [ROOT / 'results/cpu/test_predictions.csv']
    with paths[0].open(newline='') as f:
        cpu = list(csv.DictReader(f))
    ids = [r['id'] for r in cpu]
    gold = np.array([r['gold'] for r in cpu])
    pred = {m: np.array([[r[m] for r in cpu]]) for m in ['majority', 'word', 'char']}
    for model in ['afriberta', 'mbert', 'xlmr']:
        runs = []
        for seed in [42, 43, 44]:
            path = ROOT / f'results/kaggle_matrix/hausacs_results/{model}_seed{seed}/test_predictions.jsonl'
            paths.append(path)
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            assert [r['id'] for r in rows] == ids
            assert [r['gold'] for r in rows] == gold.tolist()
            runs.append([r['pred'] for r in rows])
        pred[model] = np.array(runs)
    eligible = np.array([sid in by_id for sid in ids])
    consistent = np.array([sid in by_id and by_id[sid]['majority_label'] == by_id[sid]['released_label'] for sid in ids])
    masks = {'full_test': np.ones(len(ids), dtype=bool), 'eligible': eligible,
             'excluded': ~eligible, 'released_equals_unique_majority': consistent}
    assert eligible.sum() == 4562 and consistent.sum() == 4492
    assert len(ids) == len(set(ids)) == 5303
    output = {'scope': 'Exploratory post-review descriptive checks; no CIs, refits or replacement labels.', 'groups': {},
              'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [joined_path, *paths]}}
    for group, mask in masks.items():
        scores = {}
        for model, runs in pred.items():
            scores[model] = {'macro_f1': float(np.mean([f1_score(gold[mask], r[mask], labels=LABELS, average='macro', zero_division=0) for r in runs]))}
            if group in ['eligible', 'released_equals_unique_majority']:
                selected = [by_id[sid] for sid, keep in zip(ids, mask) if keep]
                scores[model]['vote_agreement'] = float(np.mean([
                    np.mean([row['vote_counts'].get(label, 0) / 3 for row, label in zip(selected, r[mask])]) for r in runs]))
        output['groups'][group] = {'n': int(mask.sum()), 'models': scores,
                                  'ranking_macro_f1': sorted(scores, key=lambda m: scores[m]['macro_f1'], reverse=True)}
    primary = json.loads((ROOT / 'results/disagreement/summary.json').read_text())
    for model in pred:
        for metric in ['macro_f1', 'vote_agreement']:
            assert np.isclose(output['groups']['eligible']['models'][model][metric], primary['groups']['eligible_all']['models'][model]['mean'][metric])
    path = ROOT / 'results/disagreement/sensitivity.json'
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output['groups'], indent=2))


if __name__ == '__main__':
    main()
