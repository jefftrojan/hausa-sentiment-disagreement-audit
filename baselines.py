"""Audit-clean official splits; select CPU models on dev; evaluate test once.

Run using an environment with numpy, scipy, scikit-learn, threadpoolctl.
"""
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).parent
LABELS = ['negative', 'neutral', 'positive']


def clean_splits():
    groups = defaultdict(list)
    for split in ('train', 'dev', 'test'):
        for line in (ROOT / 'data' / f'{split}.jsonl').read_text().splitlines():
            row = json.loads(line)
            groups[row['normalized_text_hash']].append(row)
    priority = {'test': 0, 'dev': 1, 'train': 2}
    clean = {s: [] for s in priority}
    exclusions = []
    for members in groups.values():
        if len({r['sentiment'] for r in members}) > 1:
            exclusions.extend({'id': r['id'], 'reason': 'normalized_label_conflict'} for r in members)
            continue
        ordered = sorted(members, key=lambda r: (priority[r['source_split']], r['id']))
        kept = ordered[0]
        clean[kept['source_split']].append(kept)
        exclusions.extend({'id': r['id'], 'reason': 'normalized_duplicate', 'retained_id': kept['id']} for r in ordered[1:])
    out = ROOT / 'data' / 'clean'
    out.mkdir(exist_ok=True)
    for split, rows in clean.items():
        rows.sort(key=lambda r: r['id'])
        (out / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    report = {'policy': 'Drop all normalized label-conflict groups. Keep one row per normalized text; priority test > dev > train; original row order breaks ties.',
              'counts': {s: {'rows': len(rows), 'labels': dict(Counter(r['sentiment'] for r in rows))} for s, rows in clean.items()},
              'excluded_rows': len(exclusions), 'exclusions': exclusions,
              'limitations': 'No reliable author/conversation IDs; normalized disjointness does not guarantee author, conversation or paraphrase disjointness.'}
    (out / 'split_policy.json').write_text(json.dumps(report, indent=2)+'\n')
    hashes = [{r['normalized_text_hash'] for r in clean[s]} for s in ('train','dev','test')]
    assert not (hashes[0]&hashes[1] or hashes[0]&hashes[2] or hashes[1]&hashes[2])
    return clean, report


def scores(y, pred):
    return {'macro_f1': float(f1_score(y, pred, labels=LABELS, average='macro', zero_division=0)),
            'report': classification_report(y, pred, labels=LABELS, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y, pred, labels=LABELS).tolist()}


def bootstrap(y, preds, repeats=2000):
    # Use identical resamples for each model and paired differences. Stratify
    # by true class; this conditions on the observed test class proportions.
    rng = np.random.default_rng(42)
    y = np.array([LABELS.index(v) for v in y])
    encoded = {name: np.array([LABELS.index(v) for v in p]) for name,p in preds.items()}
    values = {name: [] for name in preds}
    strata = [np.flatnonzero(y==i) for i in range(3)]
    for _ in range(repeats):
        ix = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        for name, pred in encoded.items():
            cm = np.bincount(y[ix]*3+pred[ix], minlength=9).reshape(3,3)
            den = cm.sum(axis=0)+cm.sum(axis=1)
            f = np.divide(2*np.diag(cm), den, out=np.zeros(3), where=den!=0).mean()
            values[name].append(float(f))
    result = {'method': '2000 paired, true-class-stratified test-row bootstrap resamples; seed 42. Conditional on fitted models, not training-seed or population uncertainty.',
              'macro_f1_ci95': {name: np.quantile(v, [.025,.975]).tolist() for name,v in values.items()}}
    result['char_minus_word_ci95'] = np.quantile(np.array(values['char'])-values['word'], [.025,.975]).tolist()
    return result


def main():
    clean, policy = clean_splits()
    x = {s: [r['text'] for r in rows] for s,rows in clean.items()}
    y = {s: [r['sentiment'] for r in rows] for s,rows in clean.items()}
    results, predictions = {}, {}
    dummy = DummyClassifier(strategy='most_frequent').fit(x['train'], y['train'])
    predictions['majority'] = dummy.predict(x['test'])
    results['majority'] = {'test': scores(y['test'], predictions['majority'])}
    for kind in ('word', 'char'):
        candidates, best = [], None
        for c in (.1, 1., 10.):
            vectorizer = TfidfVectorizer(analyzer='word' if kind=='word' else 'char',
                                        ngram_range=(1,2) if kind=='word' else (3,5),
                                        min_df=2, max_features=60000, sublinear_tf=True)
            model = Pipeline([('tfidf',vectorizer), ('lr',LogisticRegression(C=c, max_iter=500, random_state=42))])
            start = time.perf_counter()
            model.fit(x['train'],y['train'])
            dev = scores(y['dev'],model.predict(x['dev']))['macro_f1']
            candidate = {'C':c, 'dev_macro_f1':dev, 'fit_seconds':time.perf_counter()-start}
            candidates.append(candidate)
            print(kind, candidate, flush=True)
            if best is None or dev>best[0]:
                best=(dev, model, c)
        predictions[kind] = best[1].predict(x['test'])
        results[kind] = {'selected_C':best[2], 'candidates':candidates, 'test':scores(y['test'],predictions[kind])}
    out = ROOT / 'results' / 'cpu'
    out.mkdir(parents=True,exist_ok=True)
    with (out/'test_predictions.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['id','gold']+list(predictions))
        for i,row in enumerate(clean['test']):
            writer.writerow([row['id'],row['sentiment']]+[p[i] for p in predictions.values()])
    import sklearn
    payload={'scope':'AfriSenti Hausa subset sentiment, not a verified code-switched subset',
             'sklearn_version':sklearn.__version__, 'labels':LABELS, 'selection':'Dev macro-F1; fixed grid C=.1,1,10; train-only vocabulary; no refit on dev',
             'split_counts':policy['counts'], 'excluded_rows':policy['excluded_rows'],
             'input_sha256':{s:hashlib.sha256((ROOT/'data'/'clean'/f'{s}.jsonl').read_bytes()).hexdigest() for s in clean},
             'models':results,'uncertainty':bootstrap(y['test'],predictions)}
    (out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({k:v['test']['macro_f1'] for k,v in results.items()},indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
