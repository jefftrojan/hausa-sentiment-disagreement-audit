"""Audit original votes, join fixed test predictions, and evaluate disagreement.

Requires only numpy. No raw text or annotator identities in published outputs.
"""
import csv
import hashlib
import json
import unicodedata
from collections import Counter,defaultdict
from decimal import Decimal,InvalidOperation
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parent
LABELS=['negative','neutral','positive']
GROUPS=['unanimous','two_to_one','three_way']
SEEDS=[42,43,44]


def norm(text):return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())


def annotator_id(value):
    value=value.strip()
    if not value:return ''
    try:return str(Decimal(value).normalize())
    except InvalidOperation:return value


def f1(gold,pred):
    cm=np.bincount(gold*3+pred,minlength=9).reshape(3,3)
    den=cm.sum(0)+cm.sum(1)
    return float(np.divide(2*np.diag(cm),den,out=np.zeros(3),where=den!=0).mean())


def evaluate(y,q,pred,indices):
    yy=y[indices]; pp=pred[indices]; qq=q[indices]
    vote=qq[np.arange(len(indices)),pp]
    return {'macro_f1':f1(yy,pp),'released_accuracy':float(np.mean(yy==pp)),
            'vote_agreement':float(vote.mean()),
            'above_uniform':float(vote.mean()-1/3),
            'ceiling_shortfall':float(qq.max(1).mean()-vote.mean())}


def main():
    test_path=ROOT/'data/clean/test.jsonl'
    source_path=ROOT/'data/raw/data_with_annotators_labels/hausa_individual_labels.csv'
    test=[json.loads(l) for l in test_path.read_text().splitlines()]
    with source_path.open(encoding='utf-8-sig',newline='') as f:source=list(csv.DictReader(f))
    source_index=defaultdict(list)
    for i,r in enumerate(source):source_index[norm(r['text'])].append((i,r))
    audit=Counter(); labels=Counter(); excluded=[]; joined=[]
    for row in test:
        candidates=source_index[norm(row['text'])]
        if len(candidates)!=1:
            reason='unmatched' if not candidates else 'nonunique_source_match'
            audit[reason]+=1;excluded.append({'id':row['id'],'reasons':[reason]});continue
        audit['unique_matched']+=1
        source_i,r=candidates[0]
        votes=[r[f'label_{i}'].strip().lower() for i in (1,2,3)]
        people=[annotator_id(r[f'annotator_{i}']) for i in (1,2,3)]
        labels.update(votes)
        reasons=[]
        if not all(votes):reasons.append('missing_vote')
        if any(v and v not in LABELS for v in votes):reasons.append('non_threeclass_vote')
        if not all(people):reasons.append('missing_annotator_id')
        elif len(set(people))!=3:reasons.append('repeated_annotator_id')
        if reasons:
            audit['matched_but_ineligible']+=1
            audit.update(reasons);excluded.append({'id':row['id'],'reasons':reasons});continue
        counts=Counter(votes)
        pattern={1:'unanimous',2:'two_to_one',3:'three_way'}[len(counts)]
        majority=counts.most_common(1)[0][0] if max(counts.values())>=2 else None
        joined.append({'id':row['id'],'released_label':row['sentiment'],'vote_counts':dict(counts),
                       'pattern':pattern,'majority_label':majority,'source_row':source_i,
                       'raw_exact_text_match':row['text']==r['text']})
    assert len(joined)==4562, 'Source/eligibility changed; inspect rather than silently changing protocol'
    ids=[r['id'] for r in joined]
    gold=np.array([LABELS.index(r['released_label']) for r in joined])
    q=np.array([[r['vote_counts'].get(c,0)/3 for c in LABELS] for r in joined])
    pattern=np.array([r['pattern'] for r in joined])
    assert np.allclose(q.sum(1),1)
    test_ids=[r['id'] for r in test]
    test_labels=[r['sentiment'] for r in test]
    positions={r['id']:i for i,r in enumerate(test)}
    take=np.array([positions[sid] for sid in ids])
    with (ROOT/'results/cpu/test_predictions.csv').open(newline='') as f:cpu=list(csv.DictReader(f))
    assert [r['id'] for r in cpu]==test_ids and [r['gold'] for r in cpu]==test_labels
    predictions={name:np.array([[LABELS.index(r[name]) for r in cpu]])[:,take] for name in ['majority','word','char']}
    prediction_paths=[ROOT/'results/cpu/test_predictions.csv']
    for name in ['afriberta','mbert','xlmr']:
        runs=[]
        for seed in SEEDS:
            path=ROOT/f'results/kaggle_matrix/hausacs_results/{name}_seed{seed}/test_predictions.jsonl'
            rows=[json.loads(l) for l in path.read_text().splitlines()]
            assert [r['id'] for r in rows]==test_ids and [r['gold'] for r in rows]==test_labels
            runs.append([LABELS.index(r['pred']) for r in rows]);prediction_paths.append(path)
        predictions[name]=np.array(runs)[:,take]
    selections={'eligible_all':np.arange(len(joined)),**{g:np.flatnonzero(pattern==g) for g in GROUPS}}
    result={'scope':'Exploratory reuse of original votes; no new participants or annotations',
            'audit':dict(audit),'matched_vote_labels':dict(labels),'eligible':len(joined),
            'pattern_counts':dict(Counter(pattern)),
            'raw_exact_matches_among_eligible':sum(r['raw_exact_text_match'] for r in joined),
            'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
            'test_sha256':hashlib.sha256(test_path.read_bytes()).hexdigest(),
            'prediction_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in prediction_paths},
            'released_vs_majority':dict(Counter('tie' if r['majority_label'] is None else 'same' if r['majority_label']==r['released_label'] else 'different' for r in joined)),
            'groups':{},'contrasts':{},
            'method':'2000 paired post resamples stratified by (released class, observed agreement pattern), jointly resampled encoder seeds, RNG 42. Votes fixed within post; CPU fits fixed; unadjusted exploratory intervals. Three-way n=6 is descriptive.'}
    values={}
    for group,ix in selections.items():
        group_result={'n':len(ix),'released_classes':dict(Counter(LABELS[v] for v in gold[ix])),
                      'vote_ceiling':float(q[ix].max(1).mean()),'models':{}}
        for name,runs in predictions.items():
            scores=[evaluate(gold,q,p,ix) for p in runs]
            group_result['models'][name]={'mean':{k:float(np.mean([v[k] for v in scores])) for k in scores[0]},'per_run':scores}
            values[group,name]={k:[] for k in scores[0] if k in ['macro_f1','vote_agreement']}
        result['groups'][group]=group_result
    strata=[np.flatnonzero((pattern==g)&(gold==c)) for g in GROUPS for c in range(3)]
    strata=[s for s in strata if len(s)]
    rng=np.random.default_rng(42)
    for _ in range(2000):
        draws=[rng.choice(s,len(s),replace=True) for s in strata]
        ix_all=np.concatenate(draws)
        seed_draw=rng.integers(0,3,size=3)
        for group in selections:
            ix=ix_all if group=='eligible_all' else ix_all[pattern[ix_all]==group]
            for name,runs in predictions.items():
                chosen=runs[seed_draw] if len(runs)==3 else runs
                scores=[evaluate(gold,q,p,ix) for p in chosen]
                for metric in values[group,name]:
                    values[group,name][metric].append(float(np.mean([s[metric] for s in scores])))
    for group,info in result['groups'].items():
        for name in predictions:
            info['models'][name]['ci95']={k:np.quantile(v,[.025,.975]).tolist() for k,v in values[group,name].items()} if group!='three_way' else None
        result['contrasts'][group]={}
        for metric in ['macro_f1','vote_agreement']:
            delta=np.array(values[group,'afriberta'][metric])-values[group,'char'][metric]
            result['contrasts'][group][metric]={'mean_delta':info['models']['afriberta']['mean'][metric]-info['models']['char']['mean'][metric],
                     'ci95':np.quantile(delta,[.025,.975]).tolist() if group!='three_way' else None}
        info['ranking']={metric:sorted(predictions,key=lambda n:info['models'][n]['mean'][metric],reverse=True) for metric in ['macro_f1','vote_agreement']}
    out=ROOT/'results/disagreement';out.mkdir(parents=True,exist_ok=True)
    private=ROOT/'data/disagreement';private.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (private/'eligible_join.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in joined))
    (private/'exclusions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in excluded))
    with (out/'item_scores.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','pattern','released_label','vote_ceiling',*[n+'_vote_agreement' for n in predictions]])
        for i,r in enumerate(joined):w.writerow([r['id'],r['pattern'],r['released_label'],q[i].max(),*[float(q[i,p[:,i]].mean()) for p in predictions.values()]])
    print(json.dumps({'eligible':len(joined),'audit':result['audit'],'patterns':result['pattern_counts'],'released_vs_majority':result['released_vs_majority'],
          'groups':{g:{'n':v['n'],'ceiling':v['vote_ceiling'],'models':{n:m['mean'] for n,m in v['models'].items()},'rankings':v['ranking']} for g,v in result['groups'].items()},
          'contrasts':result['contrasts']},indent=2))


if __name__=='__main__':main()
