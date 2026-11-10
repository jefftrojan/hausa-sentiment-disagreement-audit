"""Kaggle GPU sentiment baselines; token LID experiments await verified labels."""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
import torch
from huggingface_hub import model_info
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_linear_schedule_with_warmup, set_seed

MODELS={'afriberta':'castorini/afriberta_large','xlmr':'FacebookAI/xlm-roberta-base','mbert':'google-bert/bert-base-multilingual-cased'}
LABELS=['negative','neutral','positive']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=MODELS,default='afriberta')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--revision',help='Pin a previously resolved model commit')
    p.add_argument('--smoke',action='store_true',help='32 train/dev examples, no test evaluation; not paper results')
    args=p.parse_args()
    assert torch.cuda.is_available(), 'Enable GPU in Kaggle settings.'
    set_seed(args.seed)
    revision=args.revision or model_info(MODELS[args.model]).sha
    config={'model':MODELS[args.model],'revision':revision,'seed':args.seed,'epochs':args.epochs,
            'lr':2e-5,'batch_size':16,'gradient_accumulation':2,'max_length':128,'smoke':args.smoke}
    rows={s:[json.loads(l) for l in (args.data/f'{s}.jsonl').read_text().splitlines() if l.strip()] for s in ['train','dev','test']}
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        assert not ({r['normalized_text_hash'] for r in rows[a]} & {r['normalized_text_hash'] for r in rows[b]}), 'Split leakage'
    config['input_sha256']={s:hashlib.sha256((args.data/f'{s}.jsonl').read_bytes()).hexdigest() for s in rows}
    if args.smoke:
        rows={s:v[:32] for s,v in rows.items()}
    out=args.output/f'{args.model}_seed{args.seed}{"_smoke" if args.smoke else ""}'
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite prior results: {out}')
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    # Keep AfriBERTa's original SentencePiece byte fallback behavior.
    tokenizer=AutoTokenizer.from_pretrained(MODELS[args.model],revision=revision,use_fast=args.model!='afriberta')
    model=AutoModelForSequenceClassification.from_pretrained(MODELS[args.model],revision=revision,num_labels=3,
            id2label=dict(enumerate(LABELS)),label2id={v:i for i,v in enumerate(LABELS)}).cuda()
    collator=DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8)
    loaders={}
    for split,data in rows.items():
        enc=tokenizer([r['text'] for r in data],truncation=True,max_length=128)
        items=[{**{k:v[i] for k,v in enc.items()},'labels':LABELS.index(r['sentiment'])} for i,r in enumerate(data)]
        loaders[split]=DataLoader(items,batch_size=16,shuffle=split=='train',collate_fn=collator)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    epochs=1 if args.smoke else args.epochs
    total_steps=epochs*math.ceil(len(loaders['train'])/2)
    scheduler=get_linear_schedule_with_warmup(optimizer,int(.1*total_steps),total_steps)
    scaler=torch.amp.GradScaler('cuda')
    def evaluate(split):
        model.eval(); pred=[]; gold=[]
        with torch.no_grad():
            for batch in loaders[split]:
                batch={k:v.cuda() for k,v in batch.items()}
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=model(**batch).logits
                pred.extend(logits.argmax(-1).cpu().tolist()); gold.extend(batch['labels'].cpu().tolist())
        return pred,gold,float(f1_score(gold,pred,labels=[0,1,2],average='macro',zero_division=0))
    best=-1; history=[]; started=time.perf_counter()
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        n=len(loaders['train'])
        for step,batch in enumerate(loaders['train']):
            batch={k:v.cuda() for k,v in batch.items()}
            # Correct scaling for an incomplete accumulation group at epoch end.
            group_size=min(2,n-(step//2)*2)
            with torch.autocast('cuda',dtype=torch.float16):
                loss=model(**batch).loss/group_size
            scaler.scale(loss).backward()
            if (step+1)%2==0 or step+1==n:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer); scaler.update(); scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (step+1)%100==0:
                print(f'epoch={epoch+1} step={step+1}/{n}',flush=True)
        _,_,score=evaluate('dev')
        history.append({'epoch':epoch+1,'dev_macro_f1':score})
        print(history[-1],flush=True)
        if score>best:
            best=score; model.save_pretrained(out/'best'); tokenizer.save_pretrained(out/'best')
    if args.smoke:
        (out/'smoke.json').write_text(json.dumps({'history':history,'status':'smoke only; no test results'}))
        return
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    model=AutoModelForSequenceClassification.from_pretrained(out/'best').cuda()
    pred,gold,score=evaluate('test')
    with (out/'test_predictions.jsonl').open('w') as f:
        for row,g,v in zip(rows['test'],gold,pred):
            f.write(json.dumps({'id':row['id'],'gold':LABELS[g],'pred':LABELS[v]})+'\n')
    import transformers
    summary={'config':config,'history':history,'test_macro_f1':score,
             'report':classification_report(gold,pred,labels=[0,1,2],target_names=LABELS,output_dict=True,zero_division=0),
             'elapsed_seconds':time.perf_counter()-started,'torch':torch.__version__,
             'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0),
             'scope':'AfriSenti Hausa sentiment only; no verified code-switched or pragmatics evaluation'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('Completed',out,'test_macro_f1',score,flush=True)


if __name__=='__main__':
    main()
