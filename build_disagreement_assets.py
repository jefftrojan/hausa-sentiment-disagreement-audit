"""Build pivot-paper tables and a vector figure from measured disagreement results."""
import hashlib
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).parent
source=ROOT/'results/disagreement/summary.json'
s=json.loads(source.read_text())
tables=ROOT/'paper/tables';tables.mkdir(exist_ok=True)
images=ROOT/'paper/images';images.mkdir(exist_ok=True)


def table(name,caption,label,columns,header,rows):
    (tables/name).write_text('\n'.join([
        '% Generated from results/disagreement/summary.json.',r'\begin{table}[t]',r'\centering',r'\footnotesize',
        r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+columns+'}',
        r'\toprule',header+r' \\',r'\midrule',*[r+r' \\' for r in rows],r'\bottomrule',r'\end{tabular}',r'\end{table}','']))


names={'afriberta':'AfriBERTa','char':'Character LR','mbert':'mBERT','word':'Word LR','xlmr':'XLM-R','majority':'Majority class'}
rows=[]
for name,display in names.items():
    all_=s['groups']['eligible_all']['models'][name]['mean']
    un=s['groups']['unanimous']['models'][name]['mean']
    split=s['groups']['two_to_one']['models'][name]['mean']
    rows.append(f"{display} & {all_['macro_f1']:.3f} & {all_['vote_agreement']:.3f} & {un['macro_f1']:.3f} & {split['macro_f1']:.3f}")
table('disagreement_models.tex',r'Model evaluation on the 4,562 eligible posts. $F$ is macro-F1 against released labels; $V$ is empirical vote agreement. $F_U$ and $F_S$ denote unanimous and two-to-one groups. Encoder values average three runs.','tab:disagreement','lrrrr',
      r'Model & $F$ & $V$ & $F_U$ & $F_S$',rows)
rows=[]
for group,display in [('unanimous','Unanimous'),('two_to_one','Two-to-one'),('three_way','Three-way')]:
    info=s['groups'][group];counts=info['released_classes']
    rows.append(' & '.join([display,f"{info['n']:,}",*[f"{counts.get(k,0):,}" for k in ['negative','neutral','positive']]]))
table('disagreement_groups.tex','Eligible agreement groups and released-label composition. Three-way results are descriptive only.','tab:groups','lrrrr',
      'Group & Posts & Neg. & Neu. & Pos.',rows)
rows=[]
for group,display in [('eligible_all','All eligible'),('unanimous','Unanimous'),('two_to_one','Two-to-one')]:
    d=s['contrasts'][group]['vote_agreement'];lo,hi=d['ci95']
    rows.append(f"{display} & {d['mean_delta']:+.4f} & [{lo:+.4f}, {hi:+.4f}]")
table('disagreement_contrasts.tex',r'Exploratory paired AfriBERTa-minus-character differences in vote agreement $V$. Intervals jointly resample posts and encoder seeds.','tab:votecontrasts','lrl',
      r'Group & $\Delta V$ & 95\% CI',rows)

groups=['eligible_all','unanimous','two_to_one']
with plt.rc_context({'font.size':8}):
    fig,(ax,contrast)=plt.subplots(2,1,figsize=(3.45,4.2),gridspec_kw={'height_ratios':[1.25,1]})
    x=np.arange(3);w=.34
    for offset,name,color,display in [(-w/2,'char','#4477AA','Character LR'),(w/2,'afriberta','#CC6677','AfriBERTa')]:
        means=np.array([s['groups'][g]['models'][name]['mean']['macro_f1'] for g in groups])
        ci=np.array([s['groups'][g]['models'][name]['ci95']['macro_f1'] for g in groups])
        ax.bar(x+offset,means,width=w,color=color,label=display,
               yerr=np.array([means-ci[:,0],ci[:,1]-means]),capsize=2)
    ax.set_xticks(x,['All eligible','Unanimous','Two-to-one'])
    ax.set_ylim(0,1);ax.set_ylabel('Released-label macro-F1');ax.legend(loc='upper left',fontsize=7,frameon=False)
    ax.set_title('(a) Released-label evaluation',loc='left',fontsize=9)
    means=np.array([s['contrasts'][g]['vote_agreement']['mean_delta'] for g in groups])
    ci=np.array([s['contrasts'][g]['vote_agreement']['ci95'] for g in groups])
    contrast.axvline(0,color='.5',ls='--',lw=.8)
    contrast.errorbar(means,x,xerr=np.array([means-ci[:,0],ci[:,1]-means]),fmt='o',color='#4477AA',capsize=3,markersize=4)
    contrast.set_yticks(x,['All eligible','Unanimous','Two-to-one']);contrast.set_ylim(2.5,-.5)
    contrast.set_xlabel('AfriBERTa − character: vote-agreement difference')
    contrast.set_title('(b) Paired vote-agreement contrasts',loc='left',fontsize=9)
    for a in [ax,contrast]:
        a.spines[['top','right']].set_visible(False);a.set_axisbelow(True)
    ax.yaxis.grid(True,alpha=.2);contrast.xaxis.grid(True,alpha=.2)
    fig.tight_layout(pad=.7,h_pad=1.6)
    dest=ROOT/'results/disagreement/agreement_evaluation.pdf'
    fig.savefig(dest);fig.savefig(dest.with_suffix('.png'),dpi=250)
shutil.copyfile(dest,images/dest.name)
manifest_path=ROOT/'paper/evidence_manifest.json'
manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
evidence = [source, dest]
sensitivity = ROOT/'results/disagreement/sensitivity.json'
if sensitivity.exists():
    evidence.append(sensitivity)
for p in evidence:manifest[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
print('Generated three disagreement tables, vector figure and evidence hashes.')
