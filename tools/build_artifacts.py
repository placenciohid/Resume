"""Regenerate portable results and article figures from the checked-in examples.

Run from the repository root after installing requirements-analysis.txt.
The four analyze.py commands regenerate real-data analysis-results.json separately.
"""
from pathlib import Path
import importlib.util
import json
import random
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'#f5f7fa',
                     'axes.titleweight':'bold','axes.labelcolor':'#26344a'})
BLUE,TEAL,ORANGE='#3261a8','#16877b','#cf742c'

def module(slug):
    spec=importlib.util.spec_from_file_location(slug,ROOT/'projects'/slug/'demo.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def save(slug,name,figure):
    folder=ROOT/'projects'/slug/'figures';folder.mkdir(exist_ok=True)
    figure.savefig(folder/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(figure)

def main():
    results={}
    for slug in ['forecasting','knowledge-search','commercial-reporting','event-prediction',
                 'optimization','ml-from-scratch','clustering-noncompliance','dynamic-zoom']:
        path=ROOT/'projects'/slug
        result=json.loads(subprocess.check_output([sys.executable,str(path/'demo.py')],text=True))
        (path/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        results[slug]=result

    records=results['forecasting']['evaluation']['predictions'][-28:]
    fig,ax=plt.subplots(figsize=(10,3.6))
    for key,label,color in [('actual','Observed', '#26344a'),('seasonal_naive','Previous-week baseline',ORANGE),('trend_plus_weekday','Trend + weekday',TEAL)]:
        ax.plot(range(28),[r[key] for r in records],label=label,color=color,lw=1.8)
    ax.set(xlabel='Last 28 held-out days',ylabel='Synthetic production units',title='Forecasts issued before each seven-day window')
    ax.legend(ncol=3,fontsize=8);fig.tight_layout();save('forecasting','backtest',fig)

    opt=module('optimization');score,assignment,loads=opt.solve()
    fig,axes=plt.subplots(1,2,figsize=(10,3.4),gridspec_kw={'width_ratios':[1,2]})
    axes[0].bar(['First fit','Exact search'],[opt.first_fit(),score],color=[ORANGE,TEAL]);axes[0].set(ylabel='Synthetic priority score',title='Same capacity, different selection',ylim=(0,32))
    colors=plt.get_cmap('tab10');used=[0,0]
    for i,(label,duration,priority) in enumerate(opt.PROCEDURES):
        room=assignment[i]
        if room:
            axes[1].barh(room,duration,left=used[room-1],color=colors(i),height=.5)
            axes[1].text(used[room-1]+duration/2,room,label,ha='center',va='center',color='white')
            used[room-1]+=duration
    axes[1].set(yticks=[1,2],yticklabels=['Room 1','Room 2'],xlabel='Allocated hours',xlim=(0,4),title='Optimal capacity assignment; no start-time model')
    fig.tight_layout();save('optimization','allocation',fig)

    ml=module('ml-from-scratch');rng=random.Random(42)
    x=[[rng.uniform(-2,2),rng.uniform(-2,2)] for _ in range(240)]
    y=[int(r[0]+.4*r[1]>0) for r in x];xx,yy=np.meshgrid(np.linspace(-2,2,65),np.linspace(-2,2,65));grid=np.c_[xx.ravel(),yy.ravel()].tolist()
    preds=[ml.knn_predict(x[:180],y[:180],grid),ml.logistic_predict(ml.fit_logistic(x[:180],y[:180]),grid),ml.stump_predict(ml.fit_stump(x[:180],y[:180]),grid)]
    fig,axes=plt.subplots(1,3,figsize=(10,3.4))
    for ax,name,pred in zip(axes,['3-nearest neighbors','Logistic regression','Decision stump'],preds):
        ax.contourf(xx,yy,np.array(pred).reshape(xx.shape),levels=[-.5,.5,1.5],colors=['#e4b98f','#9ad6d0'],alpha=.55)
        test=np.array(x[180:]);ax.scatter(test[:,0],test[:,1],c=y[180:],cmap='coolwarm',s=18,edgecolor='#283849',linewidth=.3)
        ax.set(title=name,xlabel='Feature 1',ylabel='Feature 2')
    fig.suptitle('Synthetic decision boundaries; dots are the 60 held-out observations',fontsize=11)
    fig.tight_layout();save('ml-from-scratch','boundaries',fig)

    causal=results['clustering-noncompliance'];fig,ax=plt.subplots(figsize=(9,2.7))
    for i,(key,label,color) in enumerate([('independent_rows','Treat 200,000 rows as independent',ORANGE),('store_level','Use 200 independent stores',TEAL)]):
        r=causal[key];lo,hi=r['normal_95pct_interval'];ax.errorbar(r['estimate'],i,xerr=[[r['estimate']-lo],[hi-r['estimate']]],fmt='o',capsize=6,color=color,label=label)
    ax.axvline(0,color='#6b7380',ls='--');ax.set(yticks=[],xlabel='Estimated sales effect and approximate 95% interval',title='Same estimated effect, very different uncertainty')
    ax.legend(loc='upper left',fontsize=8);fig.tight_layout();save('clustering-noncompliance','uncertainty',fig)

    video=results['dynamic-zoom'];fig,ax=plt.subplots(figsize=(9,3.3))
    for key,color in [('fifo',ORANGE),('latest',TEAL)]:
        frames=video[key]['frames'];ax.plot([f['frame_id'] for f in frames],[f['latency_ms'] for f in frames],label=key.upper(),color=color)
    ax.set(xlabel='Source frame ID',ylabel='Capture-to-display latency (ms)',title='Simulated overload: 30 fps capture, 60 ms processing per frame')
    ax.legend();fig.tight_layout();save('dynamic-zoom','latency',fig)

    clustered=json.loads((ROOT/'projects/wine-clustering/analysis-results.json').read_text())
    fig,ax=plt.subplots(figsize=(8,3.4));ax.plot([r['k'] for r in clustered['candidates']],[r['silhouette'] for r in clustered['candidates']],marker='o',color=TEAL)
    ax.set(xlabel='Number of clusters',ylabel='Silhouette (higher is better)',title='UCI red wine: chemistry-only cluster comparison');ax.set_xticks(range(2,11))
    fig.tight_layout();save('wine-clustering','selection',fig)

    for slug,title in [('wine-neural-networks','Wine quality: grouped holdout'),('spaceship-titanic','Passenger prediction: unseen travel groups')]:
        r=json.loads((ROOT/'projects'/slug/'analysis-results.json').read_text());fig,ax=plt.subplots(figsize=(8,3.4))
        keys=list(r['holdout']);locations=np.arange(len(keys))
        for offset,metric,color in [(-.18,'balanced_accuracy',TEAL),(.18,'roc_auc',BLUE)]:
            ax.bar(locations+offset,[r['holdout'][k][metric] for k in keys],width=.32,label=metric.replace('_',' ').title(),color=color)
        ax.set(xticks=locations,xticklabels=[k.replace('_',' ').title() for k in keys],ylim=(0,1),title=title,ylabel='Score on fixed holdout')
        ax.legend(loc='lower right',fontsize=8);fig.tight_layout();save(slug,'evaluation',fig)
    print('Regenerated eight runnable snapshots and eight evidence figures.')

if __name__=='__main__':main()
