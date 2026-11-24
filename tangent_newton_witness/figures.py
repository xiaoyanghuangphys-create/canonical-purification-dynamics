"""Render standalone figures from published worker archives; no solver calls."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

LABELS=['A-old','A-new','B-old','B-new','Old-only','New-only']
COLORS=dict(zip(LABELS,['#a84d03','#037a72','#c37908','#2362a2','#925f35','#44417d']))
MARKERS=dict(zip(LABELS,['o','s','o','s','o','s']))


def cohort(root,name):
    path=root/name/'measurements.jsonl'
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def events(root,name,row,member):
    with ZipFile(root/name/(row['worker']+'.zip')) as z:
        return [json.loads(line) for line in z.read(member).splitlines()]


def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=165,bbox_inches='tight')
    plt.close(fig)


def matched(root,out):
    name='paired_confirm1'
    data=cohort(root,name)
    fig,axs=plt.subplots(2,2,figsize=(10.4,7.3),constrained_layout=True)
    colors=['#175a74','#64b493','#d6c498','#a3b9c7']
    csv_rows=[]
    for j,family in enumerate(['D','E']):
        a,b=axs[:,j]
        group=[r for r in data if r['family']==family]
        for i,label in enumerate(LABELS):
            rows=[r for r in group if r['label']==label]
            if len(rows)!=4 or any(r['status']!='completed' or r['global_cp_bound']>r['cert_tol'] for r in rows):
                continue
            x=[r['production_seconds'] for r in rows]
            y=[r['global_cp_error'] for r in rows]
            a.scatter(x,y,color=COLORS[label],s=18,alpha=.5,marker=MARKERS[label])
            a.plot(median(x),median(y),marker=MARKERS[label],color=COLORS[label],ms=8,label=label)
            parts=[]
            for r in rows:
                witness=r['witness_seconds']
                predictor=r['predictor_seconds']
                parts.append([predictor,witness,r['production_seconds']-predictor-witness,r['scoring_seconds']])
            left=0.
            for k,values in enumerate(np.asarray(parts).T):
                # Mean component times add to the mean measured total.
                value=np.mean(values)
                b.barh(i,value,left=left,color=colors[k],height=.62)
                left+=value
            cert=[r['global_cp_bound'] for r in rows]
            csv_rows.append(dict(family=family,label=label,repeats=len(rows),median_actual=median(y),
                 median_bound=median(cert),median_witness_s=median(r['witness_seconds'] for r in rows),
                 median_production_s=median(x),min_production_s=min(x),max_production_s=max(x),
                 median_total_s=median(r['total_seconds'] for r in rows),
                 peak_rss_mib=max(r['peak_rss_mib'] for r in rows)))
        a.axhline(1e-5,color='0.55',ls=':',lw=1)
        a.set(xscale='log',yscale='log',xlabel='Production time (s)',title=('Correlated D, T = 0.1' if j==0 else 'Parity E, T = 1'))
        if j==0:
            a.set_ylabel('Global root error')
        a.grid(alpha=.2,which='both')
        b.set(yticks=range(len(LABELS)),yticklabels=LABELS,xlabel='Mean time per trajectory (s)')
        b.invert_yaxis()
    axs[0,0].legend(ncol=2,fontsize=8,loc='best')
    axs[1,1].legend([plt.Rectangle((0,0),1,1,color=c) for c in colors],
                     ['Predictor','Witness','Other production','Scoring'],fontsize=8)
    fig.suptitle('Eight spins, one interval, four fresh runs per method')
    save(fig,out,'matched')
    with (out/'matched.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=csv_rows[0]);writer.writeheader();writer.writerows(csv_rows)


def tracking(root,out):
    name='paired_late_confirm'
    fig,axs=plt.subplots(1,3,figsize=(13.0,4.0),constrained_layout=True)
    for i,family in enumerate(('Dstress','E')):
        a=axs[i]
        for label in LABELS:
            row=next(r for r in cohort(root,name) if r['family']==family and r['label']==label)
            trail=events(root,name,row,'endpoints.jsonl')
            xx=[0]+[e['time'] for e in trail if e['accepted']]
            yy=[0]+[sum(v['endpoint_seconds'] for v in trail[:i+1]) for i,e in enumerate(trail) if e['accepted']]
            a.plot(xx,yy,color=COLORS[label],marker=MARKERS[label],ms=3,label=label)
            if not trail[-1]['accepted']:
                a.scatter(trail[-1]['time'],yy[-1]+trail[-1]['endpoint_seconds'],marker='x',s=48,color=COLORS[label])
        a.set(xlabel='Physical time',title=('Strong correlated D' if i==0 else 'Parity E'))
        if i==0:
            a.set_ylabel('Cumulative witness time (s)')
        a.grid(alpha=.22)
    a=axs[2]
    for i,family in enumerate(('Dstress','E')):
        for j,label in enumerate(LABELS):
            r=next(r for r in cohort(root,name) if r['family']==family and r['label']==label)
            current=events(root,name,r,'endpoints.jsonl')
            a.scatter([j-.12+i*.24],[sum(e['correction_updates'] for e in current if e['accepted'])],
                      color=COLORS[label],marker=['o','^'][i],s=43)
    a.set(xticks=range(6),xticklabels=LABELS,ylabel='Corrections at accepted endpoints',title='Physical updates')
    a.tick_params(axis='x',rotation=38)
    a.legend([Line2D([],[],color='0.3',marker=m,ls='') for m in ('o','^')],['Strong D','E'],fontsize=8)
    axs[0].legend(fontsize=7,ncol=2)
    fig.suptitle('Three spins, eight prescribed intervals to T = 4. Crosses mark rejected attempts.')
    save(fig,out,'tracking')


def frozen(root,out):
    group='paired_frozen_replay'
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.85),constrained_layout=True)
    for label in ('A-old','A-new','Old-only','New-only'):
        row=next(r for r in cohort(root,group) if r['label']==label)
        endpoint=events(root,group,row,'endpoints.jsonl')[-1]
        records=[e for e in endpoint['attempts'] if e['phase']=='correction_trial' and e['improved']]
        origin=min(endpoint['prior_root_bound'],endpoint['predictor_bound'] or float('inf'))
        y=[origin]+[e['bound'] for e in records]
        x=[0]+[e['elapsed_seconds'] for e in records]
        for a,axis in zip(axes,(range(len(y)),x)):
            a.plot(axis,y,label=label,color=COLORS[label],marker=MARKERS[label],ms=3)
    for ax,xlab in zip(axes,('Accepted physical correction updates','Elapsed endpoint time (s)')):
        ax.set(yscale='log',ylabel='Global certificate',xlabel=xlab)
        ax.axhline(1.e-7,color='0.55',ls=':')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Same saved witness at t = 3.5, same strong-D target at t = 4')
    save(fig,out,'frozen')


def scale(root,out):
    names=('paired_confirm1','paired_scale')
    data=[r for name in names for r in cohort(root,name) if r['cert_tol']==1.e-5]
    fig,axs=plt.subplots(2,3,figsize=(12,6.5),constrained_layout=True)
    for j,family in enumerate(('D','E')):
        a,b,c=axs[j]
        for label in ('A-old','A-new','Old-only','New-only'):
            groups=defaultdict(list)
            for r in data:
                if r['family']==family and r['label']==label:
                    groups[r['n']].append(r)
            x=[];y=[];rss=[];fb=[];rb=[]
            for n,rows in sorted(groups.items()):
                if any(r['status']!='completed' for r in rows):
                    a.plot(n,.96,marker='x',color=COLORS[label],transform=a.get_xaxis_transform())
                    continue
                x.append(n);y.append(median(r['witness_seconds'] for r in rows))
                rss.append(median(r['peak_rss_mib'] for r in rows))
                fb.append(rows[0]['factor_bond']);rb.append(rows[0]['root_bond'])
            a.plot(x,y,color=COLORS[label],marker=MARKERS[label],label=label)
            b.plot(x,rss,color=COLORS[label],marker=MARKERS[label])
            c.plot(x,fb,color=COLORS[label],marker=MARKERS[label])
            c.plot(x,rb,color=COLORS[label],marker=MARKERS[label],ls='--',alpha=.65)
        for ax in (a,b,c):
            ax.set(xscale='log',xticks=[8,16,32,64],xticklabels=['8','16','32','64'],xlabel='Spins')
            ax.minorticks_off();ax.grid(alpha=.2)
        a.set(yscale='log',ylabel=family+' witness time (s)')
        b.set(ylabel='Peak process RSS (MiB)')
        c.set(ylabel='Stored bond',title='Factor (solid), root (dashed)' if j==0 else '')
    axs[0,0].legend(fontsize=7)
    fig.suptitle('Completed one-interval runs at the common $10^{-5}$ certificate; crosses indicate a miss')
    save(fig,out,'scale')



def frontier(root,out):
    """Fastest repeated setting whose worst certificate is below the plotted target."""
    fig,ax=plt.subplots(figsize=(6.5,4.2),constrained_layout=True)
    defaults='paired_confirm1'
    groups=[(name,cohort(root,name)) for name in (defaults,'paired_tuned_confirm')]
    lines=[]
    thresholds=[1.e-5,3.e-6,1.e-6,3.e-7,1.e-7,5.e-8]
    for label in ('A-old','A-new'):
        points=[]
        for tol in thresholds:
            eligible=[]
            for name,rows in groups:
                runs=[r for r in rows if r['family']=='D' and r['n']==8 and r['label']==label]
                if len(runs)==4 and all(r['status']=='completed' and r['global_cp_bound']<=tol for r in runs):
                    eligible.append((median(r['production_seconds'] for r in runs),name,runs))
            if eligible:
                cost,name,runs=min(eligible,key=lambda v:v[0])
                points.append((tol,cost))
                lines.append(dict(label=label,threshold=tol,seconds=cost,cohort=name,
                                  max_bound=max(r['global_cp_bound'] for r in runs),
                                  max_actual=max(r['global_cp_error'] for r in runs)))
        xx,yy=zip(*points)
        ax.plot(xx,yy,marker=MARKERS[label],color=COLORS[label],label=label)
    ax.set(xscale='log',yscale='log',xlabel='Required global certificate',ylabel='Median production time (s)',
           title='Correlated D, 8 spins, T = 0.1')
    ax.invert_xaxis(); ax.grid(alpha=.2,which='both'); ax.legend()
    save(fig,out,'frontier')
    with (out/'frontier.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=lines[0]);writer.writeheader();writer.writerows(lines)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path(__file__).parent/'results')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    for figure in (matched,tracking,frozen,scale,frontier):
        figure(args.results,args.out)
