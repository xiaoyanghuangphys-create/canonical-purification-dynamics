"""Summarize the checked frozen-channel cohorts; read loose or archived worker files."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import median
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def data(path):
    return json.loads(path.read_text())


def worker_data(folder, relative):
    if (folder/relative).exists():
        return data(folder/relative)
    with ZipFile(folder/'workers.zip') as z:
        return json.loads(z.read(relative))


def table(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen',required=True,type=Path)
    p.add_argument('--confirm',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    rows=data(args.screen/'measurements.json')
    counts=Counter((r['family'],r.get('variant','all before witness'),r['status']) for r in rows)
    table(args.out/'screen_counts.csv',[dict(family=f,variant=v,status=s,count=k)
                                        for (f,v,s),k in sorted(counts.items())])
    rows=data(args.confirm/'measurements.json')
    groups=defaultdict(list)
    for r in rows:
        groups[(r['family'],r['n'],r['predictor'],r.get('variant'))].append(r)
    summary=[]
    for (fam,n,pred,var), matches in sorted(groups.items()):
        good=[r for r in matches if r['status']=='accepted']
        valid=len(good)==len(matches)
        summary.append(dict(family=fam,n=n,predictor=pred,variant=var,successful=len(good),
            repeats=len(matches),median_endpoint=median(r['endpoint_seconds'] for r in matches) if valid else None,
            median_predictor_plus_endpoint=median(r['endpoint_seconds']+r['predictor_seconds'] for r in matches) if valid else None,
            median_accepted_corrections=median(r['updates'] for r in good) if valid else None,
            max_global_cp_error=max((r['actual_error'] for r in good),default=None),
            max_candidate_certificate=max(r['bound'] for r in matches)))
    table(args.out/'confirmation.csv',summary)
    fig,axes=plt.subplots(2,2,figsize=(9.0,6.0),sharey='row',layout='constrained')
    manifest=data(args.screen/'manifest.json')
    for axrow,(fam,noise) in zip(axes,[('strong-D','Correlated dephasing'),('loss','Correlated loss')]):
        code=next(i for i,c in enumerate(manifest['configs']) if list(c)==[fam,2,4.,1e-7,0])
        path=f'{code:03d}_{fam}2_t4_tol1e-07_r0/B'
        for var,col,style in [('fixed','#254d82','o-'),('adaptive','#c05f16','s--'),('new','#18845c','D-')]:
            if fam=='strong-D' and var=='adaptive':
                continue
            info=worker_data(args.screen,f'{path}/{var}.json')
            events=[r for r in info['attempts'] if r['phase'] in ('carry','predictor','correction_trial')]
            best,values,times=float('inf'),[],[]
            for x in events:
                if x.get('bound') is None:
                    continue
                best=min(best,x['bound'])
                values.append(best)
                times.append(1000*x['elapsed_seconds'])
            for axis,time in zip(axrow,[range(len(values)),times]):
                axis.semilogy(time,values,style,markersize=3,linewidth=1.35,label=var,color=col)
        for axis in axrow:
            axis.axhline(1e-7,linewidth=.8,color='.45',ls=':')
            axis.grid(alpha=.2)
        axrow[0].set_ylabel(f'{noise}\nBest global certificate')
    axes[0][1].legend(loc='upper right',fontsize=8)
    axes[1][1].legend(loc='upper right',fontsize=8)
    axes[1][0].set_xlabel('Endpoint merit evaluation')
    axes[1][1].set_xlabel('Elapsed endpoint time (ms)')
    fig.suptitle('Two-spin frozen B predictor at T = 4')
    fig.savefig(args.out/'frozen_merit.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()
