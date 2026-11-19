"""Summarize every attempted configuration and build repeated measured-error frontiers."""
import argparse
import csv
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from statistics import median
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
from .benchmarks import csv_table

TARGETS=(1e-3,1e-5,1e-7)


def measurements(folder):
    """Read preserved worker measurements from an expanded run or its verified archive."""
    with (folder/'measurements.csv').open() as f:
        index=list(csv.DictReader(f))
    if (folder/'workers.zip').exists():
        archive=ZipFile(folder/'workers.zip')
    else:
        archive=nullcontext(None)
    result=[]
    with archive as z:
        for entry in index:
            member=f"{entry['directory']}/measurement.json"
            if entry['status']=='not_run_budget':
                result.append(dict(entry,global_cp_error=None,global_cp_bound=None))
                continue
            data=json.loads(z.read(member) if z else (folder/member).read_bytes())
            data.update(directory=entry['directory'])
            result.append(data)
    return result


def frontier(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[(row['family'],int(row['n']),row['method'],float(row['total']),int(row['steps']),
                int(row['chi']),int(row['work']),float(row['cert_tol']))].append(row)
    settings=[]
    for key,group in sorted(groups.items()):
        f,n,m,t,k,chi,work,cert=key
        reps=int(group[0]['repeats'])
        valid=len(group)==reps and len({r['repeat'] for r in group})==reps
        successes=[r for r in group if r['status']=='completed' and r.get('reached_time')==t]
        scored=[r for r in successes if r.get('global_cp_error') is not None and np.isfinite(r['global_cp_error'])]
        all_ok=valid and len(scored)==len(group)
        production=[r['setup_seconds']+r['integration_seconds'] for r in scored
                    if not r.get('native_scoring_and_io_inclusive')]
        settings.append(dict(family=f,n=n,method=m,total=t,steps=k,chi=chi,work=work,cert_tol=cert,
                             repeats=reps,observed=len(group),completed=len(successes),scored=len(scored),
                             statuses='|'.join(sorted({r['status'] for r in group})),
                             max_error=max((r['global_cp_error'] for r in scored),default=None),
                             all_completed_measured=all_ok,
                             median_worker_s=median(r['total_seconds'] for r in scored) if all_ok else None,
                             median_process_s=median(r['wall_seconds'] for r in scored) if all_ok else None,
                             median_setup_evolution_s=median(production) if all_ok and len(production)==len(group) else None,
                             median_peak_mib=median(r['peak_rss_mib'] for r in scored) if all_ok else None))
    winners=[]
    for f,n,m,t in sorted({(s['family'],s['n'],s['method'],s['total']) for s in settings}):
        for target in TARGETS:
            eligible=[s for s in settings if (s['family'],s['n'],s['method'],s['total'])==(f,n,m,t)
                      and s['repeats']>=3 and s['all_completed_measured'] and s['max_error']<=target]
            if eligible:
                fastest=min(eligible,key=lambda x:x['median_worker_s'])
                winners.append(dict(fastest,target=target,attained=True))
            else:
                winners.append(dict(family=f,n=n,method=m,total=t,target=target,attained=False))
    return settings,winners


def pack(folder):
    """Preserve all raw worker bytes and verify the archive before excluding loose duplicates from Git."""
    dirs=sorted(p for p in folder.iterdir() if p.is_dir() and p.name[:3].isdigit())
    if not dirs: return
    dest=folder/'workers.zip'
    files=[p for d in dirs for p in sorted(d.rglob('*')) if p.is_file()]
    if not dest.exists():
        with ZipFile(dest,'x',compression=ZIP_DEFLATED,compresslevel=6) as z:
            for path in files: z.write(path,path.relative_to(folder))
    with ZipFile(dest) as z:
        if set(z.namelist())!={str(p.relative_to(folder)) for p in files}:
            raise ValueError('Archive membership differs from raw worker files')
        for path in files:
            if z.read(str(path.relative_to(folder)))!=path.read_bytes(): raise ValueError(path)
    (folder/'.gitignore').write_text('# Byte-verified duplicates of workers.zip\n[0-9][0-9][0-9]_*/\n')
    (folder/'archive.sha256').write_text(hashlib.sha256(dest.read_bytes()).hexdigest()+'  workers.zip\n')


def plot(settings,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'A_dense_one':'A dense · one site','A_dense_two':'A dense · two sites','A_mpo_two':'A approximate MPO',
            'B_one':'B · one site','B_two':'B · two sites','B_witness':'B + witness',
            'native_witness':'original root + witness','correction_only':'correction only',
            'factor_explicit':'factor · explicit','factor_lsqr':'factor · LSQR'}
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.8),sharey=True,layout='constrained')
    for ax,f in zip(axes,('D','E')):
        for i,(method,label) in enumerate(labels.items()):
            data=sorted((s for s in settings if s['family']==f and s['n']==3 and s['repeats']>=3 and s['method']==method and s['all_completed_measured']),key=lambda s:s['steps'])
            if not data: continue
            ax.loglog([s['median_worker_s'] for s in data],[s['max_error'] for s in data],
                      marker=('o','s','^','D','v','P','X','*','<','>')[i],ms=5,lw=1,color=plt.cm.tab10(i),label=label)
        ax.set_title(f'{"Correlated circuit" if f=="D" else "Noncommuting parity"} · 3 spins')
        ax.set_xlabel('Median worker time, including scoring (s)')
        ax.grid(alpha=.24,which='both')
    axes[0].set_ylabel('Largest measured root error in three repeats')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=5,fontsize=8)
    fig.savefig(out/'matched_cost.png',dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--pack',action='store_true')
    args=p.parse_args()
    if args.pack: pack(args.run)
    if args.out:
        args.out.mkdir(parents=True,exist_ok=True)
        settings,choices=frontier(measurements(args.run))
        csv_table(args.out/'configurations.csv',settings)
        csv_table(args.out/'matched_accuracy.csv',choices)
        plot(settings,args.out)

if __name__=='__main__':
    main()
