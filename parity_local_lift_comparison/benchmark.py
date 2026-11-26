"""Isolated parity workers, explicit grids, and independent raw canonical-root scoring."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import traceback
from zipfile import ZipFile, ZIP_DEFLATED

from galerkin_sylvester_tdvp import benchmarks as bm
from galerkin_sylvester_tdvp.witness_pair_scaling import THREADS
import numpy as np
import scipy

PKG=Path(__file__).resolve().parent
ROOT=PKG.parent
LOCAL=('Half-Petz','Local-fit')
RAW=('A-raw','B-raw')
WITNESS=('A+witness','B+witness','Witness-only')
METHODS=RAW+WITNESS+LOCAL
TARGETS=(1.e-2,1.e-3,1.e-5,1.e-6,1.e-7)
BASE='23fbd4c89a630688e65d441d9edb6a0056808d21'


def config(method,n=8,steps=1,total=1.,**kwargs):
    times=kwargs.pop('times',np.linspace(0.,total,steps+1)[1:].tolist())
    options=dict(method=method,n=n,p=.6,b0=.4,times=times,
                 cert_tol=1.e-6,chi=4 if method in LOCAL else 2,
                 width=1,fit_width=1,protocol='active' if method in LOCAL else 'generator',
                 completion='native',audit_inactive=False,max_elements=2_000_000,
                 memory_mib=768,timeout=45.,save_tensors=False)
    options.update(kwargs)
    return options


def grid(name):
    if name=='smoke':
        return [config(m,4) for m in METHODS]+[config(m,4,2,protocol='uniform') for m in LOCAL]
    if name=='width':
        rows=[]
        for w in (1,2,3,5,7):
            rows.append(config('Half-Petz',8,width=w,fit_width=w))
            for r in sorted({w,min(8,w+1)}):
                rows.append(config('Local-fit',8,width=w,fit_width=r))
        for w,r in [(1,4),(2,4),(3,4),(4,4)]:
            rows.append(config('Local-fit',4,width=w,fit_width=r))
        rows.extend(config('Half-Petz',4,width=w,fit_width=w) for w in (3,4))
        return rows
    if name=='refine':
        return [config(m,6,k,protocol=pr,width=2,fit_width=3 if m=='Local-fit' else 2)
                for k in (1,2,5,10,20,50,100) for pr in ('active','uniform') for m in LOCAL]
    if name=='horizon':
        return [config(m,6,5,t) for t in (.1,.3,1.,2.,4.,8.) for m in METHODS]
    if name=='controls':
        return [config(m,6,2,p=p,b0=b) for p,b in ((0.,.4),(.6,0.),(.08,.4),(.88,.4)) for m in METHODS]
    if name=='scale':
        rows=[]
        for n in (8,16,32,64,128,256,512):
            for m in METHODS:
                rows.append(config(m,n,timeout=75. if n>=256 else 30.))
            rows.extend(config(m,n,protocol='uniform',timeout=45.) for m in LOCAL)
        return rows
    if name=='historical-grid':
        return [config(m,n,times=[.02,.1,.3,.6,1.],timeout=120.) for n in (8,128,512)
                for m in ('A+witness','Witness-only','Half-Petz','Local-fit')]
    if name in ('screen','confirm'):
        ns=(8,32) if name=='screen' else (8,)
        rows=[]
        for n in ns:
            rows.extend(config(m,n,k) for k in (1,5,20) for m in RAW)
            rows.extend(config(m,n,cert_tol=tol) for tol in (1e-2,1e-3,1e-5,1e-6,1e-7) for m in WITNESS)
            rows.extend(config(m,n,protocol=pr) for pr in ('active','uniform') for m in LOCAL)
        return rows
    raise ValueError(name)


def dump(path,record):
    bm.save(path,record)


def timer_experiment(c,out):
    from .oracles import root_mpo,score
    local=c['method'] in LOCAL
    if local:
        from .local import LocalLift
    else:
        from .global_methods import GlobalEvolution
    t0=perf_counter()
    row=dict(status='running',last_accepted_time=0.,completed_intervals=0,
             setup_seconds=0.,update_seconds=0.,checkpoint_seconds=0.,scoring_seconds=0.,
             max_accepted_cp_error=0.,global_cp_error=None,global_cp_bound=None)
    scores=[]
    try:
        if local:
            factor=None
            solver=LocalLift(root_mpo(c['n'],c['p'],c['b0']),method=c['method'],
                             width=c['width'],fit_width=c['fit_width'],protocol=c['protocol'],
                             chi=c['chi'],completion=c['completion'],audit_inactive=c['audit_inactive'],
                             max_elements=c['max_elements'])
        else:
            solver=GlobalEvolution(c,out)
        row['setup_seconds']=perf_counter()-t0
        if local:
            row['setup_cache_seconds']=solver.setup_cache_seconds
            row['guarded_elements']=solver.guarded_elements
        times=[float(t) for t in c['times']]
        if not times or any(t<=l for l,t in zip([0.]+times,times)):
            raise ValueError('Require ordered positive endpoints')
        for i,(left,right) in enumerate(zip([0.]+times,times),1):
            tic=perf_counter()
            try:
                if local:
                    diag=solver.step(right-left)
                    state,pred=solver.state,None
                else:
                    state,diag,pred=solver.step(left,right)
                    factor=getattr(solver,'factor',None)
            finally:
                row['update_seconds']+=perf_counter()-tic
            tic=perf_counter()
            try:
                check=score(state,right,c['p'],c['b0'])
                error=float(check['global_cp_error'])
                if not np.isfinite(error):
                    raise ArithmeticError('Nonfinite global score')
                if pred is not None:
                    raw_error=error if pred is state else float(score(pred,right,c['p'],c['b0'])['global_cp_error'])
                else:
                    raw_error=error if c['method'] in RAW else None
                bound=diag.get('bound')
                if c['method'] in WITNESS or 'Newton' in c['method']:
                    if bound is None or not np.isfinite(bound) or bound>c['cert_tol'] or error>bound+2e-10:
                        raise ArithmeticError(f'Endpoint bound failed independent audit: {bound}, actual {error}')
            finally:
                row['scoring_seconds']+=perf_counter()-tic
            tic=perf_counter()
            endpoint=dict(time=right,step=i,actual=check,diagnostics=diag,predictor_raw_cp_error=raw_error)
            endpoint['spatial_root_bonds']=[a.shape[-1] for a in state.tensors[:-1]]
            if factor is not None:
                endpoint['spatial_factor_bonds']=[a.shape[-1] for a in factor.tensors[:-1]]
            bm.append_event(out/'checkpoints.jsonl',endpoint)
            scores.append(error)
            row.update(last_accepted_time=right,completed_intervals=i,max_accepted_cp_error=max(scores),
                       last_error=error,last_bound=bound,root_bond=state.bond_dimension(),
                       factor_bond=factor.bond_dimension() if factor is not None else None)
            if c['save_tensors'] and i==len(times):
                arrays={f'x{j}':a for j,a in enumerate(state.tensors)}
                if factor is not None:
                    arrays.update({f'g{j}':a for j,a in enumerate(factor.tensors)})
                np.savez_compressed(out/'accepted.npz',time=right,**arrays)
            dump(out/'progress.json',row)
            row['checkpoint_seconds']+=perf_counter()-tic
        row.update(status='completed',global_cp_error=row['last_error'],global_cp_bound=row['last_bound'])
    except Exception as exc:
        row.update(status=getattr(exc,'status',None) or 'failed',
                   reason=f'{type(exc).__name__}: {exc}',details=getattr(exc,'details',{}))
        (out/'exception.txt').write_text(traceback.format_exc())
    row['worker_total_seconds']=perf_counter()-t0
    row['production_seconds']=row['worker_total_seconds']-row['scoring_seconds']
    row['peak_rss_mib']=bm.memory_mib()
    if 'solver' in locals():
        row['adapter_timings']=getattr(solver,'timings',{})
    dump(out/'result.json',row)


def worker(c,out):
    stop=bm.start_memory_watchdog(out,c['memory_mib'],interval=.04)
    try:
        timer_experiment(c,out)
    finally:
        stop.set()


def eligible(r,tol):
    return (r.get('status')=='completed' and r.get('last_accepted_time')==r['times'][-1]
            and r.get('global_cp_error') is not None and 0<=r['global_cp_error']<=tol
            and r.get('production_seconds',0)>0)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    directories=['parity_local_lift_comparison','galerkin_sylvester_tdvp','scalable_cp_witness/code',
                 'half_petz_evolution','tilted_half_petz_comparison','factor_tdvp_comparison',
                 'local_cp_large','factor_tdvp','factor_tdvp_accelerated','cp_tdvp']
    return {str(p.relative_to(ROOT)):sha(p) for d in directories for p in sorted((ROOT/d).glob('*.py'))}


def run(out,configs,reps,suite,argv):
    out=out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    cache=out/'workers';cache.mkdir()
    (out/'.gitignore').write_text('/workers/\n')
    sources_at_start=sources()
    dump(out/'manifest.json',dict(suite=suite,repeats=reps,configs=configs,argv=argv,threads=THREADS,
         source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         base_sha=BASE,source_files_sha256=sources_at_start,python=sys.version,numpy=np.__version__,
         scipy=scipy.__version__,platform=platform.platform(),machine=platform.machine(),
         blas=str(getattr(np.__config__,'CONFIG',{})),seed='deterministic physical inputs',
         order='Forward on odd repetitions and reversed on even repetitions'))
    rows=[]
    with ZipFile(out/'workers.zip','x',ZIP_DEFLATED) as archive:
        for repeat in range(1,reps+1):
            order=list(enumerate(configs))
            if repeat%2==0:
                order.reverse()
            for idx,c in order:
                key=f'{idx:03d}_r{repeat}'
                folder=cache/key;folder.mkdir()
                dump(folder/'config.json',c)
                before=sources()
                tic=perf_counter()
                with (folder/'stdout.log').open('w') as stream:
                    try:
                        pr=subprocess.run([sys.executable,'-m','parity_local_lift_comparison.benchmark',
                                     '--worker',str(folder)],cwd=ROOT,env=dict(os.environ,**THREADS),
                                     stdout=stream,stderr=subprocess.STDOUT,timeout=c['timeout'])
                        status='worker_exit_'+str(pr.returncode)
                    except subprocess.TimeoutExpired:
                        status='timeout'
                wall=perf_counter()-tic
                if (folder/'result.json').exists():
                    row=json.loads((folder/'result.json').read_text())
                else:
                    row=dict(status=status,last_accepted_time=0.,completed_intervals=0)
                    if (folder/'progress.json').exists():
                        row.update(json.loads((folder/'progress.json').read_text()),status=status)
                    if (folder/'memory_limit_event.json').exists():
                        row.update(json.loads((folder/'memory_limit_event.json').read_text()))
                    row.update(global_cp_error=None,global_cp_bound=None)
                row.update(c,repeat=repeat,config_index=idx,worker=key,wall_seconds=wall,
                           sources_changed_during_worker=[k for k,v in sources().items() if before.get(k)!=v],
                           source_deltas={k:v for k,v in before.items() if sources_at_start.get(k)!=v})
                if 'worker_total_seconds' in row:
                    row['process_overhead_seconds']=wall-row['worker_total_seconds']
                bm.append_event(out/'measurements.jsonl',row)
                for f in sorted(folder.iterdir()):
                    if f.is_file():archive.write(f,key+'/'+f.name)
                rows.append(row)
                print(key,c['method'],c['n'],row['status'],row.get('global_cp_error'),round(wall,3),flush=True)
    with ZipFile(out/'workers.zip') as archive:
        if archive.testzip() is not None:raise IOError('Invalid worker archive')
    dump(out/'archive.json',dict(workers=len(rows),sha256=sha(out/'workers.zip')))
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worker',type=Path)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--suite',default='smoke',choices=('smoke','width','refine','horizon','controls','scale','historical-grid','screen','confirm'))
    ap.add_argument('--grid',type=Path,help='JSON list of full configs or a recorded manifest')
    ap.add_argument('--repeats',type=int,default=1)
    ap.add_argument('--sizes',nargs='+',type=int)
    ap.add_argument('--methods',nargs='+')
    args=ap.parse_args()
    if args.worker:
        worker(json.loads((args.worker/'config.json').read_text()),args.worker)
        return
    if not args.out or args.repeats<1:ap.error('Require --out and positive repeats')
    configs=grid(args.suite)
    if args.grid:
        data=json.loads(args.grid.read_text());configs=data['configs'] if isinstance(data,dict) else data
    configs=[c for c in configs if (not args.sizes or c['n'] in args.sizes) and
            (not args.methods or c['method'] in args.methods)]
    if not configs:ap.error('Empty selected grid')
    run(args.out,configs,args.repeats,args.suite,sys.argv)


if __name__=='__main__':
    main()
