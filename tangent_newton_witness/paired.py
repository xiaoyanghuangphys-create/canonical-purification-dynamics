"""Bounded isolated D/E comparisons, with immutable cohorts and optional checkpoint replay."""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import partial
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import traceback
from unittest.mock import patch
from zipfile import ZipFile, ZIP_DEFLATED

from galerkin_sylvester_tdvp import benchmarks as bm  # sets one-thread BLAS before NumPy
from galerkin_sylvester_tdvp import witness_pair_worker as previous
from galerkin_sylvester_tdvp.witness_block_stage import make_block_stage_floor
from galerkin_sylvester_tdvp.witness_precond_solver import solve_velocity
from galerkin_sylvester_tdvp.witness_pair_scaling import THREADS
from galerkin_sylvester_tdvp import witness_preconditioned_run as native
from galerkin_sylvester_tdvp.models import make_model, tn, witness, ts
import numpy as np
from scipy.linalg import eigvalsh

ROOT = bm.ROOT
SELECTION = ROOT/'galerkin_sylvester_tdvp/results/witness_preconditioned_20261119/selection.json'
LABELS = ('A-old','A-new','B-old','B-new','Old-only','New-only','A-fit','A-oldfit','A-newfit','A-newcorr','A-gn','GN-only')


def trained_tolerance(family, n, steps, tol):
    selection = json.loads(SELECTION.read_text())
    key = f'{family}{n:02d}_s{min(2,steps)}_tol{tol:g}'
    chosen = selection.get(key, {}).get('B_pre', '')
    return float(chosen.split('_r')[-1]) if chosen else 1e-9


def configuration(args, family, n, steps, tol, label):
    total = args.total if args.total else (1. if family=='E' else .1)
    levels = [[4,6,8],[6,8,12]] if family=='E' else [[4,6,10],[6,8,12],[8,12,16]]
    if family!='E' and tol<=1e-7:
        levels += [[10,14,20],[12,16,24]]
    predictor = label[0] if label[0] in 'AB' else None
    work = (8 if family=='E' else 10) if predictor!='B' else (16 if family=='E' else 64)
    variants = {'A-fit': ('new','none'), 'A-oldfit':('old','none'),
                'A-newfit':('new','old'), 'A-newcorr':('old','new'),'A-gn':('gn','gn'),'GN-only':('none','gn')}
    fit, correction = variants.get(label, ('old','old') if 'old' in label.lower() else ('new','new'))
    return dict(family=family,n=n,total=total,steps=steps,cert_tol=tol,label=label,predictor=predictor,
                fit=fit if predictor else 'none',correction=correction,levels=levels,mode=args.mode,
                chi=2 if family=='E' else 4,work=work,passes=8 if family=='E' else 4,rtol=trained_tolerance(family,n,steps,tol),
                atol=2e-13*2.**(-n/2),maxiter=80,max_elements=args.max_elements,timeout=args.timeout,
                memory_mib=args.memory_mib,freeze_step=args.freeze_step,new_rtol=args.new_rtol,native=args.native,preconditioner=args.preconditioner,checkpoint=str(args.checkpoint) if args.checkpoint else None)


def timed_call(row, key, fun, *args, **kwargs):
    tic=perf_counter()
    try:
        return fun(*args,**kwargs)
    finally:
        row[key] = row.get(key,0.)+perf_counter()-tic


def save_state(path, state, time, bound):
    factor, root, eta=state
    np.savez_compressed(path,time=time,bound=bound,eta=eta,
                        **{f'w{i}':a for i,a in enumerate(factor.tensors)},
                        **{f'x{i}':a for i,a in enumerate(root.tensors)})


def load_checkpoint(path, c, step, t):
    """Read a previously certified native-format trajectory endpoint and its provenance."""
    with ZipFile(path) as archive:
        origin=json.loads(archive.read('config.json'))
        for field in ('family','n','steps','total'):
            if origin[field]!=c[field]:
                raise ValueError(f'Checkpoint {field} differs from requested physical grid')
        with np.load(BytesIO(archive.read(f'accepted_{step}.npz')),allow_pickle=False) as arrays:
            if not np.isclose(float(arrays['time']),t,atol=1e-14,rtol=0):
                raise ValueError('Checkpoint time differs from replay origin')
            def train(name):
                cores=[arrays[f'{name}{i}'] for i in range(c['n'])]
                result=tn.MPS(c['n'],max_bond=max(a.shape[-1] for a in cores))
                result.tensors,result.center=cores,c['n']-1
                return result
            state=(train('w'),train('x'),float(arrays['eta']))
            return state,float(arrays['bound'])


def score(model, engine, carried, pred, state, target, t):
    ref,rho=model.reference(t)
    row = dict(physical_reference_discrepancy=tn.distance(target,rho),
               anchor_drift_frobenius=tn.distance(engine.physical,target),
               anchor_drift_relative_to_initial=tn.distance(engine.physical,target)/tn.norm(engine.physical),
               warm_root_cp_error=None,raw_predictor_cp_error=None,global_cp_error=None)
    if ref is not None:
        row.update(warm_root_cp_error=tn.distance(carried[1],ref),
                   raw_predictor_cp_error=tn.distance(pred,ref) if pred is not None else None,
                   global_cp_error=tn.distance(state[1],ref) if state is not None else None)
    if target.n<=8:
        spectral=eigvalsh(ts.matrix(target),ts.matrix(engine.physical))
        row.update(target_over_anchor_min=spectral[0],target_over_anchor_max=spectral[-1])
    return row


def prediction(c, x, t0, t1, bound, model, engine, floor, row, folder):
    if c['predictor']=='A':
        return witness.tdvp_prediction(x,t0,t1,engine,c['chi'],c['work'])
    if c['predictor']=='B':
        solver=partial(solve_velocity,anchor=engine.root_preconditioner,preconditioner='preparation')
        with patch.object(previous,'solve_velocity',solver):
            return previous._b_prediction(x,t0,t1,bound,model,floor,c,row,folder)
    return None


def physical_model(family, n, work):
    parity = family == 'E'
    model = make_model('E' if parity else 'D', n, work)
    if parity:
        return model, witness.WitnessAudit.parity_chain(n), witness.parity_power(n,.25)
    polarization, angle = (.35,.8) if family=='Dstress' else (.12,.35)
    engine = witness.WitnessAudit(n,polarization=polarization,angle=angle)
    factor = witness.prepared_power(n,.25,polarization,angle)
    if family=='Dstress':
        model.root = witness.prepared_power(n,.5,polarization,angle)
        model.floor = ((1-polarization)/2)**(n/2)
        model.parameters.update(polarization=polarization,angle=angle)
        def reference(t):
            target=witness.channel(engine.physical,t)
            if n>8: return None,target
            eigen,vectors=np.linalg.eigh(ts.matrix(target))
            return ts.matrix_to_mpo((vectors*np.sqrt(eigen))@vectors.conj().T,n),target
        model.reference=reference
    return model,engine,factor


def worker(c, folder):
    if c.get('native'):
        if c['label'] not in ('A-old','B-old'):
            raise ValueError('native reproduction requires A-old or B-old')
        measured=native.worker(dict(c,method=c['predictor'],preconditioner='preparation',residual_bond=24),folder)
        bm.save(folder/'worker.json',measured)
        return
    from .endpoint import run_endpoint
    start=perf_counter()
    folder=Path(folder)
    stop=bm.start_memory_watchdog(folder,c['memory_mib'],interval=.04)
    row=defaultdict(float, c, accepted_steps=0,reached_time=0.,status='running',
                    global_cp_bound=None,global_cp_error=None)
    try:
        tic=perf_counter()
        family,n=c['family'],c['n']
        model,engine,factor=physical_model(family,n,c['work'])
        row['model_parameters']=model.parameters
        state=witness.factor_to_root(factor,c['chi'])
        floor=make_block_stage_floor('E' if family=='E' else 'D',n,model) if c['predictor']=='B' else None
        check,_=previous._audit(engine,state[1],state[2],engine.physical,0.,c['cert_tol'])
        bound=check['bound']
        if bound>c['cert_tol']:
            raise ValueError('initial endpoint failed certificate')
        row['setup_seconds']=perf_counter()-tic
        grid=np.linspace(0,c['total'],c['steps']+1)
        
        start_step=0
        if c.get('checkpoint'):
            start_step=c['freeze_step']-1
            tic=perf_counter()
            state,bound=load_checkpoint(ROOT/c['checkpoint'],c,start_step,grid[start_step])
            audit,_=previous._audit(engine,state[1],state[2],witness.channel(engine.physical,grid[start_step]),
                                    grid[start_step],c['cert_tol'])
            if audit['bound']>c['cert_tol'] or abs(audit['bound']-bound)>2e-11:
                raise ValueError('Frozen input failed independent certificate replay')
            row.update(reached_time=grid[start_step],last_accepted_bound=bound,prefix_accepted_steps=start_step,
                       frozen_prefix_seconds=perf_counter()-tic)
        for i,(t0,t1) in enumerate(zip(grid[start_step:-1],grid[start_step+1:]),start_step+1):
            tic=perf_counter()
            frozen=c['freeze_step'] is not None
            # A carried root from the native old A trajectory supplies a common
            # warm start. The timed comparison then begins with the same arrays.
            if frozen and i<c['freeze_step']:
                native_cfg=dict(c,method='A')
                proposal=witness.tdvp_prediction(state[1],t0,t1,engine,c['chi'],8 if family=='E' else 10)
                state,info=previous._witness_endpoint(engine,state,proposal,t1,native_cfg,defaultdict(float),folder)
                if not info['accepted']:
                    raise ValueError(f'uncertified frozen checkpoint generator at {t1}')
                bound=info['bound']
                row['frozen_prefix_seconds']+=perf_counter()-tic
                row.update(reached_time=t1,last_accepted_bound=bound,prefix_accepted_steps=i)
                continue
            def save_progress(phase):
                bm.save(folder/'progress.json',dict(row,phase=phase,step=i,trial_time=float(t1)))
            save_progress('predictor')
            carried=state
            pred=timed_call(row,'predictor_seconds',prediction,c,state[1],t0,t1,bound,model,engine,floor,row,folder)
            target=timed_call(row,'physical_density_seconds',witness.channel,engine.physical,t1)
            save_progress('witness')
            state,info=timed_call(row,'witness_seconds',run_endpoint,engine,carried,pred,target,t1,
                                  levels=c['levels'],tolerance=c['cert_tol'],fit=c['fit'],
                                  correction=c['correction'],mode=c['mode'],passes=c['passes'],
                                  rtol=c['new_rtol'],max_elements=c['max_elements'],preconditioner=c['preconditioner'],
                                  on_event=lambda event: bm.append_event(folder/'attempts.jsonl',dict(step=i,**event)))
            bm.append_event(folder/'endpoints.jsonl',dict(step=i,time=t1,**{k:v for k,v in info.items() if k!='time'}))
            actual=timed_call(row,'scoring_seconds',score,model,engine,carried,pred,state if info['accepted'] else None,target,t1)
            if frozen:
                def fingerprint(s):
                    h=hashlib.sha256()
                    for core in s.tensors:
                        h.update(str(core.shape).encode()); h.update(np.ascontiguousarray(core).tobytes())
                    return h.hexdigest()
                actual.update(carried_factor_sha256=fingerprint(carried[0]),carried_root_sha256=fingerprint(carried[1]),
                              prediction_sha256=fingerprint(pred) if pred is not None else None,target_sha256=fingerprint(target))
            bm.append_event(folder/'scores.jsonl',dict(step=i,time=t1,**actual))
            if not info['accepted']:
                row.update(status='witness_uncertified',failed_bound=info['bound'],rollback=info['rollback'])
                break
            bound=info['bound']
            if actual['global_cp_error'] is not None and actual['global_cp_error']>bound+2e-11:
                raise ValueError('independent CP error exceeded conditional certificate')
            row.update(accepted_steps=row['accepted_steps']+1,reached_time=t1,last_accepted_bound=bound,
                       root_bond=state[1].bond_dimension(),factor_bond=state[0].bond_dimension())
            timed_call(row,'checkpoint_seconds',save_state,folder/f'accepted_{i}.npz',state,t1,bound)
            if (frozen and i==c['freeze_step']) or i==c['steps']:
                row.update(status='completed',global_cp_bound=bound,global_cp_error=actual['global_cp_error'])
                break
    except Exception as exc:
        row.update(status=getattr(exc,'status','failed'),error=f'{type(exc).__name__}: {exc}')
        (folder/'exception.txt').write_text(traceback.format_exc())
    finally:
        stop.set()
    row['total_seconds']=perf_counter()-start
    row['production_seconds']=row['total_seconds']-row['scoring_seconds']-row['frozen_prefix_seconds']
    row['peak_rss_mib']=bm.memory_mib()
    bm.save(folder/'worker.json',row)


def source_hashes():
    packages=['galerkin_sylvester_tdvp','scalable_cp_witness/code','factor_tdvp','factor_tdvp_accelerated',
              'factor_tdvp_comparison','tangent_newton_witness']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for package in packages for p in sorted((ROOT/package).glob('*.py'))}


def run(args):
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    configs=[configuration(args,f,n,s,t,m) for f in args.families for n in args.sizes for t in args.tolerances
             for s in args.steps for m in args.labels]
    hashes=source_hashes()
    git=lambda *words:subprocess.check_output(['git',*words],cwd=ROOT,text=True).strip()
    bm.save(out/'manifest.json',dict(source_sha=git('rev-parse','HEAD'),base_sha=git('rev-parse','origin/research/galerkin-sylvester-tdvp'),
            source_files_sha256=hashes,hardware=subprocess.run(['sysctl','-n','machdep.cpu.brand_string'],capture_output=True,text=True).stdout.strip() if sys.platform=='darwin' else platform.processor(),python=sys.version,platform=platform.platform(),machine=platform.machine(),
            numpy=np.__version__,scipy=bm.scipy.__version__,blas=np.__config__.show(mode='dicts')['Build Dependencies']['blas'],
            threads=THREADS,argv=sys.argv,seed=None,cohort=args.phase,grid=configs,repeats=args.repeats))
    if args.checkpoint and (not args.freeze_step or args.freeze_step<2):
        raise ValueError('--checkpoint requires a positive previous endpoint index')
    if args.freeze_step and (args.freeze_step<1 or any(s<args.freeze_step for s in args.steps)):
        raise ValueError('freeze-step must index an endpoint in each grid')
    for rep in range(1,args.repeats+1):
        for index,cfg in enumerate(configs if rep%2 else reversed(configs)):
            folder=out/f'r{rep:02d}_{index:03d}_{cfg["family"]}{cfg["n"]}_{cfg["label"]}'
            folder.mkdir()
            cfg=dict(cfg,repeat=rep)
            bm.save(folder/'config.json',cfg)
            fingerprint=source_hashes()
            tic=perf_counter()
            with (folder/'worker.log').open('w') as log:
                try:
                    process=subprocess.run([sys.executable,'-m','tangent_newton_witness.paired','--worker',str(folder)],
                        cwd=ROOT,env=dict(os.environ,**THREADS),timeout=cfg['timeout'],stdout=log,stderr=subprocess.STDOUT)
                    code=process.returncode
                except subprocess.TimeoutExpired:
                    code=None
            wall=perf_counter()-tic
            if code==0 and (folder/'worker.json').exists():
                summary=json.loads((folder/'worker.json').read_text())
            else:
                summary=dict(cfg,status='timeout' if code is None else 'worker_failure',returncode=code,
                             global_cp_bound=None,global_cp_error=None)
                if (folder/'progress.json').exists():
                    progress=json.loads((folder/'progress.json').read_text())
                    summary.update({k:progress[k] for k in ('phase','reached_time','accepted_steps','trial_time')})
                if (folder/'memory_limit_event.json').exists():
                    summary.update(json.loads((folder/'memory_limit_event.json').read_text()))
            changed={k:v for k,v in fingerprint.items() if hashes.get(k)!=v}
            changed_during_run={k:v for k,v in source_hashes().items() if fingerprint.get(k)!=v}
            summary.update(wall_seconds=wall,worker=folder.name,phase=args.phase,source_deltas=changed,
                           source_deltas_during_worker=changed_during_run)
            bm.append_event(out/'measurements.jsonl',summary)
            print(folder.name,summary['status'],summary.get('global_cp_bound'),round(wall,3),flush=True)
            # Publish the worker inputs, diagnostics, and checkpoints together.
            # Keep only the validated ZIP in git, leaving the originals intact locally.
            with ZipFile(folder.with_suffix('.zip'),'x',ZIP_DEFLATED) as z:
                for p in sorted(folder.iterdir()):
                    z.write(p,p.name)
            with ZipFile(folder.with_suffix('.zip')) as z:
                if z.testzip() is not None: raise IOError('failed worker archive verification')
            with (out/'.gitignore').open('a') as stream: stream.write('/'+folder.name+'/\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path); p.add_argument('--native',action='store_true',help='reproduce the unmodified earlier A/B worker')
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--labels',nargs='+',choices=LABELS,default=LABELS[:6])
    p.add_argument('--families',nargs='+',choices=['D','E','Dstress'],default=['D','E'])
    p.add_argument('--sizes',nargs='+',type=int,default=[8]); p.add_argument('--steps',nargs='+',type=int,default=[1])
    p.add_argument('--total',type=float); p.add_argument('--tolerances',nargs='+',type=float,default=[1e-5,1e-7])
    p.add_argument('--timeout',type=float,default=30); p.add_argument('--memory-mib',type=float,default=768)
    p.add_argument('--max-elements',type=int,default=2_000_000); p.add_argument('--new-rtol',type=float,default=1e-8)
    p.add_argument('--preconditioner',choices=['none','jacobi'],default='none')
    p.add_argument('--mode',choices=['one','two'],default='two'); p.add_argument('--phase',default='screen')
    p.add_argument('--checkpoint',type=Path,help='replay a certified previous endpoint in this worker ZIP')
    p.add_argument('--freeze-step',type=int); p.add_argument('--repeats',type=int,default=1)
    args=p.parse_args()
    if args.worker: worker(json.loads((args.worker/'config.json').read_text()),args.worker)
    elif args.out: run(args)
    else: p.error('supply --out')


if __name__=='__main__':
    main()
