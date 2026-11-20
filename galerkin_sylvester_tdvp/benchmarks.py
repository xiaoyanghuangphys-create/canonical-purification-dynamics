"""Bounded independent processes, immutable output directories and prespecified grids."""
from __future__ import annotations
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
import uuid

for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/galerkin-mpl')
import numpy as np
import scipy
from factor_tdvp_accelerated.repeated_scaling import start_memory_watchdog, native_witness
from factor_tdvp_accelerated.run_benchmarks import memory_mib
from factor_tdvp import factor_tdvp as fd
from factor_tdvp import rank_bridge
from factor_tdvp.baseline import _legacy_module
from .models import make_model, parity_floor_setup, tn, mpo, ts, witness

ROOT=Path(__file__).resolve().parents[1]
BASE='b337e8b7fb86fc3cccec64e13e2ddb35cc14c4e1'
OBJECTIVES=dict(A_dense_one='Euclidean root projection, dense Sylvester',A_dense_two='Euclidean root projection, dense Sylvester',A_mpo_two='Euclidean root projection, approximate two-pass MPO force',B_one='one-site Sylvester metric',B_two='two-site Sylvester metric',B_witness='two-site Sylvester metric + positive endpoint witness',native_witness='native approximate root + positive endpoint witness',correction_only='native positive endpoint correction',factor_explicit='density residual in factor tangent, real SVD',factor_lsqr='density residual in factor tangent, MPO LSQR')

def serial(x):
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)

def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,default=serial,allow_nan=False,indent=1)+'\n')
    temp.replace(path)

def csv_table(path, rows):
    if rows:
        keys=list(dict.fromkeys(k for row in rows for k in row))
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,keys)
            writer.writeheader()
            writer.writerows(rows)

def append_event(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row,default=serial,allow_nan=False)+'\n')

def config(family,n,method,steps=2,chi=4,work=16,total=.1,repeats=1,**extra):
    return dict(family=family,n=n,method=method,steps=steps,chi=chi,work=work,total=total,
                repeats=repeats,rtol=1e-9,atol=2e-13*2.**(-n/2),cert_tol=1e-5,maxiter=80,**extra)

def grid(suite):
    if suite=='smoke':
        return [config('D',2,m) for m in ('A_dense_one','B_one','B_two')]
    if suite in ('bench','baselines','galerkin'):
        rows=[]
        for f in ('D','E'):
            for k in (2,6):
                for m in OBJECTIVES:
                    rows.append(config(f,3,m,steps=k,work=8,repeats=3))
        if suite=='baselines': return [r for r in rows if not r['method'].startswith('B_')]
        if suite=='galerkin': return [r for r in rows if r['method'].startswith('B_')]
        return rows
    if suite=='comparison':
        rows=grid('bench')
        # Prespecified follow-up grid after the pilot, common T and endpoint counts.
        for family in ('D','E'):
            for method in OBJECTIVES:
                c=config(family,3,method,steps=20,work=8,repeats=3)
                c['cert_tol']=1e-7
                rows.append(c)
        return rows
    if suite=='physics':
        rows=[config(f,n,m,steps=4,total=.03,chi=4,work=32)
              for n in (2,3,4,5) for f in ('loss_mixed','loss_pure','pair_mixed','pair_pure')
              for m in ('B_one','B_two','A_dense_two')]
        rows.extend(config(f,3,m,steps=4,chi=4) for f in ('product','commuting','coherent','cluster')
                    for m in ('B_one','B_two','A_dense_two'))
        return rows
    if suite=='scaling':
        return [config(f,n,'B_two',steps=2,chi=4 if f=='D' else 2,work=16,
                       total=.1 if f=='D' else 1.) for n in (8,16,32,64) for f in ('D','E')]
    if suite=='refine':
        return [config('loss_mixed',4,'B_two',steps=k,total=.03,chi=b,work=32) for k in (2,4,8) for b in (2,4,8,16)]
    raise ValueError(suite)

def score(state, model, t, family):
    reference,rho=model.reference(t)
    density=mpo.multiply(state,mpo.adjoint(state))
    herm=tn.distance(state,mpo.adjoint(state))
    result=dict(global_cp_error=None if reference is None else tn.distance(state,reference),
                global_cp_bound=None,density_error=tn.distance(density,rho),
                hermiticity_defect=herm,trace_defect=abs(tn.norm(state)**2-1),
                root_bond=state.bond_dimension())
    if state.n<=8:
        a=ts.matrix(state)
        U,s,_=np.linalg.svd(a,full_matrices=False)
        eig=np.linalg.eigvalsh((a+a.conj().T)/2)
        result.update(min_root_eigenvalue=eig[0],negative_part_norm=np.linalg.norm(np.minimum(eig,0)),
                      polar_defect=np.linalg.norm(a-(U*s)@U.conj().T))
    elif reference is None and family=='D':
        # Distance to a known positive initial root bounds the negative part of Herm(X).
        # For a unital channel rho(t)>=lambda_min(rho0)*I. Bound applies even
        # when the independently scored raw state has a small Hermiticity defect.
        x,_=tn.compress(tn.add(state,mpo.adjoint(state),.5,.5),2*state.bond_dimension(),tolerance=0.)
        eta=tn.distance(x,model.root)
        residual=tn.distance(mpo.multiply(x,x),rho)
        result.update(global_cp_bound=tn.distance(x,state)+2*eta+residual/model.floor,
                      bound_kind='positive initial witness + unital physical spectral floor; floating-point')
    return result


def source_hashes():
    here=Path(__file__).parent
    return {name: hashlib.sha256((here/name).read_bytes()).hexdigest()
            for name in ('benchmarks.py','models.py','galerkin_solver.py','tangent_frame.py','evolution.py')}

def worker(c,out):
    stop=start_memory_watchdog(out,c['memory_mib'],interval=.04)
    begin=perf_counter()
    row=dict(c,source_files_sha256=source_hashes(),status='failed',global_cp_error=None,global_cp_bound=None,accepted_steps=0,rejected_steps=0,
             reached_time=0.,objective=OBJECTIVES[c['method']],scoring_seconds=0.)
    try:
        method,f,n,chi,work=c['method'],c['family'],c['n'],c['chi'],c['work']
        times=np.linspace(0,c['total'],c['steps']+1)
        model=make_model(f,n,work)
        if model.pure: times=c['total']*np.linspace(0,1,c['steps']+1)**2
        row.update(parameters=model.parameters,clock='sqrt' if model.pure else 'physical')
        state,compression=tn.compress(model.root,chi,normalize=True)
        row.update(initial_global_cp_error=tn.distance(state,model.root),initial_discard= compression['discarded_bound'])
        history=[]
        row['initial_root_bond']=state.bond_dimension()
        def event(info):
            append_event(out/'events.jsonl',info)
        if method.startswith('B_'):
            from .evolution import integrate, KrausStartup, PositiveWitness
            opts=dict(mode='one' if method=='B_one' else 'two',clock=row['clock'],
                      dense_audit_max_n=6, initial_lower_bound=model.floor-row['initial_global_cp_error'],
                      solver_options=dict(rtol=c['rtol'],atol=c['atol'],maxiter=c['maxiter'],work_bond=work),
                      on_event=event)
            if model.pure:
                opts.update(seed=KrausStartup(*model.dense_physics()),initial_lower_bound=None)
            if f=='E' and n>6:
                opts['stage_lower_bound']=parity_floor_setup(n)
            if method=='B_witness':
                engine=witness.WitnessAudit.parity_chain(n) if f=='E' else witness.WitnessAudit(n)
                factor=witness.parity_power(n,.25) if f=='E' else witness.prepared_power(n,.25)
                opts['witness']=PositiveWitness(engine,factor,chi,work,tolerance=c['cert_tol'])
            row['setup_seconds']=perf_counter()-begin
            state,info=integrate(state,model.generator,times,chi,**opts)
            save(out/'integration.json',info)
            row.update(status=info['status'],reached_time=info['final_time'],accepted_steps=info['accepted_steps'],
                       rejected_steps=info['rejected_steps'])
            history=info['history']
            for name in ('force_seconds','retraction_seconds','audit_seconds','witness_seconds','seed_seconds'):
                row[name]=info[name]
            if method=='B_witness' and history and 'witness' in history[-1]:
                row['global_cp_bound']=history[-1]['witness']['cp_bound']
                row['factor_bond']=history[-1]['witness']['factor_bond']
            if history and not history[-1].get('accepted',False):
                row['error']=history[-1].get('error')
            row['generator_max_local_discard']=max(model.generator.discards,default=0.)
            for key in ('error','message','failure','reason'):
                if key in info: row[key]=info[key]
        elif method.startswith('A_dense'):
            legacy=_legacy_module()
            h,jumps=model.dense_physics()
            force=legacy.DenseForce(n,rank_bridge.lindblad_superoperator(h,jumps),clock=row['clock'])
            def checked_force(t,y):
                v,info=force(t,y)
                if info['null_forcing']>1e-10: raise ArithmeticError('unsupported kernel forcing')
                return v,info
            row['setup_seconds']=perf_counter()-begin
            for i,(a,b) in enumerate(zip(times[:-1],times[1:])):
                if model.pure and i==0:
                    seed=legacy.positive_start_root(ts.matrix(state),h,jumps,b)
                    state,re=tn.compress(ts.matrix_to_mpo(seed,n),chi,normalize=True)
                    detail=dict(kind='positive_seed',discarded_bound=re['discarded_bound'])
                else:
                    t0,t1=(np.sqrt(a),np.sqrt(b)) if model.pure else (a,b)
                    state,detail=tn.midpoint_step(state,t0,t1-t0,checked_force,chi,retract=legacy.polar_retractor(n,chi),
                                                  mode=method.split('_')[-1])
                history.append(detail)
                row.update(accepted_steps=i+1,reached_time=b)
                event(dict(time=b,step=i+1,diagnostics=detail))
            row['status']='completed'
        elif method=='A_mpo_two':
            engine=witness.WitnessAudit.parity_chain(n) if f=='E' else witness.WitnessAudit(n)
            row['setup_seconds']=perf_counter()-begin
            for i,(a,b) in enumerate(zip(times[:-1],times[1:])):
                state=witness.tdvp_prediction(state,a,b,engine,chi,work)
                row.update(accepted_steps=i+1,reached_time=b)
                event(dict(time=b,step=i+1))
            row['status']='completed'
        elif method.startswith('factor'):
            factor,_=ts.normalize_factor(ts.matrix_to_mpo(fd.positive_power(ts.matrix(model.root),.5),n,chi))
            row['initial_global_cp_error']=tn.distance(ts.root_mpo(factor),model.root)
            h,jumps=model.dense_physics()
            options={}
            if method=='factor_lsqr':
                from factor_tdvp_accelerated.matrix_free_solver import factor_velocity
                options.update(velocity_solver=factor_velocity,solver_options=dict(tolerance=c['rtol'],maxiter=c['maxiter']))
            row['setup_seconds']=perf_counter()-begin
            # Stream accepted physical-clock steps so a later failure preserves progress.
            for i,(a,b) in enumerate(zip(times[:-1],times[1:])):
                factor,batch=fd.evolve(factor,h,jumps,[a,b],chi,work_bond=work,**options)
                history.extend(batch)
                event(batch[0])
                row.update(accepted_steps=i+1,reached_time=b)
            state=ts.root_mpo(factor)
            row.update(status='completed',accepted_steps=len(history),reached_time=times[-1],factor_bond=factor.bond_dimension())
        else:
            row['setup_seconds']=perf_counter()-begin
            # Original driver performs independent scoring internally; keep native inclusive timing.
            raw=native_witness(dict(n=n,family=f,steps=c['steps'],total=c['total'],tolerance=c['cert_tol'],
                               method='witness' if method=='native_witness' else 'correction_only'),out)
            row.update(raw,native_scoring_and_io_inclusive=True,
                       native_levels=[[4,6,8],[6,8,12]] if f=='E' else [[4,6,10],[6,8,12],[8,12,16]])
            filename=next(out.glob('*_last_accepted.npz'))
            with np.load(filename) as checkpoint:
                state=tn.MPS(n,max_bond=raw['root_bond'])
                state.tensors=[checkpoint[f'x{j}'] for j in range(n)]
                state.center=n-1
        row['integration_seconds']=perf_counter()-begin-row['setup_seconds']
        if np.isclose(row['reached_time'],times[-1],rtol=0,atol=1e-14) and row['status'] in ('completed','ok','success'):
            row['status']='completed'
            start=perf_counter()
            endpoint=score(state,model,times[-1],f)
            if row.get('global_cp_bound') is not None and endpoint['global_cp_bound'] is None:
                endpoint['global_cp_bound']=row['global_cp_bound']
            row.update(endpoint)
            row['scoring_seconds']=perf_counter()-start
        else:
            row['global_cp_error']=None
            row['global_cp_bound']=None
        def values(term,obj):
            if isinstance(obj,dict):
                for k,v in obj.items():
                    if k==term and isinstance(v,(int,float,np.number)): yield v
                    elif isinstance(v,(dict,list)): yield from values(term,v)
            elif isinstance(obj,list):
                for x in obj: yield from values(term,x)
        for key in ('galerkin_residual_abs','galerkin_residual_rel','projected_residual_abs','physical_residual_abs','physical_residual_rel','orthogonal_residual_abs','compression_discard_bound','discarded_bound','svd_discard_norm','discarded_weight','condition_estimate','peak_product_bond','force_residual'):
            numbers=list(values(key,history))
            if numbers: row['max_'+key]=max(numbers)
        for key in ('iterations','solve_seconds','preconditioner_setup_seconds','correction_seconds'):
            numbers=list(values(key,history))
            if numbers: row['total_'+key]=sum(numbers)
    except Exception as exc:
        (out/'exception.txt').write_text(traceback.format_exc())
        row.update(status=getattr(exc,'status','failed'),error=f'{type(exc).__name__}: {exc}',rejected_steps=1)
        row['global_cp_error']=row['global_cp_bound']=None
    finally:
        stop.set()
    row.update(total_seconds=perf_counter()-begin,peak_rss_mib=memory_mib())
    save(out/'measurement.json',row)


def execute(c,out,timeout):
    start=perf_counter()
    cmd=[sys.executable,'-m','galerkin_sylvester_tdvp.benchmarks','--worker',json.dumps(c),'--out',str(out)]
    try:
        with (out/'worker.log').open('w') as stream:
            p=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout,check=False)
        if (out/'measurement.json').exists():
            row=json.loads((out/'measurement.json').read_text())
        elif (out/'memory_limit_event.json').exists():
            row=dict(c,**json.loads((out/'memory_limit_event.json').read_text()))
        else:
            row=dict(c,status='worker_failed',error=f'exit {p.returncode}')
    except subprocess.TimeoutExpired:
        row=dict(c,status='timeout',error=f'worker exceeded {timeout}s deadline')
    row.update(wall_seconds=perf_counter()-start,timeout_seconds=timeout)
    row.setdefault('global_cp_error',None)
    row.setdefault('global_cp_bound',None)
    if row['status'] not in ('completed',) and (out/'events.jsonl').exists():
        row['partial_event_count']=sum(1 for _ in (out/'events.jsonl').open())
    save(out/'measurement.json',row)
    return row

def run(args):
    if args.timeout<=0 or args.memory_mib<=0 or args.budget<=0: raise ValueError('positive resource limits required')
    out=args.out or (Path(__file__).parent/'results'/f'{args.suite}_{uuid.uuid4().hex[:10]}')
    out=out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    sources=source_hashes()
    try:
        from threadpoolctl import threadpool_info
        blas=threadpool_info()
    except ImportError: blas=[]
    save(out/'environment.json',dict(platform=platform.platform(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
         processor=platform.processor(),blas=blas,blas_build=np.__config__.show(mode='dicts')['Build Dependencies']['blas'] if hasattr(np.__config__,'CONFIG') else None,threads_requested=1,base_sha=BASE,source_sha=source,source_files_sha256=sources,
         memory_limit_semantics='per-worker peak RSS watchdog sampled every 40 ms; OS scheduling permits overshoot',command=sys.argv))
    configs=grid(args.suite)
    save(out/'grid.json',configs)
    rows=[]
    start=perf_counter()
    for i,c in enumerate(configs):
        for r in range(1,c['repeats']+1):
            key=f"{i:03d}_{c['family']}{c['n']}_{c['method']}_s{c['steps']}_r{r}"
            folder=out/key
            folder.mkdir()
            cfg=dict(c,repeat=r,memory_mib=args.memory_mib,base_sha=BASE,source_sha=source,source_files_sha256=sources)
            remaining=args.budget-(perf_counter()-start)
            if remaining<=0:
                row=dict(cfg,status='not_run_budget',global_cp_error=None)
            else:
                row=execute(cfg,folder,min(args.timeout,remaining))
            row['directory']=key
            rows.append(row)
            print(key,row['status'],row.get('global_cp_error'),flush=True)
            csv_table(out/'measurements.csv',rows)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',default='smoke',choices=['comparison','smoke','bench','baselines','galerkin','physics','scaling','refine'])
    p.add_argument('--out',type=Path)
    p.add_argument('--timeout',type=float,default=40.)
    p.add_argument('--budget',type=float,default=1200.)
    p.add_argument('--memory-mib',type=float,default=1024.)
    p.add_argument('--worker')
    args=p.parse_args()
    if args.worker: worker(json.loads(args.worker),args.out)
    else: run(args)
