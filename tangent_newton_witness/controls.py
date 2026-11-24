"""Raw multi-step predictors, bounded time bisection, and finite-channel Ising startup."""
import argparse
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import expm

from .adaptive import advance
from .paired import physical_model, prediction, save_state
from .stress import floor, prepare, spectral_power, ts, sw, write, auto
from .endpoint import run_endpoint
from galerkin_sylvester_tdvp.evolution import KrausStartup
from galerkin_sylvester_tdvp.witness_pair_worker import _audit


def run_raw(folder,family,n,method,tend,steps):
    model,engine,factor=physical_model(family,n,24)
    carry=sw.factor_to_root(factor,4 if family!='E' else 2)
    x=carry[1]
    rows=[]
    start=perf_counter()
    for a,b in zip(np.linspace(0,tend,steps+1)[:-1],np.linspace(0,tend,steps+1)[1:]):
        # Request a direct stored-stage small-N floor at the start; retain the
        # predictor's displacement audit for the second stage.
        c=dict(predictor=method,chi=4 if family!='E' else 2,work=24,
               rtol=1.e-8,atol=1.e-14,maxiter=80)
        tic=perf_counter()
        try:
            x=prediction(c,x,a,b,model.floor+1,model,engine,floor,defaultdict(float),folder)
        except Exception as exc:
            rows.append(dict(start=a,end=b,status='failed',message=str(exc)))
            break
        elapsed=perf_counter()-tic
        root,target=model.reference(b)
        eta=sw.tn.distance(x,sw.mpo.multiply(factor,sw.mpo.adjoint(factor)))
        check,_=_audit(engine,x,eta,target,b,1.e-5)
        rows.append(dict(time=b,status='raw_endpoint',predictor_seconds=elapsed,
                         error=sw.tn.distance(root,x),carry_witness_bound=check['bound']))
    write(folder/'raw.json',dict(family=family,n=n,method=method,total=tend,steps=steps,rows=rows,
                                 inclusive_seconds=perf_counter()-start))


def adaptive_product(folder,method):
    e,state,target_at=prepare(dict(family='product',n=2))
    def propose(previous,left,right):
        return run_endpoint(e,previous,None,target_at(right),right,levels=[(4,6,12),(8,12,20)],
                            passes=8,fit='none',correction=method,tolerance=1.e-5,factor_floor=floor)
    state,log=advance(e,state,0.,.5,propose,max_halvings=4)
    if log['last_accepted_time']>0:
        target=ts.matrix(target_at(log['last_accepted_time']))
        log['actual_error']=np.linalg.norm(ts.matrix(state[1])-spectral_power(target,.5))
        save_state(folder/'last_accepted.npz',state,log['last_accepted_time'],
                   next(a['info']['bound'] for a in reversed(log['attempts']) if a['info']['accepted']))
    write(folder/'attempts.json',log)


def pure_control(folder,method):
    e,state,target_at=prepare(dict(family='ising-pure',n=2))
    # This split positive finite channel spans the full first integration interval.
    # Only the caller's independent certificate can admit the result.
    grid=np.linspace(0,.01,4)
    audit=auto.classify(ts.matrix(state[1]),e.model,grid[1],1.e-5)
    tic=perf_counter()
    seeded,seed_info=KrausStartup(e.model.h,e.model.jumps,substeps=4)(state[1],grid[1])
    raw_factor=ts.matrix_to_mpo(spectral_power(ts.matrix(seeded),.5),2)
    seed=sw.factor_to_root(raw_factor,8)
    seed_seconds=perf_counter()-tic
    log=[]
    for j,t in enumerate(grid[1:]):
        target=target_at(t)
        start=perf_counter()
        trial,info=run_endpoint(e,seed if j==0 else state,None,target,t,
                                levels=[(4,6,12),(8,12,20)],passes=8,
                                fit='none',correction=method,tolerance=1.e-5,
                                factor_floor=floor)
        actual=None
        if info['accepted']:
            dense=(expm(t*e.liouv)@e.rho0.reshape(-1,order='F')).reshape(e.rho0.shape,order='F')
            actual=np.linalg.norm(ts.matrix(trial[1])-spectral_power(dense,.5))
            if actual>info['bound']+5.e-10:
                raise AssertionError('Gap-free certificate missed reference')
            state=trial
            save_state(folder/'last_accepted.npz',state,t,info['bound'])
        log.append(dict(time=t,info=info,error=actual,seconds=perf_counter()-start))
        if not info['accepted']:
            break
    write(folder/'pure.json',dict(initial_audit=audit,seed_backend=seed_info,seed_seconds=seed_seconds,
         reached=max((a['time'] for a in log if a['info']['accepted']),default=0.),total=.01,checkpoints=log))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    for method in ('old','new'):
        folder=args.out/('adaptive_'+method)
        folder.mkdir()
        adaptive_product(folder,method)
    for method in ('new','gn'):
        folder=args.out/('pure_'+method)
        folder.mkdir()
        pure_control(folder,method)
    for family,total,n,steps in [('D',.1,8,2),('E',1.,8,2),('Dstress',4.,3,8)]:
        for method in ('A','B'):
            folder=args.out/(family+'_'+method)
            folder.mkdir()
            run_raw(folder,family,n,method,total,steps)
