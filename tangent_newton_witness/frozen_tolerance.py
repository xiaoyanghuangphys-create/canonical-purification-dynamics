"""Replay the difficult three-spin loss checkpoint with fixed numerical policies."""
from collections import defaultdict
from pathlib import Path
import argparse
import itertools
from time import perf_counter
from types import SimpleNamespace

from galerkin_sylvester_tdvp import benchmarks as bm
from galerkin_sylvester_tdvp.models import Generator, witness
from galerkin_sylvester_tdvp.witness_pair_worker import _audit
from factor_tdvp.tangent_solver import matrix, matrix_to_mpo
from .frozen_channels import power, current_floor, trace_hash
from .paired import prediction
from .endpoint import run_endpoint
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    tol=1e-7
    n=3
    e=witness.WitnessAudit(n,.35,.8,noise='loss')
    prior=witness.channel(e.physical,3.75,'loss')
    state=witness.factor_to_root(matrix_to_mpo(power(matrix(prior),.25),n),6)
    initial,_=_audit(e,state[1],state[2],prior,3.75,tol)
    target=witness.channel(e.physical,4.,'loss')
    ref=power(matrix(target),.5)
    cfg=dict(chi=4,work=16,rtol=1e-8,atol=2e-13*2**(-n/2),maxiter=80,max_elements=2_000_000,predictor='B')
    model=SimpleNamespace(floor=0.,generator=Generator(n,'loss',np.ones(n),work=16))
    pred=prediction(cfg,state[1],3.75,4.,initial['bound'],model,e,current_floor,defaultdict(float),args.out)
    raw=[]
    grid=list(itertools.product([1e-8,1e-6,1e-4],['pauli','dense','dense-jacobi']))
    bm.save(args.out/'manifest.json',dict(family='correlated loss',n=n,previous=3.75,time=4.,
            tolerance=tol,polarization=.35,angle=.8,mode='two',grid=grid,
            source='frozen_channels exact previous-factor oracle',
            predictor_hash=trace_hash(pred),warm_hash=trace_hash(state[0],state[1]),
            target_hash=trace_hash(target),spectral_checks='small dense optional floor'))
    for i,(rtol,policy) in enumerate(grid):
        start=perf_counter()
        # Each trial independently sets up its target-time metric and fit.
        from copy import copy
        out,info=run_endpoint(copy(e),state,pred,target,4.,levels=[(4,6,10),(6,8,12),(8,12,16)],
                          tolerance=tol,fit='new',correction='new',rtol=rtol,
                          mode='two',passes=8,fit_iterations=3,
                          preconditioner='jacobi' if policy=='dense-jacobi' else 'none',
                          factor_floor=None if policy=='pauli' else current_floor)
        elapsed=perf_counter()-start
        actual=np.linalg.norm(matrix(out[1])-ref) if info['accepted'] else None
        if actual is not None and actual>info['bound']+2e-11:
            raise ArithmeticError("Score exceeds bound")
        bm.save(args.out/f'endpoint_{i:02d}.json',info)
        raw.append(dict(rtol=rtol,policy=policy,status=info['status'],bound=info['bound'],
                        actual=actual,updates=info['correction_updates'],seconds=elapsed))
        bm.save(args.out/'measurements.json',raw)


if __name__=='__main__':
    main()
