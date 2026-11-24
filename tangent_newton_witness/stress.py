"""Small independent stress families. Run from the repository root with -m."""
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import traceback

import numpy as np
import scipy
from scipy.linalg import eigh, expm

from factor_tdvp import tangent_solver as ts
from factor_tdvp.baseline import _legacy_module, _liouvillian
from galerkin_sylvester_tdvp import witness_pair_worker as pair
from galerkin_sylvester_tdvp.models import Generator, kron_state
from galerkin_sylvester_tdvp.galerkin_solver import _evidence
from .endpoint import run_endpoint
from .support import auto, finite_start
from .paired import prediction as paired_prediction, save_state
import scalable_cp_witness as sw

tn, mpo = ts.tn, ts.mpo
TINY = np.finfo(float).tiny


def write(path, data):
    path.write_text(json.dumps(_evidence(data), allow_nan=False, indent=1))


def product_power(n, p, a):
    vals=np.power([(1+p)/2,(1-p)/2], a)
    local=(sum(vals)*mpo.I + (vals[0]-vals[1])*mpo.X)/2
    return kron_state([local]*n)


def floor(w):
    a=ts.matrix(w)
    h=(a+a.conj().T)/2
    eig=np.linalg.eigvalsh(h)
    return float(eig[0]-32*np.finfo(float).eps*len(a)*max(abs(eig)))


def spectral_power(a,p):
    vals,u=eigh((a+a.conj().T)/2)
    if vals.min() < -32*np.finfo(float).eps*len(a)*max(abs(vals)):
        raise ArithmeticError("Indefinite density in independent scoring")
    return (u * np.maximum(vals,0)**p) @ u.conj().T


class InteractingAudit:
    """Scalar no-jump inverse bound or trace-one uniform gap-free weight."""
    def __init__(self,n,kind):
        self.n=n
        h,jumps,unused,pure=_legacy_module().spin_model(n,'loss')
        self.model=auto.Model(h,jumps)
        self.liouv=_liouvillian(h,jumps)
        d=2**n
        mix=.001 if kind=='near' else 0.
        self.rho0=(1-mix)*pure + mix*np.eye(d)/d
        self.floor=mix/d
        self.decay=np.linalg.norm(sum(j.conj().T@j for j in jumps),2)
        self.trace_error=None

    def target(self,time):
        # Independent order-40 Taylor propagation with an analytic truncation bound
        # in Frobenius norm and then trace norm. Small chains only.
        a=self.liouv*time
        z=np.linalg.norm(a,2)
        term=self.rho0.reshape(-1,order='F')
        total=term.copy()
        for j in range(1,41):
            term=a@term/j
            total+=term
        self.trace_error=(np.sqrt(len(self.rho0))*np.linalg.norm(self.rho0)*
                          np.exp(z)*z**41/math.factorial(41))
        value=total.reshape(self.rho0.shape,order='F')
        hermitian=(value+value.conj().T)/2
        density=ts.matrix_to_mpo(hermitian,self.n)
        # A separate allowance covers both Hermitianization and TT-SVD conversion.
        self.trace_error+=np.sqrt(len(self.rho0))*np.linalg.norm(ts.matrix(density)-value)
        return density

    def endpoint(self,root,eta,target,time,tol):
        # pair._audit has paid for symmetrizing the stored root.
        defect=tn.add(target,mpo.multiply(root,root),beta=-1.)
        delta=tn.norm(defect)
        weight=np.sqrt(2**self.n) if self.floor==0 else np.exp(time*self.decay/2)/np.sqrt(self.floor)
        weighted=weight*delta
        term=np.sqrt(weighted+self.trace_error) if self.floor==0 else weighted+weight*self.trace_error
        return dict(bound=2*eta+term,witness=eta,weighted_residual=weighted,
                    trace_error=self.trace_error,metric='uniform gap-free' if self.floor==0 else 'no-jump scalar'),defect


def prepare(c):
    n,kind=c['n'],c['family']
    if kind.startswith('ising'):
        audit=InteractingAudit(n,'pure' if kind=='ising-pure' else 'near')
        w=ts.matrix_to_mpo(spectral_power(audit.rho0,.25),n)
        return audit,sw.factor_to_root(w,8),audit.target
    noise='loss' if kind=='loss' else 'dephasing'
    p,angle=(.86,0.) if kind=='product' else (.35,.8)
    audit=sw.WitnessAudit(n,p,angle,noise=noise)
    power=sw.prepared_power
    if kind=='product':
        for k,a in dict(physical=1.,inverse=-.5,inverse_density=-1.,preconditioner=-.375,
                        factor_preconditioner=-.125,root_preconditioner=-.25).items():
            setattr(audit,k,product_power(n,p,a))
        w=product_power(n,p,.25)
    else:
        w=power(n,.25,p,angle)
    state=sw.factor_to_root(sw.hermitian_round(w,8),8)
    return audit,state,lambda t:sw.channel(audit.physical,t,noise)


def grid(suite):
    def cfg(f,n,steps,total,label,**kw):
        return dict(family=f,n=n,steps=steps,total=total,label=label,
                    tolerance=1.e-5,mode='two',**kw)
    if suite=='short':
        return [cfg(f,2,3,t,m) for f,t,methods in [
            ('product',1.5,['Old-only','New-only']),('strong-D',.3,['Old-only','New-only']),
            ('loss',.6,['Old-only','Adaptive-only','New-only']),
            ('ising-near',.03,['New-only','GN-only']),('ising-pure',.01,['New-only','GN-only'])]
                for m in methods]
    if suite=='late':
        configs=[cfg('product',2,16,8.,m) for m in ('A-old','A-new','B-old','B-new','Old-only','New-only')]
        configs += [cfg('product',n,k,4.,m) for n,k in [(1,8),(2,4),(2,16),(4,8)] for m in ('Old-only','New-only')]
        configs += [cfg(f,2,8,t,m) for f,t,methods in [
            ('strong-D',4.,['Old-only','New-only']),('loss',4.,['Old-only','Adaptive-only','New-only'])]
                    for m in methods]
        return configs
    if suite=='frozen':
        return [cfg('product',n,1,4.,m,frozen_previous=3.5) for n in (2,4)
                for m in ('Old-only','Recentered-only','New-only','GN-only')]
    if suite=='confirm':
        return [cfg('product',2,8,4.,m,repeat=i) for i in range(1,5)
                for m in (('New-only','Old-only') if i%2 else ('Old-only','New-only'))]
    raise ValueError(suite)


def worker(c,folder):
    n,label=c['n'],c['label']
    start=perf_counter()
    e,state,target_at=prepare(c)
    ising=isinstance(e,InteractingAudit)
    initial=ts.matrix(state[1])@ts.matrix(state[1])
    anchors=(getattr(e,'factor_preconditioner',None),getattr(e,'preconditioner',None))
    variant='gn' if label=='GN-only' else ('new' if 'new' in label.lower() else 'old')
    model=SimpleNamespace(generator=Generator(n,'dephasing',np.full(n,.5),work=24),
                          floor=np.sqrt(np.min(np.linalg.eigvalsh(initial))))
    t0=c.get('frozen_previous',0.)
    if t0:
        state=sw.factor_to_root(product_power(n,.86*np.exp(-t0),.25),8)
        if label=='Recentered-only':
            anchors=tuple(product_power(n,.86*np.exp(-t0),p) for p in (-.125,-.375))
    times=np.linspace(t0,c['total'],c['steps']+1)
    row=dict(c,status='running',last_accepted_time=t0,checkpoints=[],setup_seconds=perf_counter()-start,
             predictor_seconds=0.,physical_seconds=0.,scoring_seconds=0.,io_seconds=0.,
             support_seconds=0.,witness_seconds=0.,global_cp_error=None,global_cp_bound=None)
    bound=0.
    def record():
        write(folder/'row.json',row)
    record()
    try:
        for ti in times[1:]:
            begin=perf_counter()
            carried=state
            seed_info=None
            if ising and t0==0.:
                state,h,seed_info=finite_start(carried,e.model,float(ti-t0),1.e-5,
                                                root_bond=8,factor_bond=8)
                if h and abs(ti-t0-h)>1.e-14:
                    # Benchmark uses the prescribed checkpoints. A shorter
                    # bridge is still a legitimate warm start for the correction.
                    seed_info['seed_time']=t0+h
            row['support_seconds']+=perf_counter()-begin
            begin=perf_counter()
            target=target_at(ti)
            row['physical_seconds']+=perf_counter()-begin
            begin=perf_counter()
            predictor=None
            if label[0] in ('A','B') and '-' in label and label.split('-')[0] in ('A','B'):
                settings=dict(predictor=label[0],chi=4,work=24,rtol=1.e-8,atol=1.e-14,maxiter=80)
                predictor=paired_prediction(settings,state[1],t0,ti,bound,model,e,floor,defaultdict(float),folder)
            row['predictor_seconds']+=perf_counter()-begin
            begin=perf_counter()
            proposed,info=run_endpoint(e,state,predictor,target,ti,levels=[(4,6,12),(8,12,20)],
                      tolerance=c['tolerance'],fit=variant if predictor else 'none',correction=variant,
                      mode=c['mode'],passes=8,rtol=1.e-8,factor_floor=floor,
                      legacy_anchors=anchors if label in ('Old-only','Recentered-only','A-old','B-old') else None)
            row['witness_seconds']+=perf_counter()-begin
            begin=perf_counter()
            dense=ts.matrix(target)
            if ising:
                dense=(expm(float(ti)*e.liouv)@e.rho0.reshape(-1,order='F')).reshape(e.rho0.shape,order='F')
            ref=spectral_power(dense,.5)
            spectrum=eigh(dense,initial,eigvals_only=True) if not ising or e.floor>0 else None
            wspec=np.linalg.eigvalsh(ts.matrix(state[0]))
            checkpoint=dict(time=ti,bound=info.get('bound'),accepted=info['accepted'],
              target_over_anchor=None if spectrum is None else [spectrum[0],spectrum[-1]],
              start_factor_eigenrange=[wspec[0],wspec[-1]],
              full_fourth_condition=(wspec[-1]/wspec[0])**3 if wspec[0]>0 else None,
              warm_error=np.linalg.norm(ts.matrix(state[1])-ref),
              predictor_error=None if predictor is None else np.linalg.norm(ts.matrix(predictor)-ref),
              actual_error=np.linalg.norm(ts.matrix(proposed[1])-ref) if info['accepted'] else None,
              support=seed_info)
            row['scoring_seconds']+=perf_counter()-begin
            begin=perf_counter()
            write(folder/f'endpoint_{len(row["checkpoints"])+1:02d}.json',info)
            row['checkpoints'].append(checkpoint)
            if not info['accepted']:
                row['status']='uncertified'
                break
            if checkpoint['actual_error']>checkpoint['bound']+5.e-10:
                raise ArithmeticError('Independent score outside certificate allowance')
            state,bound,t0=proposed,info['bound'],ti
            row.update(last_accepted_time=t0)
            save_state(folder/'last_accepted.npz',state,t0,bound)
            row['io_seconds']+=perf_counter()-begin
            record()
        else:
            row.update(status='completed',global_cp_error=checkpoint['actual_error'],global_cp_bound=bound)
    except Exception as exc:
        row.update(status='failed',error=str(exc))
        (folder/'exception.txt').write_text(traceback.format_exc())
    row['total_seconds']=perf_counter()-start
    row['production_seconds']=row['total_seconds']-row['scoring_seconds']
    row['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
    record()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--suite',choices=['short','late','frozen','confirm'],default='short')
    p.add_argument('--worker')
    p.add_argument('--timeout',type=float,default=45.)
    args=p.parse_args()
    if args.worker:
        worker(json.loads(args.worker),args.out)
        return
    args.out.mkdir(parents=True,exist_ok=False)
    configs=grid(args.suite)
    here=Path(__file__).parent
    env=dict(os.environ,**{k:'1' for k in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
                                            'MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    write(args.out/'manifest.json',dict(configs=configs,timeout=args.timeout,base_sha=subprocess.check_output(
        ['git','rev-parse','origin/research/galerkin-sylvester-tdvp'],text=True).strip(),
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in here.glob('*.py')},
        python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
        machine=platform.machine(),threads=1,seeds='deterministic inputs'))
    for i,c in enumerate(configs):
        folder=args.out/f'{i:03d}_{c["family"]}{c["n"]}_{c["label"]}'
        folder.mkdir()
        with (folder/'worker.log').open('w') as stream:
            try:
                child=subprocess.run([sys.executable,'-m','tangent_newton_witness.stress','--worker',
                                      json.dumps(c),'--out',str(folder)],env=env,timeout=args.timeout,
                                     stdout=stream,stderr=subprocess.STDOUT)
                failure='worker_exit_'+str(child.returncode) if child.returncode else None
            except subprocess.TimeoutExpired:
                failure='timeout'
        if failure:
            path=folder/'row.json'
            row=json.loads(path.read_text()) if path.exists() else dict(c,last_accepted_time=c.get('frozen_previous',0.))
            row.update(status=failure,global_cp_error=None,global_cp_bound=None)
            write(path,row)


if __name__=='__main__':
    main()
