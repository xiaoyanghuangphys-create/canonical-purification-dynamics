"""Production A/B TDVP and endpoint adapters. One solver per isolated worker.

Only preparation powers are built here. The witness target and Jensen metric
are obtained by application of the physical channel to initial MPOs. The raw
solvers have neither an endpoint density nor a time-dependent root provider.
"""
from collections import defaultdict
from contextlib import contextmanager, ExitStack
from functools import partial
from math import hypot, sqrt
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

from galerkin_sylvester_tdvp import witness_pair_worker as pair
from galerkin_sylvester_tdvp import witness_precond_solver as pcg
from galerkin_sylvester_tdvp.evolution import EvolutionError
from galerkin_sylvester_tdvp.models import Generator, mpo, parity_floor_setup, tn, witness
import numpy as np

METHODS = ('A-raw','B-raw','A+witness','B+witness','Witness-only','A+Newton','B+Newton')


def _audit_engine(n, p, b, residual_bond):
    e = witness.WitnessAudit.__new__(witness.WitnessAudit)
    e.n, e.noise, e.jensen, e.local_correction = n, 'dephasing', True, False
    e.metric_time, e.residual_bond = None, residual_bond
    for name, power in [('physical', 1.), ('inverse', -.5), ('inverse_density', -1.),
                        ('preconditioner', -.375), ('root_preconditioner', -.25),
                        ('factor_preconditioner', -.125)]:
        setattr(e, name, witness.parity_power(n,power,p,b))
    e.inverse_norm = 2.**(n/2)/sqrt(1-hypot(p,b))
    return e


@contextmanager
def _observe_operations(record, limit):
    """Guard and count concrete algebra cores, irrespective of nominal work caps."""
    product, add, compress = mpo.multiply, tn.add, tn.compress

    def shapes(a, b, op):
        for j,(x,y) in enumerate(zip(a.tensors,b.tensors)):
            left = op(x.shape[0], y.shape[0]) if j else 1
            right = op(x.shape[-1], y.shape[-1]) if j+1<a.n else 1
            entries = int(4*left*right)
            record['peak_core_elements'] = max(record['peak_core_elements'], entries)
            record['peak_intermediate_bond'] = max(record['peak_intermediate_bond'], left, right)
            if entries > limit:
                raise EvolutionError('memory_budget',requested_core_elements=entries,max_elements=limit)
    def product_checked(a,b,*args,**kwargs):
        shapes(a,b,int.__mul__)
        return product(a,b,*args,**kwargs)
    def add_checked(a,b,*args,**kwargs):
        shapes(a,b,int.__add__)
        return add(a,b,*args,**kwargs)
    def observed_compress(a,*args,**kwargs):
        for core in a.tensors:
            record['peak_intermediate_bond']=max(record['peak_intermediate_bond'],*core.shape[::2])
            record['peak_core_elements']=max(record['peak_core_elements'],int(core.size))
            if core.size > limit:
                raise EvolutionError('memory_budget',requested_core_elements=int(core.size),max_elements=limit)
        ans,info = compress(a,*args,**kwargs)
        record['compressions'] += 1
        record['max_compression_discard'] = max(record['max_compression_discard'],info['discarded_bound'])
        return ans,info
    with patch.object(mpo,'multiply', product_checked), patch.object(tn,'add',add_checked), \
         patch.object(tn,'compress',observed_compress):
        yield


def _a_step(state,t0,t1,engine,cfg,events,clocks):
    """Invoke native two-pass force; audit its final unprojected equation."""
    midpoint = tn.midpoint_step
    congruence = witness.congruence
    def instrumented(initial,t,dt,provider,chi,retract,mode):
        def audited_force(u,x):
            sources=[]
            def capture(anchor,source,work,coefficient):
                sources.append(source)
                return congruence(anchor,source,work,coefficient)
            with pair.timed(clocks,'ambient_force_seconds'), patch.object(witness,'congruence',capture):
                force,info=provider(u,x)
            with pair.timed(clocks,'force_audit_seconds'):
                rhs,intermediate=sources
                applied=tn.add(mpo.multiply(x,force),mpo.multiply(force,x))
                events.append(dict(time=u,stage='start' if not events else 'midpoint',fixed_passes=2,
                        rhs_norm=tn.norm(rhs),first_residual_abs=tn.norm(intermediate),
                        final_ambient_residual_abs=tn.distance(rhs,applied)))
            return force,info
        def timed_retract(x):
            with pair.timed(clocks,'retraction_seconds'):
                return retract(x)
        return midpoint(initial,t,dt,audited_force,chi,retract=timed_retract,mode=mode)
    with patch.object(tn,'midpoint_step',instrumented):
        return witness.tdvp_prediction(state,t0,t1,engine,cfg['chi'],cfg['work'])


@contextmanager
def _observe_fit(info):
    native_fit, native_round = witness.fit_witness,witness.hermitian_round
    def counted_fit(*args,**kwargs):
        trials=[]
        def observe(*a,**kw):
            trial=native_round(*a,**kw)
            trials.append(trial)
            return trial
        with patch.object(witness,'hermitian_round',observe):
            factor,eta=native_fit(*args,**kwargs)
        info['fit_trials']=len(trials)
        info['fit_updates']=next((i for i,obj in enumerate(trials,1) if obj is factor),0)
        return factor,eta
    with patch.object(witness,'fit_witness',counted_fit):
        yield


def _correction_only(engine,previous,t,cfg,clocks,out):
    """Native carry/correct loop with precisely zero predictor and fitting calls."""
    factor,root,eta=previous
    with pair.timed(clocks,'density_reference_seconds'):
        target=witness.channel(engine.physical,t)
    with pair.timed(clocks,'certification_seconds'):
        check,residual=pair._audit(engine,root,eta,target,t,cfg['cert_tol'])
    carry=check['bound']
    updates=attempts=0
    for fb,rb,wb in cfg['levels']:
        if check['bound']<=cfg['cert_tol']:
            break
        for k in range(cfg['passes']):
            with pair.timed(clocks,'correction_seconds'):
                direction=engine.correction(residual,wb)
            for scale in (1.,.5,.25):
                with pair.timed(clocks,'correction_seconds'):
                    trial=witness.hermitian_round(tn.add(factor,direction,beta=scale),fb)
                    proposal=witness.factor_to_root(trial,rb)
                with pair.timed(clocks,'certification_seconds'):
                    checked,r=pair._audit(engine,proposal[1],proposal[2],target,t,cfg['cert_tol'])
                better=checked['bound']<check['bound']
                attempts+=1
                pair.bm.append_event(out/'corrections.jsonl',dict(time=t,iteration=k+1,
                            factor_cap=fb,root_cap=rb,work_cap=wb,step_length=scale,improved=better,**checked))
                if better:
                    factor,root,eta=proposal
                    check,residual=checked,r
                    updates+=1
                    break
            else:
                break
            if check['bound']<=cfg['cert_tol']:
                break
    return (factor,root,eta),dict(check,prior_root_bound=carry,predictor_bound=None,
                              accepted=check['bound']<=cfg['cert_tol'],predictor_selected=False,
                              fit_trials=0,fit_updates=0,correction_updates=updates,correction_trials=attempts)


class GlobalEvolution:
    def __init__(self,config,out):
        self.out=Path(out)
        self.out.mkdir(parents=True,exist_ok=True)
        self.method=config['method']
        n=config['n']
        p,b=config.get('p',.6),config.get('b0',.4)
        if self.method not in METHODS or n<2 or not hypot(p,b)<1:
            raise ValueError('Require a supported method, N>=2, and a full-rank preparation')
        if config.get('protocol','generator') not in ('generator','uniform'):
            raise ValueError('Global predictors apply the uniform onsite generator')
        self.certified=self.method not in ('A-raw','B-raw')
        self.cfg=dict(chi=2,work=16 if self.method.startswith('B') else 8,
                      levels=[[4,6,8],[6,8,12]],passes=8,rtol=1e-9,atol=2e-13*2.**(-n/2),
                      cert_tol=1e-6,maxiter=80,max_elements=2_000_000,residual_bond=24)
        self.cfg.update(config)
        c=self.cfg
        if (min(c['chi'],c['work'],c['max_elements'],c['passes'],c['residual_bond'])<1 or
            not c['cert_tol']>0 or not c['levels'] or
            any(len(l)!=3 or min(l)<1 for l in c['levels'])):
            raise ValueError('Invalid global numerical budgets')
        self.t=0.
        self.timings=defaultdict(float)
        self.setup_operations=defaultdict(float)
        self.initial_bound=None
        self.floor=None
        self.newton=None
        self.factor=None
        self.bound=None
        self.model=None
        self.initial_info={}
        with pair.timed(self.timings,'setup_seconds'), _observe_operations(self.setup_operations,c['max_elements']):
            if self.certified:
                self.engine=_audit_engine(n,p,b,c['residual_bond'])
                self.factor,self.root,self.eta=witness.factor_to_root(witness.parity_power(n,.25,p,b),c['chi'])
                info,_=pair._audit(self.engine,self.root,self.eta,self.engine.physical,0.,c['cert_tol'])
                self.bound=self.initial_bound=info['bound']
                self.initial_info=info
                if self.bound>c['cert_tol']:
                    raise EvolutionError('initial_witness_uncertified',**info)
            else:
                self.engine=SimpleNamespace(noise='dephasing',root_preconditioner=witness.parity_power(n,-.25,p,b))
                self.root=witness.hermitian_round(witness.parity_power(n,.5,p,b),c['chi'],normalize=True)
            if self.method.startswith('B'):
                self.floor=parity_floor_setup(n)
                self.model=SimpleNamespace(generator=Generator(n,'dephasing',np.full(n,.5),work=c['work']),
                                    floor=2.**(-n/2)*sqrt(1-hypot(p,b)))
            if self.method.endswith('Newton'):
                from .global_newton import load_endpoint
                self.newton=load_endpoint()
        pair.bm.append_event(self.out/'global_setup.jsonl',dict(n=n,p=p,b0=b,method=self.method,
                initial_audit=self.initial_info,operations=dict(self.setup_operations),budgets=c,
                source='initial preparation powers only; uniform onsite Z dephasing',
                witness_physical_data='exact channel on initial density and inverse density' if self.certified else None))

    def step(self,t0,t1):
        if t0!=self.t or not t1>t0:
            raise ValueError('Step must start at the last accepted time and advance it')
        c=self.cfg
        ops=defaultdict(float)
        details={}
        pred=None
        events=[]
        before=dict(vars(self.engine))
        try:
            with _observe_operations(ops,c['max_elements']):
                if self.method.startswith('A'):
                    with pair.timed(self.timings,'predictor_seconds'):
                        pred=_a_step(self.root,t0,t1,self.engine,c,events,self.timings)
                    for e in events:
                        pair.bm.append_event(self.out/'ambient_solves.jsonl',e)
                    details['ambient_stages']=events
                elif self.method.startswith('B'):
                    solve=partial(pcg.solve_velocity,anchor=self.engine.root_preconditioner,preconditioner='preparation')
                    with pair.timed(self.timings,'predictor_seconds'),patch.object(pair,'solve_velocity',solve):
                        pred=pair._b_prediction(self.root,t0,t1,self.bound if self.bound is not None else np.inf,
                                                self.model,self.floor,c,self.timings,self.out)
                if self.certified:
                    previous=(self.factor,self.root,self.eta)
                    with pair.timed(self.timings,'witness_seconds'):
                        if self.newton:
                            with pair.timed(self.timings,'density_reference_seconds'):
                                target=witness.channel(self.engine.physical,t1)
                            proposal,check=self.newton(self.engine,previous,pred,target,t1,levels=c['levels'],
                                       tolerance=c['cert_tol'],passes=c['passes'],maxiter=c['maxiter'],
                                       max_elements=c['max_elements'],rtol=c.get('newton_rtol',1.e-8),
                                       on_event=lambda e:pair.bm.append_event(self.out/'newton.jsonl',dict(e,time=t1)))
                            for k,v in check['timings'].items():
                                self.timings[k]+=v
                        elif pred is None:
                            proposal,check=_correction_only(self.engine,previous,t1,c,self.timings,self.out)
                        else:
                            with _observe_fit(details):
                                proposal,check=pair._witness_endpoint(self.engine,previous,pred,t1,c,self.timings,self.out)
                    details.update(check)
                    details['positivity_contribution']=2*check['witness'] if 'witness' in check else None
                    if not check['accepted']:
                        raise EvolutionError('witness_uncertified',**details)
                    self.factor,self.root,self.eta=proposal
                    self.bound=check['bound']
                else:
                    self.root=pred
            self.t=t1
        except Exception as exc:
            vars(self.engine).clear()
            vars(self.engine).update(before)
            pair.bm.append_event(self.out/'global_failures.jsonl',dict(start=t0,end=t1,method=self.method,
                last_accepted=self.t,status=getattr(exc,'status','exception'),reason=str(exc),
                details=getattr(exc,'details',{}),operations=dict(ops)))
            raise
        details.update(operations=dict(ops),root_bond=self.root.bond_dimension(),
                factor_bond=self.factor.bond_dimension() if self.factor is not None else None,
                generator_site_applications=2*self.root.n if pred is not None else 0,
                density_and_metric_site_applications=2*self.root.n if self.certified else 0)
        return self.root,details,pred if self.certified else None
