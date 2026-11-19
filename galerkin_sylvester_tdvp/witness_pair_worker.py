"""Isolated two-site A/B predictors with the same positive-witness endpoint.

A uses the existing two-pass approximate ambient MPO force. B applies the
projected Sylvester solve to the actual current and midpoint numerical roots.
The witness layer compares the raw proposal with the old root, then uses the
native factor correction direction, backtracking, and bond escalation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from time import perf_counter
import traceback

from . import benchmarks as bm  # set single-thread environment before NumPy import
from .evolution import EvolutionError, _retract
from .galerkin_solver import solve_velocity
from .models import make_model, mpo, tn, witness
import numpy as np


@contextmanager
def timed(row, key):
    start = perf_counter()
    try:
        yield
    finally:
        row[key] += perf_counter() - start


def _audit(engine, state, eta, target, t, tol):
    """The historical bound assumes Hermiticity; charge any symmetry repair."""
    fixed = all(np.array_equal(core, core[:, [0, 2, 1, 3], :].conj()) for core in state.tensors)
    h = state if fixed else tn.add(state, mpo.adjoint(state), .5, .5)
    skew = 0. if fixed else tn.distance(h, state)
    info, residual = engine.endpoint(h, eta + skew, target, t, tol)
    info = dict(info, skew_allowance=skew, bound=info['bound'] + skew)
    if not np.isfinite(info['bound']) or info['bound'] < 0:
        raise EvolutionError('invalid_certificate', **info)
    return info, residual


def _witness_endpoint(engine, previous, prediction, t, cfg, row, out):
    factor, root, eta = previous
    levels, tol = cfg['levels'], cfg['cert_tol']
    with timed(row, 'density_reference_seconds'):
        target = witness.channel(engine.physical, t)
    with timed(row, 'certification_seconds'):
        check, residual = _audit(engine, root, eta, target, t, tol)
    carry_bound = check['bound']
    with timed(row, 'fit_seconds'):
        # Match the native warm fit, including an accepted larger factor.
        fb, _, wb = levels[0]
        fb = max(fb, factor.bond_dimension())
        wb = max(wb, fb, prediction.bond_dimension())
        fitted, fitted_eta = witness.fit_witness(factor, prediction, engine.factor_preconditioner, fb, wb)
    with timed(row, 'certification_seconds'):
        predicted_check, predicted_residual = _audit(engine, prediction, fitted_eta, target, t, tol)
    use_prediction = bool(predicted_check['bound'] < check['bound'])
    if use_prediction:
        root, factor, eta = prediction, fitted, fitted_eta
        check, residual = predicted_check, predicted_residual
    improved = 0
    attempts = 0
    for fb, rb, wb in levels:
        if check['bound'] <= tol:
            break
        for it in range(cfg['passes']):
            with timed(row, 'correction_seconds'):
                direction = engine.correction(residual, wb)
            for scale in (1., .5, .25):
                with timed(row, 'correction_seconds'):
                    trial_f = witness.hermitian_round(tn.add(factor, direction, beta=scale), fb)
                    trial_f, trial_x, trial_eta = witness.factor_to_root(trial_f, rb)
                with timed(row, 'certification_seconds'):
                    trial, r = _audit(engine, trial_x, trial_eta, target, t, tol)
                better = bool(trial['bound'] < check['bound'])
                attempts += 1
                bm.append_event(out/'corrections.jsonl',dict(time=t,factor_cap=fb,root_cap=rb,work_cap=wb,
                                iteration=it+1,step_length=scale,improved=better,**trial))
                if better:
                    factor, root, eta = trial_f, trial_x, trial_eta
                    check, residual = trial, r
                    improved += 1
                    break
            else:
                break
            if check['bound'] <= tol:
                break
    info = dict(time=t, prior_root_bound=carry_bound, predictor_bound=predicted_check['bound'],
                predictor_selected=use_prediction, accepted=check['bound'] <= tol,
                corrected_distance=tn.distance(prediction,root), correction_updates=improved,
                correction_trials=attempts, root_bond=root.bond_dimension(),
                factor_bond=factor.bond_dimension(), **check)
    return (factor, root, eta), info


def _b_prediction(root, t0, t1, last_bound, model, stage_floor, cfg, row, out):
    """Finite, retracted explicit midpoint. Only the two solve inputs need HPD."""
    width = cfg['chi']
    def stage(x, known, label):
        with timed(row, 'stage_audit_seconds'):
            lower, source = known, 'weyl'
            if not lower > 0:
                lower, source = stage_floor(x), 'structural'
            bm.append_event(out/'stage_bounds.jsonl',dict(start=t0,stage=label,lower_bound=lower,source=source))
            if not np.isfinite(lower) or lower <= 0:
                raise EvolutionError('rank_support_uncertain', time=t0, stage=label, lower_bound=lower)
        with timed(row, 'galerkin_force_seconds'):
            velocity, diag = solve_velocity(x, model.generator, mode='two', lower_bound=lower,
                        constrained=True, rtol=cfg['rtol'], atol=cfg['atol'],
                        maxiter=cfg['maxiter'], work_bond=cfg['work'],
                        max_elements=cfg.get('max_elements',2_000_000))
        bm.append_event(out/'solves.jsonl',dict(start=t0,stage=label,**diag))
        row['cg_iterations'] += diag['iterations']
        return velocity, lower

    # Unital physical channel preserves the initial physical spectral floor;
    # the witness certificate bounds the current stored-root displacement.
    v, start_floor = stage(root, model.floor - last_bound, 'start')
    dt = t1 - t0
    with timed(row, 'retraction_seconds'):
        mid, rm = _retract(tn.add(root,v,beta=dt/2),width,False,0)
    with timed(row, 'stage_audit_seconds'):
        provisional = start_floor - tn.distance(root,mid)
    v, _ = stage(mid, provisional, 'midpoint')
    with timed(row, 'retraction_seconds'):
        candidate, re = _retract(tn.add(root,v,beta=dt),width,False,0)
    bm.append_event(out/'retractions.jsonl',dict(start=t0,mid=rm,end=re))
    return candidate


def _checkpoint(out, state, t, bound):
    factor, root, _ = state
    with (out/'last_accepted.tmp').open('wb') as stream:
        np.savez_compressed(stream, time=t, bound=bound,
                            **{f'g{j}':a for j,a in enumerate(factor.tensors)},
                            **{f'x{j}':a for j,a in enumerate(root.tensors)})
    (out/'last_accepted.tmp').replace(out/'last_accepted.npz')


def worker(cfg, out):
    """Persist an atomic measurement and the last accepted root/factor checkpoint."""
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    fam,n,method = cfg['family'],cfg['n'],cfg['method']
    if fam not in ('D','E') or method not in ('A','B') or n<2:
        raise ValueError('Expected a D or E chain, at least two sites, and predictor A or B')
    defaults = dict(chi=4 if fam=='D' else 2,work=64 if fam=='D' else 16,
                    levels=[[4,6,10],[6,8,12],[8,12,16]] if fam=='D' else [[4,6,8],[6,8,12]],
                    passes=4 if fam=='D' else 8,rtol=1e-9,atol=2e-13*2.**(-n/2),maxiter=80)
    cfg=dict(defaults,**cfg)
    begin = perf_counter()
    stop = bm.start_memory_watchdog(out,cfg['memory_mib'],interval=.04)
    row = dict(cfg,status='running',accepted_steps=0,rejected_steps=0,reached_time=0.,
               global_cp_error=None,global_cp_bound=None,peak_rss_mib=None,
               error_kind='measured_global' if fam=='E' or n<=8 else 'conditional_global_bound',
               objective='approximate ambient Sylvester + Euclidean two-site projector'
                         if method=='A' else 'constrained two-site Galerkin Sylvester',
               cg_iterations=0,predictors_selected=0,correction_updates=0,correction_trials=0)
    clocks = ('setup_seconds','predictor_seconds','witness_seconds','checkpoint_seconds',
              'density_reference_seconds','certification_seconds','fit_seconds','correction_seconds',
              'galerkin_force_seconds','retraction_seconds','stage_audit_seconds','scoring_seconds')
    row.update({key:0. for key in clocks})
    def progress(phase):
        bm.save(out/'progress.json',dict(row,phase=phase,elapsed_seconds=perf_counter()-begin,
                                        peak_rss_mib=bm.memory_mib()))
    try:
        progress('setup')
        with timed(row,'setup_seconds'):
            model=make_model(fam,n,cfg['work'])
            engine=witness.WitnessAudit(n) if fam=='D' else witness.WitnessAudit.parity_chain(n)
            factor=witness.prepared_power(n,.25) if fam=='D' else witness.parity_power(n,.25)
            state=witness.factor_to_root(factor,cfg['chi'])
            row['initial_representation_error']=tn.distance(state[1],model.root)
            floor=None
            if method=='B':
                from .witness_stage import make_stage_floor
                floor=make_stage_floor(fam,n,model,record=lambda x:bm.append_event(out/'floor_implementation.jsonl',x))
            initial,_=_audit(engine,state[1],state[2],engine.physical,0.,cfg['cert_tol'])
            last_bound=initial['bound']
            if last_bound>cfg['cert_tol']:
                raise EvolutionError('initial_witness_uncertified', **initial)
        with timed(row,'checkpoint_seconds'):
            _checkpoint(out,state,0.,last_bound)
        times=np.linspace(0.,cfg['total'],cfg['steps']+1)
        if len(times)<2 or not cfg['total']>0:
            raise ValueError('A positive horizon and number of steps are required')
        for t0,t1 in zip(times[:-1],times[1:]):
            progress('predictor')
            with timed(row,'predictor_seconds'):
                if method=='A':
                    pred=witness.tdvp_prediction(state[1],t0,t1,engine,cfg['chi'],cfg['work'])
                else:
                    pred=_b_prediction(state[1],t0,t1,last_bound,model,floor,cfg,row,out)
            progress('witness')
            with timed(row,'witness_seconds'):
                proposal,info=_witness_endpoint(engine,state,pred,t1,cfg,row,out)
            bm.append_event(out/'events.jsonl',info)
            if not info['accepted']:
                raise EvolutionError('witness_uncertified', **info)
            state=proposal
            last_bound=info['bound']
            row.update(reached_time=t1,root_bond=info['root_bond'],factor_bond=info['factor_bond'],
                       last_accepted_bound=last_bound,accepted_steps=row['accepted_steps']+1,
                       predictors_selected=row['predictors_selected']+int(info['predictor_selected']),
                       correction_updates=row['correction_updates']+info['correction_updates'],
                       correction_trials=row['correction_trials']+info['correction_trials'])
            with timed(row,'checkpoint_seconds'):
                _checkpoint(out,state,t1,last_bound)
        row['production_seconds']=perf_counter()-begin
        progress('independent_scoring')
        with timed(row,'scoring_seconds'):
            ref,rho=model.reference(cfg['total'])
            x=state[1]
            actual=None if ref is None else tn.distance(x,ref)
            if actual is not None and (not np.isfinite(actual) or actual>last_bound+2e-11):
                raise EvolutionError('score_inconsistent_with_bound', measured=actual, bound=last_bound)
            row.update(global_cp_bound=last_bound,global_cp_error=actual,
                       density_error=tn.distance(mpo.multiply(x,mpo.adjoint(x)),rho),
                       hermiticity_defect=tn.distance(x,mpo.adjoint(x)),
                       trace_defect=abs(tn.norm(x)**2-1),status='completed')
    except Exception as exc:
        row.update(status=getattr(exc,'status','failed'),error=str(exc),failure_details=getattr(exc,'details',{}),
                   rejected_steps=int(row['reached_time']<cfg['total']),global_cp_error=None,global_cp_bound=None)
        (out/'exception.txt').write_text(traceback.format_exc())
    finally:
        stop.set()
    row.setdefault('production_seconds',None)
    row.update(total_seconds=perf_counter()-begin,peak_rss_mib=bm.memory_mib())
    bm.save(out/'measurement.json',row)
    return row


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    worker(json.loads(args.config.read_text()),args.out)
