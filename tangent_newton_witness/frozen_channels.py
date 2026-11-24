"""Small frozen-checkpoint comparison of correlated dephasing and amplitude loss."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import copy
import hashlib
import itertools
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

from galerkin_sylvester_tdvp import benchmarks as bm  # set thread limits before numpy
from galerkin_sylvester_tdvp.models import Generator, tn, witness
from galerkin_sylvester_tdvp.witness_pair_worker import _audit
from factor_tdvp.tangent_solver import matrix, matrix_to_mpo
from .endpoint import run_endpoint
from .paired import prediction, save_state

import numpy as np
import scipy
from scipy.linalg import eigh


def power(x, exponent):
    values, vectors = eigh((x+x.conj().T)/2)
    if min(values) <= 0:
        raise ArithmeticError("Frozen oracle needs a full-rank previous target")
    return (vectors * values**exponent) @ vectors.conj().T


def current_floor(x):
    """Validated small-system spectral check of the stored numerical input."""
    a = matrix(x)
    vals = np.linalg.eigvalsh((a + a.conj().T)/2)
    return float(vals[0] - 64*np.finfo(float).eps*len(a)*np.linalg.norm(a, 2))


def trace_hash(*states):
    h = hashlib.sha256()
    for state in states:
        for x in state.tensors:
            h.update(str(x.shape).encode())
            h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


def checkpoint(folder, n, family, time, step, tolerance, rep, predictors):
    """Shared state preparation and predictors occur once for all comparees."""
    start = perf_counter()
    noise = 'loss' if family == 'loss' else 'dephasing'
    prev_time = time - step
    e = witness.WitnessAudit(n, .35, .8, noise=noise)
    fixed = (e.factor_preconditioner, e.preconditioner)
    prior = witness.channel(e.physical, prev_time, noise)
    w = matrix_to_mpo(power(matrix(prior), .25), n)
    state = witness.factor_to_root(w, 6)
    carry, _ = _audit(e, state[1], state[2], prior, prev_time, tolerance)
    if carry['bound'] > tolerance:
        raise ArithmeticError("Frozen seed failed its previous-time certificate")
    warm_seconds = perf_counter() - start
    tick = perf_counter()
    target = witness.channel(e.physical, time, noise)
    reference_seconds = perf_counter() - tick
    tick = perf_counter()
    r0, physical = matrix(e.physical), matrix(target)
    ref = power(physical, .5)
    ratios = eigh(physical, r0, eigvals_only=True)
    values = np.linalg.eigvalsh(matrix(state[0]))
    meta = dict(family=family, n=n, time=time, previous_time=prev_time, step=step,
                repeated_tracking=False, tol=tolerance, repeat=rep, noise=noise,
                warm_start='dense previous-time physical fourth-root oracle',
                warm_hash=trace_hash(state[0], state[1]), target_hash=trace_hash(target),
                target_over_initial=[float(ratios[0]), float(ratios[-1])],
                current_fourth_condition=float((values[-1]/values[0])**3),
                normalized_commutator=np.linalg.norm(r0@physical-physical@r0, 'fro')/
                (np.linalg.norm(r0,'fro')*np.linalg.norm(physical,'fro')),
                warm_error=np.linalg.norm(matrix(state[1])-ref),
                setup_and_oracle_seconds=warm_seconds, density_reference_seconds=reference_seconds,
                spectral_scoring_seconds=perf_counter()-tick, previous_certificate=carry['bound'])
    save_state(folder/'frozen.npz', state, prev_time, carry['bound'])
    bm.save(folder/'checkpoint.json', meta)
    cfg = dict(chi=4, work=16, rtol=1e-8, atol=2e-13*2**(-n/2), maxiter=80,
               max_elements=2_000_000)
    # floor=0 forces a check of the actual start root. At the midpoint the
    # predecessor either subtracts its measured Frobenius displacement or
    # requests a new actual-stage bound. In particular no unital lower bound
    # is reused for the loss channel. The caches now reflect the previous time.
    model = SimpleNamespace(floor=0.,
                            generator=Generator(n, noise, np.full(n, 1. if noise=='loss' else .5), work=16))
    variants = ['fixed', 'adaptive', 'new'] if noise=='loss' else ['fixed', 'new']
    ordered = variants[rep % len(variants):] + variants[:rep % len(variants)]
    rows = []
    for tag in predictors:
        pred_folder = folder/tag
        pred_folder.mkdir()
        tick = perf_counter()
        detail = defaultdict(float)
        try:
            pred = prediction(dict(cfg, predictor=None if tag=='carry' else tag),
                              state[1], prev_time, time, carry['bound'], model, e,
                              current_floor, detail, pred_folder)
        except (ArithmeticError, RuntimeError, MemoryError) as err:
            rows.append(dict(meta, predictor=tag, status='predictor_failed', error=str(err),
                             predictor_seconds=perf_counter()-tick))
            continue
        pred_time = perf_counter()-tick
        pred_hash = trace_hash(pred) if pred is not None else None
        tick = perf_counter()
        pred_error = np.linalg.norm(matrix(pred)-ref) if pred is not None else None
        pred_scoring = perf_counter() - tick
        for label in ordered:
            engine = copy(e)
            kind = 'new' if label=='new' else 'old'
            cfg_rows = dict(levels=[(4,6,10),(6,8,12),(8,12,16)], tolerance=tolerance,
                            mode='two', rtol=1e-8, maxiter=80, passes=8, fit_iterations=3)
            # Copy only committed caches. Each algorithm independently pays for
            # the same target-time metric and for any corresponding cache update.
            out, result = run_endpoint(engine, state, pred, target, time, **cfg_rows,
                                       fit=kind, correction=kind,
                                       legacy_anchors=fixed if label=='fixed' else None)
            shot = pred_folder/(label+'.json')
            bm.save(shot, result)
            tick = perf_counter()
            actual = np.linalg.norm(matrix(out[1])-ref) if result['accepted'] else None
            if actual is not None and actual > result['bound']+2e-11:
                raise ArithmeticError("Independent frozen-endpoint score exceeds the certificate")
            score_seconds = perf_counter()-tick + pred_scoring + meta['spectral_scoring_seconds']
            rows.append(dict(meta, predictor=tag, variant=label, predictor_hash=pred_hash,
                             raw_predictor_error=pred_error, predictor_seconds=pred_time,
                             predictor_clocks=dict(detail), endpoint_seconds=result['endpoint_seconds'],
                             status=result['status'], bound=result['bound'], actual_error=actual,
                             updates=result['correction_updates'], fit_updates=result['fit_updates'],
                             selected=result['predictor_selected'], scoring_seconds=score_seconds,
                             reached_time=time if result['accepted'] else prev_time,
                             peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                             (1024**2 if sys.platform=='darwin' else 1024),
                             evidence=str(shot.relative_to(folder))))
            bm.save(folder/'rows.json', rows)
    bm.save(folder/'rows.json', rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--sizes', type=int, nargs='+', default=[2,3])
    p.add_argument('--families', choices=['strong-D','loss'], nargs='+', default=['strong-D','loss'])
    p.add_argument('--times', type=float, nargs='+', default=[1.,2.,4.])
    p.add_argument('--step', type=float, default=.25)
    p.add_argument('--tolerances', type=float, nargs='+', default=[1e-5,1e-7])
    p.add_argument('--predictors', choices=['A','B','carry'], nargs='+', default=['A','B','carry'])
    p.add_argument('--reps', type=int, default=1)
    p.add_argument('--timeout', type=float, default=35.)
    p.add_argument('--worker', nargs=5)
    args=p.parse_args()
    if args.worker:
        family,n,time,tol,rep=args.worker
        checkpoint(args.out,int(n),family,float(time),args.step,float(tol),int(rep),args.predictors)
        return
    if min(args.times)<args.step or args.step<=0 or min(args.sizes)<2 or max(args.sizes)>4:
        p.error("Require 2–4 spins and a positive step no larger than the horizon")
    args.out.mkdir(parents=True,exist_ok=False)
    configs=list(itertools.product(args.families,args.sizes,args.times,args.tolerances,range(args.reps)))
    files=[Path(__file__).with_name(s) for s in ('frozen_channels.py','endpoint.py','operator.py','paired.py')]
    manifest=dict(argv=sys.argv,base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  source_sha256={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in files},
                  python=sys.version,platform=platform.platform(),machine=platform.machine(),
                  numpy=np.__version__,scipy=scipy.__version__,threads=1,configs=configs,
                  step=args.step,predictors=args.predictors,timeout=args.timeout,
                  order='rotated variants within shared predictor and checkpoint',seeds='deterministic')
    bm.save(args.out/'manifest.json',manifest)
    aggregate=[]
    for i,case in enumerate(configs):
        fam,n,time,tol,rep=case
        directory=args.out/f'{i:03d}_{fam}{n}_t{time:g}_tol{tol:g}_r{rep}'
        directory.mkdir()
        cmd=[sys.executable,'-m','tangent_newton_witness.frozen_channels','--out',str(directory),
             '--step',str(args.step),'--predictors',*args.predictors,'--worker',*(str(x) for x in case)]
        with (directory/'worker.log').open('w') as stream:
            try:
                rc=subprocess.run(cmd,timeout=args.timeout,stdout=stream,stderr=subprocess.STDOUT).returncode
                status='complete' if rc==0 else f'exit_{rc}'
            except subprocess.TimeoutExpired:
                status='timeout'
        found=directory/'rows.json'
        if found.exists():
            aggregate.extend(json.loads(found.read_text()))
        if status!='complete':
            aggregate.append(dict(family=fam,n=n,time=time,tol=tol,repeat=rep,status=status))
        bm.save(directory/'process.json',dict(status=status))
        bm.save(args.out/'measurements.json',aggregate)


if __name__=='__main__':
    main()
