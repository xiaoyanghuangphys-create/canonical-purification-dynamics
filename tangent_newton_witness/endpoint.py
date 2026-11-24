"""Transactional witness endpoints. The caller owns physical propagation and step control.

The native (factor, stored_root, attained_distance) tuple is committed only at a
passing endpoint. A predictor-less call starts from exactly the carried tuple.
The new Galerkin direction sees only a target and the current numerical factor.
"""
from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

import numpy as np

from galerkin_sylvester_tdvp.evolution import EvolutionError
from galerkin_sylvester_tdvp.galerkin_solver import GalerkinError, _evidence
from galerkin_sylvester_tdvp.models import tn, mpo, witness
from galerkin_sylvester_tdvp.witness_pair_worker import _audit


@contextmanager
def _timed(times, key):
    start = perf_counter()
    try:
        yield
    finally:
        times[key] += perf_counter() - start


def _product(a, b, limit):
    if a.n != b.n:
        raise ValueError("Product lengths disagree")
    needed = max(x.shape[0] * y.shape[0] * 4 * x.shape[-1] * y.shape[-1]
                 for x, y in zip(a.tensors, b.tensors))
    if needed > limit:
        raise GalerkinError("memory_budget", "Witness product exceeds the local core budget",
                            requested_core_elements=needed, max_elements=limit)
    return mpo.multiply(a, b)


def _positive(factor, limit):
    return _product(factor, mpo.adjoint(factor), limit)


def _distance(a, b):
    value = tn.distance(a, b)
    if not np.isfinite(value):
        raise ArithmeticError("Nonfinite MPO distance")
    return float(value)


def _fourth_diagnostics(factor, root, target, work, limit, raw_trace=None):
    """Audit W4 with an explicit interval, charging the W W† compression.

    ||P²-Q²||F <= ||P-Q||F (2 ||Q||F + ||P-Q||F). No truncated
    fourth power from this diagnostic participates in Newton's solve.
    """
    positive = _positive(factor, limit)
    compact, rounding = tn.compress(positive, work)
    approximate_fourth = _product(compact, compact, limit)
    stored_square = _product(root, root, limit)
    delta = rounding['discarded_bound']
    allowance = delta * (2 * tn.norm(compact) + delta)
    def interval(value, error):
        return [max(0., value - error), value + error]
    fourth = _distance(target, approximate_fourth)
    gap = _distance(stored_square, approximate_fourth)
    info = dict(factor_fourth_residual=interval(fourth, allowance),
                stored_square_residual=_distance(target, stored_square),
                fourth_vs_stored_square=interval(gap, allowance),
                fourth_diagnostic_allowance=allowance,
                diagnostic_positive_bond=positive.bond_dimension(),
                diagnostic_fourth_bond=approximate_fourth.bond_dimension(),
                stored_root_norm=tn.norm(root))
    if raw_trace is not None:
        raw = _distance(target, tn.scale(approximate_fourth, raw_trace))
        info['fourth_before_normalization'] = interval(raw, raw_trace * allowance)
    return info


def _direction(kind, factor, target, power, *, engine, legacy_source, anchor,
               work, mode, rtol, maxiter, max_elements, factor_floor,
               rank_rtol, damping, preconditioner):
    if kind == 'old':
        if power == 4 and anchor is None:
            return engine.correction(legacy_source, work), dict(method='legacy_stored_root_residual')
        if anchor is None:
            anchor = engine.factor_preconditioner
        return witness.congruence(anchor, legacy_source, work, 1. / power), dict(method='legacy_congruence')
    if kind == 'gn':
        from .gn import solve_direction
        return solve_direction(factor, target, power, mode=mode, damping=damping,
                               rtol=rtol, maxiter=maxiter, max_elements=max_elements)
    from .operator import solve_direction
    lower = factor_floor(factor) if factor_floor is not None else None
    return solve_direction(factor, target, power, mode=mode, lower_bound=lower,
                           rtol=rtol, maxiter=maxiter, max_elements=max_elements,
                           rank_rtol=rank_rtol, preconditioner=preconditioner)


def _retract(factor, direction, scale, bond, kind):
    raw = tn.add(factor, direction, beta=scale)
    if kind == 'gn':
        factor, info = tn.compress(raw, bond)
        return factor, info['discarded_bound']
    out = witness.hermitian_round(raw, bond)
    return out, _distance(raw, out)


def _normalize_candidate(factor, bond, limit):
    positive = _positive(factor, limit)
    norm = tn.norm(positive)
    if not np.isfinite(norm) or norm <= 0:
        raise ArithmeticError("Cannot normalize zero or nonfinite factor product")
    # Same operations and normalization as witness.factor_to_root. The trace
    # before normalization is useful for measuring its nonlinear effect.
    root = witness.hermitian_round(positive, bond, normalize=True)
    factor = tn.scale(factor, 1 / np.sqrt(norm))
    return (factor, root, _distance(root, tn.scale(positive, 1 / norm))), norm**2


def run_endpoint(engine, previous, prediction, target, time, *, levels, tolerance,
                 fit='new', correction='new', passes=4, fit_iterations=3, mode='two',
                 rtol=1e-8, maxiter=80, max_elements=2_000_000, rank_rtol=1e-13,
                 backtracks=(1., .5, .25), factor_floor=None, preconditioner='none',
                 damping=1e-8, legacy_anchors=None, on_event=None):
    """Fit then correct, choosing trials by attained distance and the original audit.

    For fixed-loss-anchor controls pass the original (fit, correction) anchor
    pair in legacy_anchors. The default delegates anchor selection to the prior
    engine (fixed for dephasing, adaptive for loss). factor_floor, if supplied,
    must give a justified lower bound on the actual argument. 'gn' uses a
    general complex factor retraction. Times are exclusive apart from the
    explicitly inclusive endpoint clock. All reported bounds are numerical.

    The returned info describes the best candidate at this endpoint, including
    on rejection; then the returned state is the *previous* accepted tuple.
    """
    algorithms = {'old', 'new', 'gn', 'none'}
    levels = list(levels)
    if fit not in algorithms or correction not in algorithms:
        raise ValueError("Unknown witness method")
    if not (levels and all(len(level) == 3 and all(int(k) == k and k > 0 for k in level)
                           for level in levels)):
        raise ValueError("Expected positive factor/root/work levels")
    if (not np.isfinite(tolerance) or tolerance <= 0 or passes < 0 or fit_iterations < 0
            or not backtracks or any(not 0 < a <= 1 for a in backtracks)
            or max_elements < 1 or mode not in ('one', 'two')):
        raise ValueError("Invalid endpoint limits")
    anchor2, anchor4 = legacy_anchors if legacy_anchors is not None else (None, None)
    # WitnessAudit installs new metric/preconditioner objects by assignment.
    # Copy references, leaving tensor data shared. Restore even on exceptions.
    snapshot = vars(engine).copy()
    committed = False
    begin = perf_counter()
    times = {name + '_seconds': 0. for name in
             ('fit', 'correction', 'reconstruction', 'certification', 'diagnostics', 'event_io')}
    attempts = []
    def emit(record):
        record = _evidence(dict(record, elapsed_seconds=perf_counter() - begin))
        attempts.append(record)
        if on_event is not None:
            with _timed(times, 'event_io_seconds'):
                on_event(record)
    def audit(root, distance):
        with _timed(times, 'certification_seconds'):
            # The engine internally allocates this exact product.
            # Check its size before entering its native implementation.
            core_multiplier = 1 if all(np.array_equal(c, c[:, [0, 2, 1, 3], :].conj())
                                       for c in root.tensors) else 2
            for j, c in enumerate(root.tensors):
                left = c.shape[0] * (core_multiplier if j else 1)
                right = c.shape[-1] * (core_multiplier if j + 1 < root.n else 1)
                if 4 * left**2 * right**2 > max_elements:
                    raise GalerkinError('memory_budget', 'Certificate square exceeds local core limit',
                                        requested_core_elements=4*left**2*right**2,
                                        max_elements=max_elements)
            return _audit(engine, root, distance, target, time, tolerance)
    def direction(kind, factor, rhs_target, power, legacy_source, anchor, work):
        return _direction(kind, factor, rhs_target, power, engine=engine,
                          legacy_source=legacy_source, anchor=anchor, work=work, mode=mode,
                          rtol=rtol, maxiter=maxiter, max_elements=max_elements,
                          factor_floor=factor_floor, rank_rtol=rank_rtol, damping=damping,
                          preconditioner=preconditioner)
    def diagnostic(factor, root, work, raw_trace=None):
        # Diagnostic failure must not penalize a method which did not need that
        # larger contraction for its direction or for its valid certificate.
        with _timed(times, 'diagnostics_seconds'):
            try:
                return _fourth_diagnostics(factor, root, target, work, max_elements, raw_trace)
            except numerical as error:
                return {'fourth_diagnostic_status': rejected(error)}
    def rejected(error):
        return dict(status=getattr(error, 'status', 'numeric_failure'),
                    message=str(error), details=getattr(error, 'details', {}))
    numerical = (ArithmeticError, np.linalg.LinAlgError, GalerkinError, EvolutionError, MemoryError)
    factor, root, eta = previous
    best = previous
    carry_bound = pred_bound = None
    predictor_selected = False
    last_error = None
    fit_updates = correction_updates = correction_trials = 0
    check = {}
    try:
        with _timed(times, 'diagnostics_seconds'):
            eta = _distance(root, _positive(factor, max_elements))
        check, residual = audit(root, eta)
        carry_bound = check['bound']
        best = factor, root, eta
        emit(dict(phase='carry', bound=carry_bound, witness=eta,
                  **diagnostic(factor, root, levels[0][2])))
        if prediction is not None:
            carry_engine = vars(engine).copy()
            try:
                fb = max(levels[0][0], factor.bond_dimension())
                work = max(levels[0][2], fb, prediction.bond_dimension())
                fitted = factor
                with _timed(times, 'fit_seconds'):
                    difference = tn.add(prediction, _positive(fitted, max_elements), beta=-1.)
                    fit_distance = tn.norm(difference)
                emit(dict(phase='fit_start', square_residual=fit_distance))
                for i in range(fit_iterations if fit != 'none' else 0):
                    if fit_distance == 0.:
                        break
                    try:
                        with _timed(times, 'fit_seconds'):
                            source = tn.compress(difference, work)[0] if fit == 'old' else None
                            update, solve = direction(fit, fitted, prediction, 2, source, anchor2, work)
                        emit(dict(phase='fit_solve', iteration=i+1, solve=solve))
                    except numerical as error:
                        last_error = rejected(error)
                        emit(dict(phase='fit_solve', iteration=i+1, failure=last_error))
                        break
                    scales = (1.,) if fit == 'old' else backtracks
                    for scale in scales:
                        try:
                            with _timed(times, 'fit_seconds'):
                                trial, rounding = _retract(fitted, update, scale, fb, fit)
                                diff = tn.add(prediction, _positive(trial, max_elements), beta=-1.)
                                distance = tn.norm(diff)
                                if not np.isfinite(distance):
                                    raise ArithmeticError('Nonfinite square fit residual')
                        except numerical as error:
                            last_error = rejected(error)
                            emit(dict(phase='fit_trial', iteration=i+1, scale=scale, improved=False,
                                      failure=last_error, factor_cap=fb))
                            continue
                        better = distance < fit_distance
                        emit(dict(phase='fit_trial', iteration=i+1, scale=scale, improved=better,
                                  square_residual=distance, factor_retraction_error=rounding,
                                  factor_cap=fb, factor_bond=trial.bond_dimension()))
                        if better:
                            fitted, difference, fit_distance = trial, diff, distance
                            fit_updates += 1
                            break
                    else:
                        break
                proposal, pr = audit(prediction, fit_distance)
                pred_bound = proposal['bound']
                predictor_selected = pred_bound < carry_bound
                emit(dict(phase='predictor', bound=pred_bound, witness=fit_distance,
                          selected=predictor_selected, **diagnostic(fitted, prediction, work)))
                if predictor_selected:
                    factor, root, eta = fitted, prediction, fit_distance
                    best, check, residual = (factor, root, eta), proposal, pr
            except numerical as error:
                last_error = rejected(error)
                vars(engine).clear()
                vars(engine).update(carry_engine)
                pred_bound, predictor_selected = None, False
                emit(dict(phase='predictor', selected=False, failure=last_error))
        for fb, rb, work in levels:
            if check['bound'] <= tolerance or correction == 'none':
                break
            for i in range(passes):
                level = dict(factor_cap=fb, root_cap=rb, work_cap=work, iteration=i+1)
                try:
                    with _timed(times, 'correction_seconds'):
                        update, solve = direction(correction, factor, target, 4, residual, anchor4, work)
                    emit(dict(phase='correction_solve', solve=solve, **level))
                except numerical as error:
                    last_error = rejected(error)
                    emit(dict(phase='correction_solve', failure=last_error, **level))
                    break
                for scale in backtracks:
                    correction_trials += 1
                    try:
                        with _timed(times, 'reconstruction_seconds'):
                            trial_f, rounding = _retract(factor, update, scale, fb, correction)
                            candidate, raw_trace = _normalize_candidate(trial_f, rb, max_elements)
                        trial_f, trial_x, trial_eta = candidate
                        trial, trial_residual = audit(trial_x, trial_eta)
                    except numerical as error:
                        last_error = rejected(error)
                        emit(dict(phase='correction_trial', scale=scale, improved=False,
                                  failure=last_error, **level))
                        continue
                    better = trial['bound'] < check['bound']
                    emit(dict(phase='correction_trial', scale=scale, improved=better,
                              factor_retraction_error=rounding, trace_before_normalization=raw_trace,
                              root_bond=trial_x.bond_dimension(), factor_bond=trial_f.bond_dimension(),
                              **trial, **level, **diagnostic(trial_f, trial_x, work, raw_trace)))
                    if better:
                        factor, root, eta = candidate
                        best, check, residual = candidate, trial, trial_residual
                        correction_updates += 1
                        break
                else:
                    break
                if check['bound'] <= tolerance:
                    break
        committed = bool(check['bound'] <= tolerance)
    except numerical as error:
        last_error = rejected(error)
        emit(dict(phase='endpoint_failure', failure=last_error))
    finally:
        if not committed:
            vars(engine).clear()
            vars(engine).update(snapshot)
    status = 'accepted' if committed else ((last_error or {}).get('status', 'witness_uncertified'))
    result = dict(check, bound=check.get('bound'), accepted=committed, status=status,
                  rollback=not committed, time=time, prior_root_bound=carry_bound,
                  predictor_bound=pred_bound, predictor_selected=predictor_selected,
                  fit_updates=fit_updates, correction_updates=correction_updates,
                  correction_trials=correction_trials, root_bond=best[1].bond_dimension(),
                  factor_bond=best[0].bond_dimension(), attempts=attempts,
                  last_failure=last_error, timings=times, endpoint_seconds=perf_counter()-begin)
    return (best if committed else previous), _evidence(result)
