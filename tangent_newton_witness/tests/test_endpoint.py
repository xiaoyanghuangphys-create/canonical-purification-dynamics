"""Independent small-matrix endpoint checks and parity with historical arithmetic."""
import json

import numpy as np
import pytest

from factor_tdvp.tangent_solver import matrix, matrix_to_mpo
from galerkin_sylvester_tdvp.models import mpo, tn, witness
from galerkin_sylvester_tdvp.witness_pair_worker import _audit, _witness_endpoint
from tangent_newton_witness.endpoint import _fourth_diagnostics, _normalize_candidate, run_endpoint


def parity(n=3):
    engine = witness.WitnessAudit.parity_chain(n)
    state = witness.factor_to_root(witness.parity_power(n, .25), 4)
    return engine, state


def positive_power(x, exponent):
    vals, basis = np.linalg.eigh(x)
    return (basis * vals**exponent) @ basis.conj().T


def measured(a, interval):
    assert interval[0] - 4e-12 <= a <= interval[1] + 4e-12


def test_original_endpoint_reproduced(tmp_path):
    engine, state = parity()
    t = .4
    pred = witness.tdvp_prediction(state[1], 0., t, engine, 2, 8)
    cfg = dict(levels=[(4, 6, 8), (6, 8, 12)], cert_tol=1e-7, passes=8)
    times = {k + '_seconds': 0. for k in ('fit', 'correction', 'certification', 'density_reference')}
    baseline, reference = _witness_endpoint(engine, state, pred, t, cfg, times, tmp_path)
    engine, state = parity()
    target = witness.channel(engine.physical, t)
    candidate, result = run_endpoint(engine, state, pred, target, t, levels=cfg['levels'],
                                     tolerance=cfg['cert_tol'], passes=8, fit='old', correction='old')
    assert result['accepted'] == reference['accepted'] == True
    for key in ('predictor_selected', 'correction_trials', 'correction_updates'):
        assert result[key] == reference[key]
    assert abs(result['bound'] - reference['bound']) < 7e-13
    assert tn.distance(candidate[1], baseline[1]) < 7e-13
    actual = np.linalg.norm(matrix(candidate[1]) - positive_power(matrix(target), .5))
    assert actual <= result['bound'] + 2e-12
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('family', ['parity', 'loss'])
def test_failed_time_interval_restores_metric_and_tensors(family):
    if family == 'loss':
        engine = witness.WitnessAudit(2, .35, .8, noise='loss')
        state = witness.factor_to_root(witness.prepared_power(2, .25, .35, .8), 4)
    else:
        engine, state = parity(2)
    _audit(engine, state[1], state[2], engine.physical, 0., 1e-7)
    compare_keys = ('metric', 'metric_time', 'root_preconditioner', 'factor_preconditioner', 'preconditioner')
    snapshot = {k: vars(engine)[k] for k in compare_keys}
    cores = [[a.copy() for a in x.tensors] for x in state[:2]]
    target = witness.channel(engine.physical, .8, engine.noise)
    out, info = run_endpoint(engine, state, None, target, .8, levels=[(1, 1, 2)],
                             tolerance=1e-14, correction='old', passes=1)
    assert out is state
    assert not info['accepted'] and info['rollback'] and info['bound'] > 1e-14
    assert len(info['attempts']) > 1 and info['correction_trials'] >= 1
    assert all(vars(engine)[k] is v for k, v in snapshot.items())
    for x, initial in zip(state[:2], cores):
        assert all(np.array_equal(a, b) for a, b in zip(x.tensors, initial))


def test_wrong_sign_stored_root_and_fixed_point():
    e, state = parity(2)
    good, ok = run_endpoint(e, state, None, e.physical, 0., levels=[(4, 4, 8)], tolerance=1e-10)
    assert ok['accepted'] and ok['correction_trials'] == 0
    bad = (good[0], tn.scale(good[1], -1), 0.)  # stale supplied eta
    out, failed = run_endpoint(e, bad, None, e.physical, 0., levels=[(4, 4, 8)],
                               tolerance=1e-10, correction='none')
    assert out is bad
    assert failed['witness'] > 1.99 and failed['weighted_residual'] < 3e-13
    assert failed['bound'] > 3.99 and not failed['accepted']


def test_fourth_discrepancy_and_normalization_charged_at_low_bond():
    rng = np.random.default_rng(140)
    m = rng.standard_normal((8, 8)) + 1j*rng.standard_normal((8, 8))
    W = np.eye(8) * 2 + (m + m.conj().T)/8
    factor = matrix_to_mpo(W, 3)
    (factor, root, eta), trace = _normalize_candidate(factor, 1, 2_000_000)
    assert np.isclose(trace, np.trace(np.linalg.matrix_power(W, 4)).real)
    Wn, x = matrix(factor), matrix(root)
    assert np.isclose(np.linalg.norm(x), 1)
    assert np.isclose(np.linalg.norm(Wn @ Wn), 1)
    assert abs(eta - np.linalg.norm(Wn @ Wn - x)) < 3e-13
    target = np.eye(8)/8
    true_fourth = np.linalg.matrix_power(Wn, 4)
    for work in (1, 6):
        info = _fourth_diagnostics(factor, root, matrix_to_mpo(target, 3), work, 2_000_000, trace)
        measured(np.linalg.norm(true_fourth - target), info['factor_fourth_residual'])
        measured(np.linalg.norm(true_fourth - x@x), info['fourth_vs_stored_square'])
        measured(np.linalg.norm(np.linalg.matrix_power(W, 4) - target), info['fourth_before_normalization'])
    assert np.linalg.norm(true_fourth - x@x) > 1e-2
    assert eta > 1e-2


@pytest.mark.parametrize('mode', ['one', 'two'])
def test_factor_floor_applies_to_every_current_iterate_and_root_error(mode):
    e, state = parity(2)
    target = witness.channel(e.physical, .6)
    observed = []
    def certified(factor):
        dense = matrix(factor)
        observed.append(dense)
        return float(np.linalg.eigvalsh(dense)[0] - 3e-13*np.linalg.norm(dense))
    # The engine retains its independent metric only; remove all old witness
    # anchors to catch accidental reuse by either new direction.
    del e.factor_preconditioner, e.preconditioner, e.root_preconditioner
    pred = witness.hermitian_round(tn.add(state[1], tn.vacuum(2), beta=.01), 4, normalize=True)
    candidate, info = run_endpoint(e, state, pred, target, .6, levels=[(4, 4, 8)],
                                   mode=mode, factor_floor=certified, tolerance=1e-8,
                                   fit='new', correction='new', passes=12, maxiter=60)
    assert observed and len(observed) > 1
    assert any(np.linalg.norm(a - observed[0]) > 1e-8 for a in observed[1:])
    assert info['accepted'] and info['fit_updates'] > 0
    actual = np.linalg.norm(matrix(candidate[1]) - positive_power(matrix(target), .5))
    assert actual <= info['bound'] + 4e-12
    trials = [log for log in info['attempts']
              if log['phase'] == 'correction_trial' and log['improved']]
    assert all(a['bound'] > b['bound'] for a,b in zip(trials, trials[1:]))
    assert abs(candidate[2] - np.linalg.norm(matrix(candidate[1]) - matrix(candidate[0])@matrix(candidate[0]).conj().T)) < 3e-12


def test_fit_only_ablation_and_infeasible_target():
    e, state = parity(2)
    bad_prediction = tn.scale(state[1], -1)
    out, info = run_endpoint(e, state, bad_prediction, e.physical, 0., levels=[(4, 4, 8)],
                             fit='new', correction='none', tolerance=1e-8, fit_iterations=4)
    assert info['accepted'] and not info['predictor_selected'] and info['correction_trials'] == 0
    attained = [r['square_residual'] for r in info['attempts'] if r['phase'] == 'fit_trial' and r['improved']]
    assert all(d >= 1. - 1e-12 for d in attained)
    assert tn.distance(out[0], state[0]) < 1e-13


def test_general_factor_lsqr_uses_complex_retraction():
    e, state = parity(2)
    phase = np.diag([1, 1j, -1, -1j])
    W = matrix(state[0]) @ phase
    state = (matrix_to_mpo(W, 2), state[1], state[2])
    target = witness.channel(e.physical, .35)
    out, info = run_endpoint(e, state, None, target, .35, levels=[(4,4,8)],
                             correction='gn', tolerance=1e-7, passes=8)
    assert info['accepted'], info['last_failure']
    assert np.linalg.norm(matrix(out[0]) - matrix(out[0]).conj().T) > .5
    assert np.linalg.norm(matrix(out[1]) - positive_power(matrix(target), .5)) <= info['bound'] + 3e-12


def test_explicit_core_limit_returns_frozen_state():
    e, state = parity(3)
    out, info = run_endpoint(e, state, None, e.physical, 0., levels=[(4, 4, 4)],
                             max_elements=1, tolerance=1e-8)
    assert out is state and info['status'] == 'memory_budget' and info['rollback']
