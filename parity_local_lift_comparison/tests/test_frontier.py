"""A frontier must represent a replicated, complete, comparable configuration."""
import pytest

from parity_local_lift_comparison.analyze import frontiers
from parity_local_lift_comparison.benchmark import config


def row(index, repeat, *, method="Local-fit", **updates):
    r = dict(config(method, n=8), run="confirm", config_index=index, repeat=repeat,
             worker=f"{index:03d}_r{repeat}", suite="confirm", T=1., steps=1,
             status="completed", last_accepted_time=1., completed_intervals=1,
             global_cp_error=.0002, production_seconds=float(repeat), scoring_seconds=.1,
             process_overhead_seconds=.4)
    r.update(updates)
    return r


def test_frontier_omits_whole_data_and_diagnostic_completions():
    rows = []
    for i, setting in enumerate((dict(width=8, fit_width=8), dict(width=3, fit_width=8),
                                 dict(width=2, fit_width=3, completion="alternate"),
                                 dict(width=1, fit_width=1, audit_inactive=True))):
        rows.extend(row(i, j, **setting) for j in (1, 2, 3))
    assert not frontiers(rows, targets=(.001,))


def test_repeats_cannot_be_pooled_across_settings_or_run_directories():
    rows = [row(1, 1), row(1, 2), row(2, 1), row(2, 2), row(1, 1, run="other")]
    assert not frontiers(rows, targets=(.001,))


def test_three_copies_of_one_worker_are_not_independent_repeats():
    rows = [row(1, 1) for _ in range(3)]
    try:
        result = frontiers(rows, targets=(.001,))
    except ValueError:
        return  # Validation may reject the malformed cohort at ingestion.
    assert all(r["status"] != "attained" for r in result)


@pytest.mark.parametrize("rejected", [dict(status="timeout"), dict(status="rank_support_uncertain"),
                                      dict(last_accepted_time=.5), dict(global_cp_error=.002),
                                      dict(global_cp_error=float("nan"))])
def test_cheap_partial_success_cannot_beat_replicated_valid_endpoint(rejected):
    rows = [row(7, j, production_seconds=.001) for j in (1, 2)]
    rows.append(row(7, 3, production_seconds=.001, **rejected))
    rows.extend(row(8, j, width=2, fit_width=3, production_seconds=float(j+1)) for j in (1, 2, 3))
    best, = frontiers(rows, targets=(.001,))
    assert best["config_index"] == 8
    assert best["settings_tested"] == 2 and best["settings_successful"] == 1
    assert (best["median_seconds"], best["min_seconds"], best["max_seconds"]) == (3., 2., 4.)
    assert best["repeats"] == 3


def test_misses_are_reported_without_successful_costs_and_protocols_stay_separate():
    rows = [row(1, j, protocol="active", global_cp_error=.02) for j in (1, 2, 3)]
    rows += [row(2, j, protocol="uniform") for j in (1, 2, 3)]
    by_protocol = {r["protocol"]: r for r in frontiers(rows, targets=(.001,))}
    assert by_protocol["active"]["status"] != "attained"
    assert "median_seconds" not in by_protocol["active"] or by_protocol["active"]["median_seconds"] is None
    assert by_protocol["uniform"]["status"] == "attained"


def test_different_horizons_parameters_and_chain_lengths_stay_separate():
    rows = []
    for i, changes in enumerate(({}, dict(T=.5, times=[.5], last_accepted_time=.5),
                                 dict(n=6), dict(p=.1), dict(b0=.1))):
        rows.extend(row(i, j, **changes) for j in (1,2,3))
    result = frontiers(rows, targets=(.001,))
    assert len(result) == 5
    assert all(r["repeats"] == 3 and r["settings_tested"] == 1 for r in result)
