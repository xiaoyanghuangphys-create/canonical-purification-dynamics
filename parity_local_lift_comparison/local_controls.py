"""Isolated validation workers for the native lift kernels and completion choices.

Run from the repository root with python -m parity_local_lift_comparison.local_controls
--output <new-directory>. This validation suite is separate from matched timings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

THREADS = ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


def configuration():
    cases = []
    for n in (2, 3, 4, 6, 8):
        for width in sorted({1, min(3, n-1)}):
            cases.append(dict(kind="dense-hp", n=n, width=width, intervals=3))
        if n <= 6:
            cases.append(dict(kind="dense-hp", n=n, width=n, intervals=2))
    for n in (5, 6):
        for b in (0., .4):
            for completion in ("native", "alternate"):
                for protocol in ("active", "uniform", "uniform-skip"):
                    cases.append(dict(kind="completion", n=n, width=2, fit_width=3,
                                      b=b, completion=completion, protocol=protocol))
    for width, fit_width in ((2, 2), (2, 3), (2, 5), (2, 6), (6, 6)):
        cases.append(dict(kind="fit-data", n=6, width=width, fit_width=fit_width))
    for method, width, fit_width in (("Half-Petz", 2, 2), ("Half-Petz", 3, 3),
                                     ("Local-fit", 3, 5), ("Local-fit", 3, 7)):
        cases.append(dict(kind="cluster", n=8, method=method, width=width,
                          fit_width=fit_width, intervals=2, horizon=.6))
    return cases


def run_case(case):
    import numpy as np
    from . import oracles as o
    from .local import LocalLift, fit, hp, _AlternateCompletion
    n, width = case["n"], case["width"]
    kind = case["kind"]
    b0 = case.get("b", .4)
    fit_width = case.get("fit_width", width)
    p = .6
    horizon = case.get("horizon", 1.)
    intervals = case.get("intervals", 1)
    if kind == "cluster":
        initial = fit.base.prepare_cluster_cp(n)
        inactive = ()
    else:
        initial = o.root_mpo(n, p, b0)
        inactive = range(n) if b0 == 0. else range(1, n)
    method = case.get("method", "Half-Petz" if kind == "dense-hp" else "Local-fit")
    protocol = case.get("protocol", "uniform" if kind == "cluster" else "active")
    lift = LocalLift(initial, method, width, fit_width, protocol, chi=4,
                     audit_inactive=kind == "completion", inactive_sites=inactive,
                     completion=case.get("completion", "native"))
    checks = {}
    if kind == "fit-data":
        physical = lift.windows[0, fit_width]
        flip = LocalLift(o.root_mpo(n, -p, b0), method, width, fit_width)
        checks["fit_cache_difference_p_vs_minus_p"] = float(
            np.linalg.norm(physical-flip.windows[0, fit_width]))
        checks["global_cp_difference_p_vs_minus_p"] = o.tn.distance(initial, o.root_mpo(n, -p, b0))
        kraus = fit.channel_operators("Z", horizon)
        noisy = fit.onsite_update(physical, fit_width, 0, kraus)
        gate = fit.fit_gate(physical, [0], 0, width, kraus)
        source, target = (o.interleaved(fit.previous.small_positive_root(x)).reshape(4**width, -1)
                          for x in (physical, noisy))
        def singular_rank(matrix):
            s = np.linalg.svd(matrix, compute_uv=False)
            return int(np.count_nonzero(s > 1e-12*s[0]))
        a, b = gate.basis.conj().T @ source, gate.basis.conj().T @ target
        cross = b @ a.conj().T
        aug = cross + 1e-12*np.eye(len(a))
        checks.update(source_rank=singular_rank(source), target_rank=singular_rank(target),
                      cross_rank=singular_rank(cross), reduced_cross_dim=len(a),
                      regularized_cross_sigma_min=float(np.linalg.svd(aug, compute_uv=False)[-1]),
                      native_local_residual=float(np.linalg.norm(gate.act(source[None])[0]-target)),
                      alternate_local_residual=float(np.linalg.norm(_AlternateCompletion(gate).act(source[None])[0]-target)))
    ds = [lift.step(horizon/intervals) for _ in range(intervals)]
    if kind == "cluster":
        expected = fit.base.analytic_root(n, horizon)
        error = float(np.linalg.norm(lift.state.dense()-o.interleaved(expected)))
        checks.update(global_cp_error=error, actual_root_bond=lift.state.bond_dimension())
    else:
        checks.update(o.score(lift.state, horizon, p, b0))
        exact = o.dense_density(n, p, b0*np.exp(-horizon))
        checks["max_physical_cache_error"] = max((
            float(np.linalg.norm(value-o.marginal(exact, range(edge, edge+size))))
            for (edge, size), value in lift.windows.items()), default=0.)
    if kind == "dense-hp":
        eig = np.linalg.eigvalsh(lift.windows[(0, width)])
        checks["physical_window_condition_number"] = float(eig[-1]/eig[0])
        rho = o.dense_density(n, p, b0)
        y = o.dense_power(rho, .5)
        for _ in range(intervals):
            y, rho = o.dense_hp_update(y, rho, horizon/intervals, width)
        checks["dense_kraus_mps_difference"] = float(np.linalg.norm(lift.state.dense()-o.interleaved(y)))
        normalized = o.tn.scale(lift.state, 1/checks["norm"])
        checks["normalized_cp_error"] = o.tn.distance(normalized, o.root_mpo(n, p, b0*np.exp(-horizon)))
    return dict(checks=checks, steps=ds, setup_cache_seconds=lift.setup_cache_seconds)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        print(json.dumps(run_case(json.load(sys.stdin)), allow_nan=False))
        return
    if args.output is None:
        parser.error("Supply a new --output directory.")
    cases = configuration()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    package = Path(__file__).resolve().parent
    repo = package.parent
    paths = (Path(__file__), package/"local.py", package/"oracles.py",
             repo/"local_cp_large/local_cp_large.py", repo/"local_cp_large/ising_cp_study.py",
             repo/"half_petz_evolution/half_petz_evolution.py",
             repo/"scalable_cp_witness/code/cluster_dephasing.py")
    manifest = dict(cases=cases, timeout_s=45, max_elements=2_000_000,
                    revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                    python=sys.version, platform=platform.platform(), threads={k:"1" for k in THREADS},
                    sha256={str(path.relative_to(repo)):hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in paths})
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    with (out/"measurements.jsonl").open("x") as stream:
        for case in cases:
            start = perf_counter()
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "parity_local_lift_comparison.local_controls", "--one"],
                    cwd=repo, input=json.dumps(case), text=True, capture_output=True,
                    env={**os.environ, **manifest["threads"]}, timeout=manifest["timeout_s"])
                row = ({"status":"completed", **json.loads(result.stdout)} if result.returncode == 0
                       else {"status":"failed", "traceback":result.stderr})
            except subprocess.TimeoutExpired:
                row = {"status":"timeout"}
            row.update(case=case, process_elapsed_s=perf_counter()-start)
            stream.write(json.dumps(row, allow_nan=False)+"\n")
            stream.flush()


if __name__ == "__main__":
    main()
