"""Independent three-spin metric, conditioning and compression experiments.

Run python -m galerkin_sylvester_tdvp.oracle_ablations --out NEW_DIRECTORY.
Dense matrices and exact eigenvalues in this module are small-system diagnostics.
Use --plot EXISTING_DIRECTORY to render the figure from the archived JSONL.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import time
import traceback

import numpy as np
import scipy
from scipy import linalg

from factor_tdvp import tangent_solver as ts
from factor_tdvp.rank_bridge import lindblad_rhs
from .galerkin_solver import GalerkinError, GalerkinOperator, solve_velocity
from .models import make_model, kron_state
from .tangent_frame import build_frame


SEED = 819
POPULATIONS = (.25, .05, .005, .0005)
SOLVE_OPTIONS = dict(rtol=1e-10, atol=1e-14, maxiter=160, max_elements=200_000)
PACKAGE = Path(__file__).resolve().parent
R = PACKAGE.parent


def correlated_root():
    """Sum of two products of positive qubit matrices, reproducible bond two."""
    rng = np.random.default_rng(SEED)
    terms = []
    for _ in range(2):
        term = np.ones((1, 1))
        for _ in range(3):
            z = rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2))
            a = z@z.conj().T+.2*np.eye(2)
            term = np.kron(term, a/linalg.norm(a))
        terms.append(term)
    x = sum(terms)
    return ts.matrix_to_mpo(x/linalg.norm(x), 3)


def basis(frame):
    """Insert each local variation directly, independently of frame.decode."""
    columns = []
    for j, (q, shape) in enumerate(zip(frame.complements, frame.shapes)):
        for a, b in np.ndindex(shape):
            local = np.zeros((q.shape[0], shape[1]), complex)
            local[:, b] = q[:, a]
            block = local.reshape(frame.ranks[j], *([4]*frame.width), frame.ranks[j+frame.width])
            for t in reversed(frame.left.tensors[:j]):
                block = np.tensordot(t, block, axes=(-1, 0))
            for t in frame.right.tensors[j+frame.width:]:
                block = np.tensordot(block, t, axes=(-1, 0))
            columns.append(ts.tn.to_matrix(block.ravel(), frame.n).ravel())
    return np.column_stack(columns)


def dense_problem(x, frame, g):
    d = x.shape[0]
    bmat = basis(frame)
    cols = bmat.T.reshape(-1, d, d)
    scols = (x@cols + cols@x).reshape(len(cols), -1).T
    m = bmat.conj().T @ scols
    rhs = bmat.conj().T @ g.ravel()
    u, q = np.linalg.eigh(x)
    f = q@((q.conj().T@g@q)/(u[:, None]+u[None, :]))@q.conj().T
    return bmat, scols, m, rhs, f


def observe_solves(record, kind, mode, root, physics, **options):
    start = time.perf_counter()
    try:
        v, diagnostics = solve_velocity(root, physics.generator, mode=mode,
                                  lower_bound=float(np.linalg.eigvalsh(ts.matrix(root))[0]),
                                  **SOLVE_OPTIONS, **options)
    except GalerkinError as error:
        record(dict(kind=kind, mode=mode, **error.details, traceback=traceback.format_exc(),
                    elapsed_seconds=time.perf_counter()-start))
        return None, None
    return ts.matrix(v), diagnostics


def objectives(record, root, physics, name):
    x = ts.matrix(root)
    h, jumps = physics.dense_physics()
    g = lindblad_rhs(x@x, h, jumps)
    for mode in ("one", "two"):
        frame = build_frame(root, mode)
        bmat, scols, metric, rhs, f = dense_problem(x, frame, g)
        radi = bmat.conj().T @ x.ravel()
        coords = [bmat.conj().T@f.ravel(), linalg.solve(metric, rhs, assume_a="pos"),
                  linalg.lstsq(scols, g.ravel())[0]]
        co = -1j*(h@x-x@h)
        rhs_co = bmat.conj().T @ (x@co+co@x).ravel()
        vco = linalg.solve(metric, rhs_co, assume_a="pos")
        vsplit = vco + linalg.solve(metric, rhs-rhs_co, assume_a="pos")
        evals = linalg.eigvalsh(metric)
        spec = linalg.eigvalsh(x)
        record(dict(kind="identity", model=name, mode=mode, dimension=frame.dimension,
                    represented_root_ranks=frame.ranks, commutator_xg=linalg.norm(x@g-g@x),
                    root_min=spec[0], root_max=spec[-1], metric_min=evals[0], metric_max=evals[-1],
                    metric_asymmetry=linalg.norm(metric-metric.conj().T), metric_condition=evals[-1]/evals[0],
                    gram_defect=linalg.norm(bmat.conj().T@bmat-np.eye(frame.dimension)),
                    galerkin_split_difference=linalg.norm(vsplit-coords[1]),
                    euclidean_coherent_shortcut_difference=linalg.norm(bmat.conj().T@co.ravel()-vco),
                    coherent_physical_identity_defect=linalg.norm(x@co+co@x+1j*(h@(x@x)-(x@x)@h)),
                    A_B_difference=linalg.norm(coords[0]-coords[1]), B_C_difference=linalg.norm(coords[1]-coords[2])))
        for label, c in zip("ABC", coords):
            v = (bmat@c).reshape(x.shape)
            err = v-f
            residual = x@v+v@x-g
            record(dict(kind="objective", model=name, mode=mode, method=label,
                        euclidean=linalg.norm(err), sylvester_weighted=np.sqrt(np.vdot(err, x@err+err@x).real),
                        density_residual=linalg.norm(residual), galerkin_stationarity=linalg.norm(bmat.conj().T@residual.ravel()),
                        normalization_drift=abs(np.vdot(x, v)), velocity_norm=linalg.norm(v)))
        free = linalg.null_space(radi.conj()[None, :])
        constrained = free@linalg.solve(free.conj().T@metric@free, free.conj().T@rhs, assume_a="pos")
        for flag in (False, True):
            v, diag = observe_solves(record, "solve_failure", mode, root, physics, constrained=flag, audit_operator=True)
            if v is None:
                continue
            c = constrained if flag else coords[1]
            radial = np.vdot(x, v).real
            retracted_v = v - radial*x
            row = dict(kind="normalization", model=name, mode=mode, constrained=flag,
                       distance_to_dense=linalg.norm(v.ravel()-bmat@c),
                       difference_after_normalizing_to_constrained=linalg.norm(retracted_v.ravel()-bmat@constrained),
                       radial_before=radial, **{k: diag[k] for k in ("iterations", "condition_estimate", "physical_residual_abs",
                       "projected_residual_abs","galerkin_residual_abs","orthogonal_residual_abs","hermiticity_defect","operator_audit")})
            record(row)


def conditioning(record):
    model = make_model("loss_full", 3)
    ham, jumps = model.dense_physics()
    for p in POPULATIONS:
        root = kron_state([np.diag(np.sqrt([1-p, p]))]*3)
        x = ts.matrix(root)
        f = build_frame(root, "two")
        bmat, _, m, _, _ = dense_problem(x, f, lindblad_rhs(x@x, ham, jumps))
        free = linalg.null_space((bmat.conj().T@x.ravel()).conj()[None, :])
        restricted = free.conj().T@m@free
        local = ts.mpo.product_preconditioner(root)
        w = np.kron(np.kron(local[0], local[1]), local[2])
        states = (bmat@free).T.reshape(-1, 8, 8)
        applied = (w@states@w).reshape(len(states), -1).T
        restricted_pre = free.conj().T@bmat.conj().T@applied
        eig, u = np.linalg.eigh(restricted_pre)
        psqrt = (u*np.sqrt(eig))@u.conj().T
        for pre in ("none", "product"):
            spectrum = np.linalg.eigvalsh(restricted if pre=="none" else psqrt@restricted@psqrt)
            velocity, diag = observe_solves(record, "conditioning_failure", "two", root, model, preconditioner=pre)
            if velocity is not None:
                record(dict(kind="conditioning", population=p, preconditioner=pre, iteration_cap=SOLVE_OPTIONS["maxiter"],
                            condition_exact=spectrum[-1]/spectrum[0], min_eigenvalue=spectrum[0], max_eigenvalue=spectrum[-1],
                            **{k: diag[k] for k in ("iterations", "history", "condition_estimate","physical_residual_abs",
                            "galerkin_residual_abs", "setup_seconds","solve_seconds","audit_seconds","preconditioner_setup_seconds")}))


def compression_and_warm(record, root, model):
    x = ts.matrix(root)
    h, jumps = model.dense_physics()
    f = build_frame(root, "one")
    _, _, m, _, _ = dense_problem(x, f, lindblad_rhs(x@x, h, jumps))
    for cap in (1, 2, 4, None):
        op = GalerkinOperator(root, f, work_bond=cap)
        if op.discard:
            op.coefficient = ts.tn.add(op.coefficient, ts.mpo.adjoint(op.coefficient), .5, .5)
        cols = np.column_stack([op.matvec(c) for c in np.eye(f.dimension)])
        audits = op.audit()
        try:
            _, diag = solve_velocity(root, model.generator, mode="one", lower_bound=float(linalg.eigvalsh(x)[0]),
                                     work_bond=cap, **SOLVE_OPTIONS)
        except GalerkinError as error:
            diag = dict(**error.details, traceback=traceback.format_exc())
        record(dict(kind="compression", work_bond=cap, fixed_discard=op.discard,
                    exact_operator_gap=linalg.norm(cols-m, 2), absolute_weyl_allowance=2*op.discard,
                    **audits, solver=diag))
    # A coherent gate on the first site moves the tangent frame but preserves
    # all active ranks. Re-evaluate the same interacting Lindbladian there.
    gate = linalg.expm(-1j*.001*np.kron(ts.mpo.X, np.eye(4)))
    nxt = ts.matrix_to_mpo(gate@x@gate.conj().T, 3)
    initial, _ = solve_velocity(root, model.generator, lower_bound=float(linalg.eigvalsh(x)[0]), **SOLVE_OPTIONS)
    results = []
    for guess in (None, initial):
        velocity, diag = observe_solves(record, "warm_failure", "one", nxt, model, initial=guess)
        if velocity is None:
            continue
        results.append(velocity)
        record(dict(kind="warm", initial="projected_previous" if guess is not None else "zero",
                    gap_to_cold=linalg.norm(results[-1]-results[0]), **{k: diag[k] for k in ("history","iterations",
                        "galerkin_residual_abs", "physical_residual_abs", "solve_seconds","audit_seconds")}))


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in (out/"measurements.jsonl").read_text().splitlines()]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.8), layout="constrained")
    for pre, color, marker in [("none", "#174E75", "o"), ("product", "#AF4C1F", "s")]:
        sub = [r for r in rows if r["kind"] == "conditioning" and r["preconditioner"] == pre]
        if not sub:
            continue
        row = min(sub, key=lambda r:r["population"])
        ax.semilogy(range(len(row["history"])), row["history"], marker=marker, markersize=4,
                    label=pre, color=color)
        bx.loglog([r["population"] for r in sub], [r["condition_exact"] for r in sub], marker=marker,
                  label=pre, color=color)
    ax.set(xlabel="PCG iteration", ylabel="Relative coordinate residual", title="Local minority population 0.0005")
    bx.set(xlabel="Local minority population", ylabel="Spectral condition number", title="Full constrained tangent spectrum")
    bx.invert_xaxis()
    for axis in (ax, bx):
        axis.legend(frameon=False, title="Preconditioner")
        axis.grid(True, which="major", alpha=.23)
    fig.savefig(out/"conditioning.png", dpi=170)
    plt.close(fig)


def run(out):
    import threading
    out.mkdir(parents=True, exist_ok=False)
    folder = PACKAGE
    src = ["oracle_ablations.py","galerkin_solver.py","tangent_frame.py","models.py"]
    wall, rss_limit, interval = 60, 512, .05
    env = dict(seed=SEED, populations=POPULATIONS, n=3, options=SOLVE_OPTIONS,
               base_sha="b337e8b7fb86fc3cccec64e13e2ddb35cc14c4e1",
               source_checkout_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=R, text=True).strip(),
               source_sha256={s:hashlib.sha256((folder/s).read_bytes()).hexdigest() for s in src},
               numpy=np.__version__, scipy=scipy.__version__, python=platform.python_version(), platform=platform.platform(),
               cpu=platform.processor(), threads={key:os.environ.get(key) for key in
                     ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS")},
               max_wall_seconds=wall, rss_limit_mib=rss_limit, rss_sampling_interval_seconds=interval)
    (out/"environment.json").write_text(json.dumps(env, indent=2)+"\n")
    data = out/"measurements.jsonl"
    start = time.perf_counter()
    def peak():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if platform.system()=="Darwin" else 1024)
    def record(row):
        row.update(wall_since_start=time.perf_counter()-start, peak_rss_mib=peak())
        with data.open("a") as s:
            s.write(json.dumps(row, allow_nan=False)+"\n")
    def timeout(*args):
        raise TimeoutError("60 second ablation wall cap")
    def watchdog():
        while not finished.wait(interval):
            if peak() > rss_limit:
                (out/"budget_failure.json").write_text(json.dumps(dict(status="memory_budget", peak_rss_mib=peak())))
                os._exit(75)
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(wall)
    finished = threading.Event()
    thread = threading.Thread(target=watchdog, daemon=True)
    thread.start()
    try:
        model = make_model("loss_full", 3)
        restricted = correlated_root()
        product = make_model("product", 3)
        objectives(record, product.root, product, "product_dephasing")
        objectives(record, restricted, model, "correlated_ising_loss")
        conditioning(record)
        compression_and_warm(record, restricted, model)
        plot(out)
    except Exception:
        (out/"failure.log").write_text(traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        finished.set()
    rows = [json.loads(s) for s in data.read_text().splitlines()]
    scalars = [{k:v for k,v in r.items() if not isinstance(v, (dict,list))} for r in rows]
    with (out/"scalars.csv").open("w", newline="") as out_csv:
        table = csv.DictWriter(out_csv, fieldnames=sorted(set().union(*scalars)))
        table.writeheader()
        table.writerows(scalars)
    print(f"{len(rows)} measurements written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path)
    group.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if args.plot is None:
        run(args.out)
    else:
        plot(args.plot)
