"""Matched A/B midpoint and stage-audit comparison on four-spin interacting loss.

The baseline adapter changes only the velocity callback inside an isolated worker.
Historical code, the shared Galerkin integrator and its retraction stay untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

from . import benchmarks as bench
from . import evolution
from .analyze_results import frontier
from .models import mpo, tn, ts
import numpy as np
import scipy
from scipy.linalg import solve_sylvester

A_METHODS = {"A_shared_one": "B_one", "A_shared_two": "B_two"}
MAIN_METHODS = tuple(A_METHODS) + ("B_one", "B_two")
SETTINGS = ((2, 2), (4, 4), (8, 4))


def euclidean_velocity(root, generator, *, mode, lower_bound, constrained, **options):
    """Small dense ambient Sylvester force, original MPO tangent projection."""
    if root.n > 5 or lower_bound is None or lower_bound <= 0 or not constrained:
        raise ValueError("This dense comparison requires a small positive root and a constrained stage")
    t0 = perf_counter()
    x = ts.matrix(root)
    # Apply exactly the same MPO Lindbladian on the stage's actual X^2 as B.
    gen = generator(mpo.multiply(root, root))
    g = ts.matrix(gen[0] if isinstance(gen, tuple) else gen)
    ambient = solve_sylvester(x, x, g)
    x_norm = np.linalg.norm(x)
    g_norm = np.linalg.norm(g)
    equation_defect = np.linalg.norm(x @ ambient + ambient @ x - g)
    if equation_defect > 1e-11 * max(g_norm, np.finfo(float).tiny):
        raise ArithmeticError("Ambient dense Sylvester force failed its relative residual audit")
    project = tn.tangent_project if mode == "one" else tn.tangent_project_two
    v_t = project(root, ts.matrix_to_mpo(ambient, root.n))
    radial = tn.overlap(root, v_t).real / x_norm**2
    velocity = tn.add(v_t, root, beta=-radial)
    v = ts.matrix(velocity)
    residual = x @ v + v @ x - g
    hermitian_defect = np.linalg.norm(v-v.conj().T)
    radial_defect = abs(np.vdot(x,v))
    if max(hermitian_defect, radial_defect) > 1e-9*max(np.linalg.norm(v), 1e-14):
        raise ArithmeticError("Euclidean projected velocity failed Hermiticity or normalization audit")
    info = dict(backend="A_dense_Sylvester_original_MPS_projector", mode=mode,
                ambient_equation_residual=float(equation_defect),
                physical_residual_abs=float(np.linalg.norm(residual)),
                physical_residual_rel=float(np.linalg.norm(residual)/g_norm) if g_norm else None,
                euclidean_velocity_error=float(np.linalg.norm(ambient-v)),
                hermiticity_defect=float(hermitian_defect),
                normalization_velocity_defect=float(radial_defect),
                radial_correction=float(radial), solve_seconds=perf_counter()-t0)
    return velocity, info


def source_identity():
    digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return dict(source_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=bench.ROOT,text=True).strip(),
                source_files_sha256=dict(bench.source_hashes(),restricted_comparison=digest))


def worker(config, out):
    actual=config["method"]
    route=A_METHODS.get(actual, actual)
    original=evolution.solve_velocity
    native_objective=bench.OBJECTIVES[route]
    # Reuse the exact benchmark worker for timing, setup, event recording, stage
    # guards, independent reference scoring and memory monitoring in every row.
    # The routing label is transient; the public record names the actual objective.
    if actual in A_METHODS:
        evolution.solve_velocity=euclidean_velocity
        bench.OBJECTIVES[route]="Euclidean dense Sylvester projection with shared retraction and audits"
    try:
        bench.worker(dict(config,method=route),out)
    finally:
        evolution.solve_velocity=original
        bench.OBJECTIVES[route]=native_objective
    measured=out/"measurement.json"
    row=json.loads(measured.read_text())
    shared=actual in MAIN_METHODS
    row.update(method=actual,same_midpoint_retraction_and_stage_checks=shared,
               geometry="shared normalized Hermitian TT-SVD" if shared else
                        ("original dense polar" if actual.startswith("A_dense") else "positive-factor representation"),
               **source_identity())
    bench.save(measured,row)


def configs(smoke):
    settings=SETTINGS[:1] if smoke else SETTINGS
    for chi,steps in settings:
        methods=list(MAIN_METHODS)
        if not smoke:
            methods += ["A_dense_two"]
            if chi in (2,4):
                methods.append("factor_lsqr")
        for method in methods:
            yield bench.config("loss_mixed",4,method,steps=steps,chi=chi,work=32,total=.03,
                               repeats=1 if smoke else 3)


def run(args):
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    grid=list(configs(args.smoke))
    bench.save(out/"grid.json",grid)
    bench.save(out/"environment.json",dict(**source_identity(),base_sha=bench.BASE,
               command=sys.argv,platform=platform.platform(),python=sys.version,numpy=np.__version__,
               scipy=scipy.__version__,threads_requested=1,serial=True,
               memory_mib=args.memory_mib,wall_seconds_per_case=args.timeout,total_budget_s=args.budget,
               resource_semantics="parent wall timeout; child peak RSS watchdog sampled at 40 ms",
               endpoints="matched physical times and independent Lindblad expm scoring",
               target_rule="all 3 repeats must finish and attain measured global CP accuracy; fastest median worker time"))
    rows=[]
    start=perf_counter()
    # Interleave configurations by repetition to distribute time-of-run drift.
    for r in range(1,grid[0]["repeats"]+1):
        for i,c in enumerate(grid):
            label=f"{i:03d}_{c['method']}_chi{c['chi']}_s{c['steps']}_r{r}"
            case=out/label
            case.mkdir()
            remaining=args.budget - (perf_counter()-start)
            cfg=dict(c,repeat=r,memory_mib=args.memory_mib,base_sha=bench.BASE,
                     source_sha=source_identity()["source_sha"],timeout_s=min(args.timeout,max(remaining,0.)))
            bench.save(case/"configuration.json",cfg)
            began=perf_counter()
            row=dict(cfg,status="not_run_budget",global_cp_error=None,global_cp_bound=None)
            if remaining>0:
                try:
                    with (case/"worker.log").open("w") as log:
                        result=subprocess.run([sys.executable,"-m","galerkin_sylvester_tdvp.restricted_comparison",
                                              "--worker",json.dumps(cfg),"--out",str(case)],cwd=bench.ROOT,
                                              timeout=min(args.timeout,remaining),stdout=log,stderr=subprocess.STDOUT)
                    if (case/"measurement.json").exists():
                        row=json.loads((case/"measurement.json").read_text())
                        if row.get("method") != c["method"]:
                            raise RuntimeError("Worker did not finish the method-specific report")
                    elif (case/"memory_limit_event.json").exists():
                        row.update(json.loads((case/"memory_limit_event.json").read_text()))
                    else:
                        row.update(status="worker_failed",exit_code=result.returncode)
                except subprocess.TimeoutExpired:
                    row.update(status="time_limit")
            row.update(wall_seconds=perf_counter()-began,directory=label)
            bench.save(case/"measurement.json",row)
            rows.append(row)
            bench.csv_table(out/"measurements.csv",rows)
            print(label,row["status"],row.get("global_cp_error"),flush=True)
    settings,winners=frontier(rows)
    bench.csv_table(out/"configurations.csv",settings)
    bench.csv_table(out/"matched_accuracy.csv",winners)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--timeout",type=float,default=35.)
    p.add_argument("--budget",type=float,default=900.)
    p.add_argument("--memory-mib",type=float,default=768.)
    p.add_argument("--worker")
    args=p.parse_args()
    if args.worker:
        worker(json.loads(args.worker),args.out)
    else:
        run(args)


if __name__=="__main__":
    main()
