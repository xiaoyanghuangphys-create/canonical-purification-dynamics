"""Bounded smooth restricted-tangent and singular-onset integration controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import traceback

from .benchmarks import (BASE, ROOT, append_event, csv_table, memory_mib, save,
                         start_memory_watchdog)
import numpy as np
import scipy
from scipy.sparse.linalg import expm_multiply

from factor_tdvp.rank_bridge import lindblad_superoperator
from .evolution import KrausStartup, integrate
from .models import kron_state, make_model, mpo, tn, ts


def grid():
    base = dict(family="correlated_loss", n=3, constrained=True, warm_start=False,
                preconditioner="none", work=16, steps=4, chi=2, mode="one",
                total=.08, rtol=1e-10, maxiter=80, step_doubling=False)
    rows = [base]
    rows += [dict(base, constrained=False), dict(base, warm_start=True),
             dict(base, preconditioner="product"), dict(base, preconditioner="product",warm_start=True)]
    rows += [dict(base, work=k) for k in (1,2,4,8)]
    rows += [dict(base, mode="two", chi=cap, steps=k) for cap in (1,2,4) for k in (2,4,8)]
    rows += [dict(base, mode="two", chi=4, steps=1, step_doubling=True)]
    rows += [dict(base, family="excited_loss", n=1, chi=1, total=.04, steps=k) for k in (4,8,16)]
    rows += [dict(base, family="excited_loss", n=1, chi=1, total=1e-18, steps=1)]
    return rows


def system(c):
    n = c["n"]
    if c["family"] == "excited_loss":
        initial = ts.matrix_to_mpo(np.diag([0.,1.]),1)
        seed = KrausStartup(np.zeros((2,2)),[mpo.SM])
        def target(t):
            return np.diag([np.sqrt(-np.expm1(-t)),np.exp(-t/2)])
        def generator(density):
            return mpo.onsite_dissipation(density,[1.],"loss")
        return initial, generator, target, seed
    # An exactly rank-two spatial MPO sum of positive products. Its tangent
    # typically misses physical force components even when the RHS is resolved.
    y = np.array([[0,-1j],[1j,0]])
    a = [.5*(mpo.I+.78*m) for m in (mpo.X, y, mpo.Z)]
    b = [.5*(mpo.I+.63*m) for m in (mpo.Z, mpo.X, y)]
    initial = tn.add(kron_state(a),kron_state(b),beta=.46)
    initial = tn.scale(initial,1/tn.norm(initial))
    model = make_model("loss_mixed",n,work=16)
    h,jumps = model.dense_physics()
    physical = lindblad_superoperator(h,jumps)
    x = ts.matrix(initial)
    rho = x @ x
    def target(t):
        evolved = expm_multiply(t*physical,rho.reshape(-1,order="F")).reshape(rho.shape,order="F")
        values, u = np.linalg.eigh((evolved+evolved.conj().T)/2)
        if values[0] <= 0:
            raise ArithmeticError("Independent full-rank control lost positivity")
        return (u*np.sqrt(values)) @ u.conj().T
    return initial, model.generator, target, None


def source_revision():
    names=("evolution_ablation.py","evolution.py","models.py","tangent_frame.py","galerkin_solver.py","benchmarks.py")
    source=Path(__file__).resolve().parent
    return dict(source_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                source_files_sha256={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names})


def single(c, out):
    stop=start_memory_watchdog(out,c["memory_mib"],interval=.04)
    row=dict(c, status="failed", global_cp_error=None, density_error=None)
    save(out/"source.json",source_revision())
    try:
        start = perf_counter()
        initial, gen, target, seed = system(c)
        row["setup_seconds"] = perf_counter()-start
        times=np.linspace(0,1,c["steps"]+1)
        times=c["total"]*(times**2 if seed else times)
        end, info=integrate(initial,gen,times,c["chi"],mode=c["mode"],clock="sqrt" if seed else "physical",
                            seed=seed,constrained=c["constrained"],warm_start=c["warm_start"],
                            step_doubling=c["step_doubling"],
                            step_tol=1e-5 if c["step_doubling"] else None,
                            solver_options=dict(preconditioner=c["preconditioner"],work_bond=c["work"],
                                                rtol=c["rtol"], maxiter=c["maxiter"]),
                            on_event=lambda event: append_event(out/"events.jsonl", event))
        save(out/"integration.json",info)
        for key in ("status","accepted_steps","rejected_steps","final_time","integration_seconds","force_seconds",
                    "retraction_seconds","audit_seconds","seed_seconds","initial_representation_error"):
            if key in info:
                row[key]=info[key]
        sol=[s for event in info["history"] for s in event.get("solves",[])]
        row.update(cg_iterations=sum(s["iterations"] for s in sol),
                   preconditioner_setup_seconds=sum(s["preconditioner_setup_seconds"] for s in sol),
                   peak_rss_mib=memory_mib())
        for key in ("condition_estimate","galerkin_residual_abs","physical_residual_abs","normalization_velocity_defect"):
            vals=[s[key] for s in sol if s.get(key) is not None]
            row["max_"+key]=max(vals,default=None)
        if info["status"]=="success":
            start=perf_counter()
            ref=target(c["total"])
            x=ts.matrix(end)
            row.update(global_cp_error=np.linalg.norm(x-ref),density_error=np.linalg.norm(x@x.conj().T-ref@ref),
                       scoring_seconds=perf_counter()-start)
        if seed and info["history"]:
            first=info["history"][0]
            row.update(seed_min_eigenvalue=first.get("seed_raw_min_eigenvalue"),
                       kernel_forcing=first.get("initial_support",{}).get("kernel_forcing"))
    except Exception as exc:
        (out/"exception.txt").write_text(traceback.format_exc())
        row.update(error=str(exc),status=getattr(exc,"status","failed"))
    finally:
        stop.set()
    save(out/"measurement.json",row)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--timeout",type=float,default=20.)
    parser.add_argument("--memory-mib",type=float,default=768.)
    parser.add_argument("--worker")
    args=parser.parse_args()
    if args.worker:
        return single(json.loads(args.worker),args.out)
    out=args.out
    out.mkdir(parents=True,exist_ok=False)
    save(out/"environment.json",dict(base_sha=BASE,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
                                      machine=platform.platform(),threads_requested=1,**source_revision(),
                                      max_seconds_per_case=args.timeout, memory_mib=args.memory_mib,
                                      caps="parent subprocess timeout; child peak RSS watchdog polled at 40 ms",
                                      command=sys.argv))
    cases=grid()
    save(out/"grid.json",cases)
    rows=[]
    for i,c in enumerate(cases):
        c=dict(c,memory_mib=args.memory_mib,time_limit_s=args.timeout,base_sha=BASE)
        folder=out/f"{i:02d}"
        folder.mkdir()
        try:
            with (folder/"worker.log").open("w") as log:
                result=subprocess.run([sys.executable,"-m","galerkin_sylvester_tdvp.evolution_ablation",
                                       "--out",str(folder),"--worker",json.dumps(c)],timeout=args.timeout,
                                       stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            measured=folder/"measurement.json"
            if measured.exists():
                row=json.loads(measured.read_text())
            else:
                row=dict(c,status="memory_limit" if (folder/"memory_limit_event.json").exists() else "worker_failed",
                         returncode=result.returncode,global_cp_error=None)
        except subprocess.TimeoutExpired:
            row=dict(c,status="time_limit",global_cp_error=None)
        rows.append(dict(row,directory=folder.name))
        csv_table(out/"measurements.csv",rows)
        print(folder.name,row["status"],row.get("global_cp_error"),flush=True)


if __name__=="__main__":
    main()
