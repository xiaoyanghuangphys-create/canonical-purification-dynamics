"""Serial, isolated A/B positive-witness comparison with a prespecified confirmation rule."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from . import benchmarks as bm


BRANCH_BASE = "1924e962c7cf7ab165accee7d18c234bc067eb4d"
METHODS = ("A", "B")
TARGETS = (1e-3, 1e-5, 1e-7)
THREADS = {key: "1" for key in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
          "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS")}


def hashes():
    pkg = Path(__file__).resolve().parent
    paths = [pkg / name for name in ("witness_pair_scaling.py", "witness_pair_worker.py",
             "witness_stage.py", "galerkin_solver.py", "evolution.py", "tangent_frame.py",
             "models.py", "benchmarks.py", "scaling_followup.py")]
    paths.extend(bm.ROOT / "scalable_cp_witness/code" / name for name in (
        "mpo_cp_force.py", "tdvp_core.py", "scalable_cp_witness.py", "guarded_cp_tdvp.py",
        "cp_accuracy_refinement.py"))
    return {str(p.relative_to(bm.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def configuration(family, n, method, steps, tol, args):
    parity = family == "E"
    levels = [[4,6,8],[6,8,12]] if parity else [[4,6,10],[6,8,12],[8,12,16]]
    work = (8 if parity else 10) if method == "A" else (16 if parity else 64)
    return dict(config_id=f"{family}{n:02d}_{method}_s{steps}_tol{tol:g}",
                family=family, n=n, method=method, steps=steps, total=1. if parity else .1,
                chi=2 if parity else 4, work=work, levels=levels, passes=8 if parity else 4,
                cert_tol=tol, rtol=1e-9, atol=2e-13*2.**(-n/2), maxiter=80,
                memory_mib=args.memory_mib, timeout_seconds=args.timeout)


def endpoint_score(row):
    field = "global_cp_error" if row["family"] == "E" or row["n"] <= 8 else "global_cp_bound"
    number = row.get(field)
    if (row.get("status") == "completed" and row.get("reached_time") == row["total"]
            and isinstance(number, (int,float)) and 0 <= number < float("inf")
            and isinstance(row.get("production_seconds"), (int,float)) and row["production_seconds"] > 0):
        return number
    return float("inf")


def run_case(out, config, phase, repeat, manifest, rows, enrollments):
    key = f"{phase}_{config['config_id']}_r{repeat}"
    folder = out / key
    cfg = dict(config, phase=phase, repeat=repeat, requested_repeats=3 if phase=="confirm" else 1,
               source_sha=manifest["source_sha"], source_files_sha256=manifest["source_files_sha256"],
               base_sha=BRANCH_BASE, directory=key)
    if not any(item["config_id"]==cfg["config_id"] and item["phase"]==phase for item in enrollments):
        enrollments.append({k:v for k,v in cfg.items() if k not in ("directory","repeat","source_files_sha256")})
        bm.save(out/"grid.json", enrollments)
    if folder.exists() and (folder/"measurement.json").exists():
        old = json.loads((folder/"measurement.json").read_text())
        if old.get("parent_complete") and old.get("source_files_sha256")==cfg["source_files_sha256"]:
            rows.append(old)
            return
    folder.mkdir(exist_ok=True)
    bm.save(folder/"config.json",cfg)
    cmd = [sys.executable, "-m", "galerkin_sylvester_tdvp.witness_pair_worker",
           "--config", str(folder/"config.json"), "--out", str(folder)]
    start = time.perf_counter()
    with (folder/"worker.log").open("w") as stream:
        try:
            result = subprocess.run(cmd,cwd=bm.ROOT,env=dict(os.environ,**THREADS),
                                    stdout=stream,stderr=subprocess.STDOUT,timeout=cfg["timeout_seconds"])
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = None
    wall = time.perf_counter()-start
    row = dict(cfg)
    if (folder/"measurement.json").exists() and code==0:
        row.update(json.loads((folder/"measurement.json").read_text()))
    else:
        if (folder/"progress.json").exists():
            progress=json.loads((folder/"progress.json").read_text())
            row.update(progress, worker_phase=progress.get("phase"), phase=cfg["phase"])
        row.update(status="timeout" if code is None else "worker_failed",
                   global_cp_error=None, global_cp_bound=None, returncode=code)
        if (folder/"memory_limit_event.json").exists():
            row.update(json.loads((folder/"memory_limit_event.json").read_text()))
    row.update(parent_complete=True,wall_seconds=wall,directory=key)
    # All executed workers receive this frozen manifest via config.json.
    row["source_files_sha256"]=manifest["source_files_sha256"]
    bm.save(folder/"measurement.json",row)
    rows.append(row)
    bm.save(out/"measurements.json",rows)
    bm.csv_table(out/"measurements.csv",rows)
    print(key, row["status"], row.get("global_cp_error"), row.get("global_cp_bound"),
          f"{wall:.2f}s", flush=True)


def run(args):
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=args.resume)
    fingerprint=hashes()
    envfile=out/"environment.json"
    if envfile.exists():
        manifest=json.loads(envfile.read_text())
        if fingerprint!=manifest["source_files_sha256"] or manifest["settings"]!=settings(args):
            raise ValueError("Source or settings changed since run creation; use a new output directory")
    else:
        manifest=dict(source_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=bm.ROOT,text=True).strip(),
                      base_sha=BRANCH_BASE, source_files_sha256=fingerprint, settings=settings(args),
                      python=sys.version,platform=platform.platform(),numpy=bm.np.__version__,
                      scipy=bm.scipy.__version__,blas=bm.np.__config__.show(mode="dicts")["Build Dependencies"]["blas"],
                      cpu=platform.machine(),threads=THREADS,wall_cap_seconds=args.timeout,peak_rss_guard_mib=args.memory_mib,
                      schedule="Two endpoints at 1e-5 and 1e-7. Four-endpoint retry on a missed terminal target. "
                      "If no 1e-3-qualified setting, try two endpoints with certificate tolerance 1e-3. "
                      "For each method confirm the fastest observed full-endpoint eligible setting for each target "
                      "in three fresh processes; a repeated setting serves all qualifying targets. "
                      "Additional confirmation of an identical grid for the pair if best settings differ.",
                      hardware_reference="../validation_20261119/hardware.json")
        bm.save(envfile,manifest)
    enrollments=[]
    rows=[]
    pairs=[(f,n) for n in args.sizes for f in args.families]
    def batch(configs,phase,rep):
        for c in configs:
            if hashes()!=fingerprint:
                raise RuntimeError("Source changed mid-cohort; check results and use a new run")
            run_case(out,c,phase,rep,manifest,rows,enrollments)
    for tol in args.tolerances:
        cases=[configuration(f,n,m,2,tol,args) for f,n in pairs for m in METHODS]
        batch(cases,"screen",1)
    retries=[]
    for r in rows:
        if endpoint_score(r)>r["cert_tol"]:
            retries.append(configuration(r["family"],r["n"],r["method"],4,r["cert_tol"],args))
    batch(retries,"screen",1)
    def group(f,n,m):
        return [r for r in rows if (r["family"],r["n"],r["method"])==(f,n,m) and r["phase"]=="screen"]
    loose=[configuration(f,n,m,2,1e-3,args) for f,n in pairs for m in METHODS
           if min((endpoint_score(r) for r in group(f,n,m)),default=float("inf"))>1e-3]
    batch(loose,"screen",1)
    configs={r["config_id"]:configuration(r["family"],r["n"],r["method"],r["steps"],r["cert_tol"],args) for r in rows}
    selection={}
    selected={}
    for f,n in pairs:
        for target in TARGETS:
            winners={}
            for m in METHODS:
                valid=[r for r in group(f,n,m) if endpoint_score(r)<=target]
                if valid:
                    winner=min(valid,key=lambda r:r["production_seconds"])
                    winners[m]=winner["config_id"]
                    selected[winner["config_id"]]=configs[winner["config_id"]]
            selection[f"{f}{n}_{target:g}"]=winners
            # Preserve paired setting costs even if methods selected different settings.
            if len(winners)==2:
                for winner_id in winners.values():
                    c=configs[winner_id]
                    for m in METHODS:
                        match=configuration(f,n,m,c["steps"],c["cert_tol"],args)
                        if any(r["config_id"]==match["config_id"] and endpoint_score(r)<=target for r in group(f,n,m)):
                            selected[match["config_id"]]=match
    bm.save(out/"screen_selection.json",selection)
    for r in (1,2,3):
        ordered=sorted(selected.values(),key=lambda c:(c["n"],c["family"],c["cert_tol"],c["steps"],c["method"]))
        batch(ordered if r%2 else list(reversed(ordered)),"confirm",r)
    bm.save(out/"measurements.json",rows)
    bm.csv_table(out/"measurements.csv",rows)
    print(dict(Counter(r["status"] for r in rows)),flush=True)


def settings(args):
    return dict(sizes=args.sizes,families=args.families,tolerances=args.tolerances,
                timeout=args.timeout,memory_mib=args.memory_mib)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--sizes",nargs="+",type=int,default=[8,16,32,64])
    ap.add_argument("--families",nargs="+",choices=["D","E"],default=["D","E"])
    ap.add_argument("--tolerances",nargs="+",type=float,default=[1e-5,1e-7])
    ap.add_argument("--timeout",type=float,default=180)
    ap.add_argument("--memory-mib",type=float,default=1536)
    ap.add_argument("--resume",action="store_true")
    run(ap.parse_args())
