"""Repeat the published parity N=512 command and QR audit in independent processes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "scalable_cp_witness" / "code"
BASE = "n512_t1_s5_parity0.6_x0.4_jensen_tdvp"
THREADS = ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
           'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')


def main(out, wall, memory):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, **{k: "1" for k in THREADS})
    settings = ["--n", "512", "--parity", "--tdvp", "--times", ".02", ".1", ".3", ".6",
                "1", "--tol", "1e-6", "--levels", "4,6,8:6,8,12", "--passes", "8"]
    stages = (("trajectory", "scalable_cp_witness.py", settings), ("qr", "validate_scalable_cp.py",
              ["--checkpoint", str(out / (BASE + "_last_accepted.npz"))]))
    record = dict(settings=settings, memory_mib=memory, wall_seconds_per_stage=wall,
                  python=sys.version, platform=platform.platform(), executable=sys.executable,
                  thread_environment={k: env[k] for k in THREADS},
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,text=True).strip(),
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(LEGACY.glob("*.py"))},
                  timing="Legacy reported time includes its analytic per-checkpoint scoring and earlier I/O.",
                  runs=[])
    for name, source, args in stages:
        # Poll child RSS inside its own process. This operates across macOS and Linux.
        child = """import sys, os, resource, threading, runpy, time
from pathlib import Path
limit=int(sys.argv.pop(1))
def guard():
    while True:
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mib=peak/(1048576 if sys.platform=='darwin' else 1024)
        if mib>limit:
            print('RSS budget exceeded',mib,flush=True)
            os._exit(73)
        time.sleep(.05)
threading.Thread(target=guard, daemon=True).start()
sys.argv=sys.argv[1:]
sys.path.insert(0,str(Path(sys.argv[0]).parent))
runpy.run_path(sys.argv[0],run_name='__main__')
"""
        start = perf_counter()
        with (out / (name + ".log")).open("w") as stream:
            try:
                result = subprocess.run([sys.executable, "-c", child, str(memory), str(LEGACY / source),
                    *args, "--out", str(out)], env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=wall)
                status = result.returncode
            except subprocess.TimeoutExpired:
                status = "wall_timeout"
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        record["runs"].append(dict(stage=name,exit=status,elapsed_seconds=perf_counter()-start,
                         peak_children_to_date_mib=peak/(1048576 if sys.platform=="darwin" else 1024)))
        (out/"provenance.json").write_text(json.dumps(record, indent=2)+"\n")
        if status != 0:
            return 1
    with (out/(BASE+".csv")).open() as f:
        history=list(csv.DictReader(f))
    if len(history)!=6 or any(r["accepted"]!="True" or float(r["actual"])>float(r["bound"])+1e-11 for r in history):
        raise ArithmeticError("Historical checkpoint discrepancy")
    return 0


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--wall", type=float, default=240)
    parser.add_argument("--memory", type=int, default=1000)
    args=parser.parse_args()
    raise SystemExit(main(args.out,args.wall,args.memory))
