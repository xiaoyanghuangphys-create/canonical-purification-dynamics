"""Bounded D/E follow-up using a finite-block positivity audit for D.

All branch tests operate on the represented current state. X0, its spectral
floor and optional inverse half-power are determined once by initialization.
Physical endpoint densities and analytic reference roots enter only scoring.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import traceback

from . import benchmarks as bm
from .models import make_model, tn, mpo, witness, parity_floor_setup
from .evolution import integrate
import numpy as np
import scipy
from guarded_cp_tdvp import grouped_operator_bound


def product(a, b, max_elements):
    """Cap the largest stored product core before allocating the exact product."""
    peak = max(4*x.shape[0]*y.shape[0]*x.shape[-1]*y.shape[-1] for x,y in zip(a.tensors, b.tensors))
    if peak > max_elements:
        raise MemoryError(f"product needs {peak} complex entries at one site")
    return mpo.multiply(a, b)


class AnchoredFloor:
    """Weyl lower bound on Herm(X), with Frobenius allowances for compression.

    The optional whitener must equal A^(-1/2) for the supplied Hermitian
    positive anchor A. Its operator norm is at most 1/sqrt(anchor_floor).
    All inequalities exclude floating-point roundoff.
    """
    def __init__(self, anchor, anchor_floor, whitener=None, *, bond=16, width=4, record=None,
                 max_elements=2_000_000):
        if not np.isfinite(anchor_floor) or anchor_floor <= 0 or not 1 <= width <= 4 or bond < 1:
            raise ValueError("invalid anchor floor, group width or audit bond")
        self.anchor, self.floor, self.white = anchor, anchor_floor, whitener
        self.bond, self.width, self.record, self.max_elements = bond, width, record, max_elements

    def __call__(self, state):
        delta = tn.add(state, self.anchor, beta=-1.)
        delta, info = tn.compress(delta, self.bond, tolerance=1e-14)
        e0 = info["discarded_bound"]
        frob = tn.norm(delta) + e0
        result = self.floor - frob
        row = dict(anchor_floor=self.floor, initial_delta_discard=e0, frobenius_movement_bound=frob,
                   frobenius_floor=result, width=self.width, audit_bond=self.bond, source="frobenius")
        if result <= 0:
            bound = grouped_operator_bound(delta, self.width) + e0
            row.update(grouped_movement_bound=bound, grouped_floor=self.floor-bound)
            if self.floor-bound > result:
                result = self.floor-bound
                row["source"] = "grouped"
        if result <= 0 and self.white is not None:
            # Errors e0, e1, e2 enter ||W delta W|| with weights ||W||²,
            # ||W|| and 1 respectively. Grouping only changes how the final
            # rounded operator norm is bounded.
            first, info1 = tn.compress(product(self.white, delta, self.max_elements), self.bond, tolerance=1e-14)
            relative, info2 = tn.compress(product(first, self.white, self.max_elements), self.bond, tolerance=1e-14)
            e1, e2 = info1["discarded_bound"], info2["discarded_bound"]
            rounded_bound = min(tn.norm(relative), grouped_operator_bound(relative, self.width))
            allowance = e0/self.floor + e1/np.sqrt(self.floor) + e2
            bound = rounded_bound + allowance
            row.update(congruence_rounded_bound=rounded_bound, congruence_discard1=e1,
                       congruence_discard2=e2, congruence_allowance=allowance,
                       congruence_movement_bound=bound)
            # If bound > 1, multiplication of the negative factor by a
            # *lower* bound on A cannot certify a lower bound on Herm(X).
            if bound < 1:
                result = self.floor*(1-bound)
                row.update(congruence_floor=result, source="congruence")
        if self.record:
            self.record(dict(row, lower_bound=result))
        return result


def endpoint_bound(state, density, candidate_floor, physical_root_floor, max_elements=2_000_000):
    """Use H=Herm(X) and H(H-S)+(H-S)S=H²-density, S=sqrt(density)."""
    if not (0 < candidate_floor < np.inf and 0 < physical_root_floor < np.inf):
        raise ValueError("endpoint bound requires positive finite floors")
    h = tn.add(state, mpo.adjoint(state), .5, .5)
    skew = tn.distance(h, state)
    residual = tn.distance(product(h, h, max_elements), density)
    return dict(positive_bound_skew=skew, hermitian_square_residual=residual,
                candidate_floor=candidate_floor, physical_root_floor=physical_root_floor,
                global_cp_bound=skew+residual/(candidate_floor+physical_root_floor),
                bound_kind="Hermitian Weyl floor + unital physical floor; conditional, floating-point")


def followup_grid():
    return [bm.config(f, n, "B_two", steps=2 if f=="D" else 8, chi=4 if f=="D" else 2,
                      work=64 if f=="D" else 16, total=.1 if f=="D" else 1.) for n in (8, 16, 32, 64) for f in ("D", "E")]


def worker(cfg, out):
    stop = bm.start_memory_watchdog(out, cfg["memory_mib"], interval=.04)
    begin = perf_counter()
    row = dict(cfg, status="running", accepted_steps=0, rejected_steps=0, reached_time=0.,
               global_cp_error=None, global_cp_bound=None, scoring_seconds=0.)
    try:
        f, n = cfg["family"], cfg["n"]
        model = make_model(f, n, cfg["work"])
        def save_bound(entry):
            bm.append_event(out/"floor_audit.jsonl", entry)
        floor = (AnchoredFloor(model.root, model.floor, witness.prepared_power(n, -.25),
                               record=save_bound) if f=="D" else parity_floor_setup(n))
        initial, info = tn.compress(model.root, cfg["chi"], normalize=True)
        row.update(parameters=model.parameters, initial_global_cp_error=tn.distance(initial, model.root),
                   initial_discard=info["discarded_bound"], setup_seconds=perf_counter()-begin,
                   objective=bm.OBJECTIVES["B_two"], clock="physical", preconditioner="none",
                   warm_start=False, constrained=True, support_audit="grouped_anchor" if f=="D" else "parity_fit")
        def event(e):
            bm.append_event(out/"events.jsonl", e)
            row["accepted_steps"] += int(e["accepted"])
            row["rejected_steps"] += int(not e["accepted"])
            if e["accepted"]:
                row["reached_time"] = e["time"]
            bm.save(out/"progress.json", dict(row, peak_rss_mib=bm.memory_mib()))
        bm.save(out/"progress.json", row)
        state, hist = integrate(initial, model.generator, np.linspace(0, cfg["total"], cfg["steps"]+1),
                        cfg["chi"], mode="two", dense_audit_max_n=0, initial_lower_bound=model.floor-row["initial_global_cp_error"],
                        stage_lower_bound=floor, on_event=event,
                        solver_options=dict(rtol=cfg["rtol"], atol=cfg["atol"], maxiter=cfg["maxiter"], work_bond=cfg["work"]))
        bm.save(out/"integration.json", hist)
        row.update(status=hist["status"], root_bond=state.bond_dimension(),
                   generator_max_local_discard=max(model.generator.discards, default=0.))
        for key in ("force_seconds", "retraction_seconds","audit_seconds","integration_seconds"):
            row[key] = hist[key]
        solves = [s for e in hist["history"] for s in e.get("solves", [])]
        row["cg_iterations"] = sum(s["iterations"] for s in solves)
        if solves:
            for k in ("physical_residual_abs", "galerkin_residual_abs","orthogonal_residual_abs","peak_product_bond"):
                row["max_"+k] = max(s[k] for s in solves)
        if hist["status"] == "success":
            start = perf_counter()
            result = bm.score(state, model, cfg["total"], f)
            if f == "D":
                # Uniform dephasing is unital, so rho(t) >= lambda_min(rho0) I.
                rho = witness.channel(witness.prepared_power(n, 1), cfg["total"])
                old = result["global_cp_bound"]
                sharper = endpoint_bound(state, rho, hist["history"][-1]["endpoint"]["lower_bound"], model.floor)
                result.update(sharper, positive_initial_witness_bound=old)
            row.update(result, status="completed", scoring_seconds=perf_counter()-start)
        else:
            row["failure_details"] = hist["history"][-1].get("details")
    except Exception as error:
        row.update(status="exception", error=repr(error), global_cp_error=None, global_cp_bound=None)
        (out/"exception.txt").write_text(traceback.format_exc())
    finally:
        stop.set()
    row.update(total_seconds=perf_counter()-begin, peak_rss_mib=bm.memory_mib())
    bm.save(out/"measurement.json", row)


def run(out):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    configs = followup_grid()
    hashes = bm.source_hashes()
    hashes["scaling_followup.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    head = subprocess.check_output(["git","rev-parse","HEAD"], cwd=bm.ROOT, text=True).strip()
    caps = dict(timeout_seconds=35, memory_mib=768, rss_sample_seconds=.04)
    bm.save(out/"grid.json", [dict(c, **caps) for c in configs])
    bm.save(out/"environment.json",dict(base_sha=bm.BASE, source_sha=head, source_files_sha256=hashes,
            platform=platform.platform(), python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
            blas=bm.np.__config__.show(mode="dicts")["Build Dependencies"]["blas"], requested_blas_threads=1,
            memory_limit_semantics="child peak RSS sampled every .04 s, overshoot possible", **caps))
    rows = []
    for c in configs:
        cfg = dict(c, base_sha=bm.BASE, source_sha=head, source_files_sha256=hashes, **caps)
        case = out/(c["family"]+str(c["n"]))
        case.mkdir()
        bm.save(case/"config.json", cfg)
        start = perf_counter()
        cmd = [sys.executable, "-m", "galerkin_sylvester_tdvp.scaling_followup", "--worker",
               json.dumps(cfg), "--out", str(case)]
        with (case/"worker.log").open("w") as log:
            try:
                process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=bm.ROOT,
                                         timeout=caps["timeout_seconds"])
                rc = process.returncode
            except subprocess.TimeoutExpired:
                rc = None
        if (case/"measurement.json").exists():
            row = json.loads((case/"measurement.json").read_text())
        else:
            row = dict(cfg, global_cp_bound=None, global_cp_error=None)
            if (case/"progress.json").exists():
                row.update(json.loads((case/"progress.json").read_text()))
            row.update(status="timeout" if rc is None else "worker_failed", returncode=rc)
            if (case/"memory_limit_event.json").exists():
                row.update(json.loads((case/"memory_limit_event.json").read_text()))
        row.update(directory=case.name, wall_seconds=perf_counter()-start)
        bm.save(case/"measurement.json", row)
        rows.append(row)
        bm.csv_table(out/"measurements.csv", rows)
        print(case.name, row["status"], row.get("global_cp_error"), row.get("global_cp_bound"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.worker), args.out)
    else:
        run(args.out)
