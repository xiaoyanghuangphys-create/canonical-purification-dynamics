"""Retracted midpoint evolution of a canonical root with the Galerkin velocity.

The supplied generator acts on the current represented density. Physical
endpoint times are separate from the integrator clock. The optional witness
uses an independently propagated physical MPO and an inverse-density metric.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
import traceback

import numpy as np

from factor_tdvp.baseline import _legacy_module
from factor_tdvp.tangent_solver import matrix, matrix_to_mpo, mpo, tn
from cp_accuracy_refinement import hermitian_round
from .galerkin_solver import GalerkinError, solve_velocity


class EvolutionError(RuntimeError):
    def __init__(self, status, **details):
        super().__init__(status)
        self.status = status
        self.details = details


def root_diagnostics(state, dense_audit_max_n=6):
    """Report actual stored-root defects, using dense spectral data only below the cap."""
    adjoint = mpo.adjoint(state)
    squared_trace = tn.overlap(adjoint, state)
    info = dict(
        root_bond=state.bond_dimension(),
        hermiticity_defect=tn.distance(state, adjoint),
        trace_defect=float(abs(squared_trace - 1)),
        norm_defect=abs(tn.norm(state) ** 2 - 1),
        min_eigenvalue=None,
        negative_part=None,
        polar_defect=None,
    )
    if state.n <= dense_audit_max_n:
        x = matrix(state)
        values = np.linalg.eigvalsh((x + x.conj().T) / 2)
        left, singular, _ = np.linalg.svd(x, full_matrices=False)
        scale = float(np.max(abs(values)))
        # Numerical spectral resolution, on the ROOT scale.
        resolution = 8 * len(x) * np.finfo(float).eps * scale
        info.update(
            min_eigenvalue=float(values[0]),
            max_eigenvalue=float(values[-1]),
            negative_part=float(np.linalg.norm(np.minimum(values, 0))),
            polar_defect=float(np.linalg.norm(x - (left * singular) @ left.conj().T)),
            support_resolution=resolution,
            numerical_support=int(np.count_nonzero(values > resolution)),
        )
    return info


def _stage_info(state, dense_cap, anchor=None, anchor_floor=None, structural=None):
    info = root_diagnostics(state, dense_cap)
    if not np.isfinite(info["hermiticity_defect"]) or info["hermiticity_defect"] > 2e-10:
        raise EvolutionError("nonhermitian_root", **info)
    if info["min_eigenvalue"] is not None:
        floor = info["min_eigenvalue"] - info["support_resolution"]
        info.update(lower_bound=floor, floor_source="dense_stage_spectrum")
        if info["min_eigenvalue"] < -info["support_resolution"]:
            raise EvolutionError("negative_root", **info)
        return info
    candidates = []
    if anchor_floor is not None:
        movement = tn.distance(state, anchor) if anchor is not None else 0.
        info["anchor_distance"] = movement
        candidates.append((anchor_floor - movement, "weyl_frobenius"))
    if structural is not None:
        lower = float(structural(state))
        if np.isfinite(lower):
            candidates.append((lower, "structural_stage_bound"))
    floor, source = max(candidates, default=(None, "unavailable"))
    info.update(lower_bound=floor, floor_source=source)
    return info


def _require_floor(info):
    lower = info["lower_bound"]
    if lower is None or not np.isfinite(lower) or lower <= 0:
        raise EvolutionError("rank_support_uncertain", **info)
    return lower


def _kernel_forcing(state, generator, info):
    """Small-system failure/seed diagnostic. Never used as an inverse on support."""
    if info.get("min_eigenvalue") is None:
        return {}
    x = matrix(state)
    val, vec = np.linalg.eigh((x + x.conj().T) / 2)
    kernel = vec[:, val <= info["support_resolution"]]
    g = generator(matrix_to_mpo(x @ x, state.n))
    if isinstance(g, tuple):
        g = g[0]
    leakage = np.linalg.norm(kernel.conj().T @ matrix(g) @ kernel)
    return dict(kernel_dimension=kernel.shape[1], kernel_forcing=float(leakage))


def _retract(raw, chi, polar, dense_cap):
    """Hermitian Pauli-coordinate TT-SVD followed by Frobenius normalization."""
    h = tn.add(raw, mpo.adjoint(raw), .5, .5)
    rounded = hermitian_round(raw, chi, normalize=False)
    discarded = tn.distance(h, rounded)
    length = tn.norm(rounded)
    if not np.isfinite(length) or length == 0:
        raise EvolutionError("retraction_failed", norm_before=length)
    result = tn.scale(rounded, 1 / length)
    info = dict(
        input_skew_removed=tn.distance(raw, h),
        svd_discard_norm=discarded, discarded_weight=discarded**2,
        norm_before=length, normalization_move=abs(1-length), polar_correction=0.,
    )
    if polar:
        if raw.n > dense_cap:
            raise EvolutionError("polar_dense_limit", n=raw.n, cap=dense_cap)
        x = matrix(result)
        u, s, _ = np.linalg.svd(x, full_matrices=False)
        positive = (u * s) @ u.conj().T
        info["polar_correction"] = float(np.linalg.norm(x - positive))
        restored, after = _retract(matrix_to_mpo(positive, raw.n), chi, False, dense_cap)
        info.update(polar_recompression=after)
        result = restored
    return result, info


@dataclass
class KrausStartup:
    """Dense finite seed for repository Ising loss and commuting dephasing.

    Supports distinct commuting dissipative layers from Hermitian involutions
    and scaled partial-isometry lowering jumps with orthogonal input/output
    support. Strang splitting places Hamiltonian halves around these layers.
    """
    hamiltonian: object
    jumps: object
    substeps: int = 1

    def __call__(self, state, elapsed):
        h = np.asarray(self.hamiltonian, complex)
        if self.substeps < 1 or elapsed <= 0 or h.shape != (2**state.n,)*2:
            raise ValueError("Invalid Kraus seed dimensions, time, or substeps")
        if not np.allclose(h, h.conj().T, rtol=1e-13, atol=1e-14):
            raise ValueError("Kraus seed needs a Hermitian Hamiltonian")
        channels = []
        normalized = []
        for op in self.jumps:
            jump = np.asarray(op, complex)
            if jump.shape != h.shape:
                raise ValueError("Kraus jump shape must match the Hamiltonian")
            rate = np.linalg.norm(jump, ord=2)**2
            if rate == 0:
                continue
            unit = jump / np.sqrt(rate)
            proj = unit.conj().T @ unit
            is_involution = (np.linalg.norm(unit - unit.conj().T) < 1e-11 and
                             np.linalg.norm(proj - np.eye(len(h))) < 1e-11)
            is_lowering = (np.linalg.norm(proj @ proj - proj) < 1e-11 and
                           np.linalg.norm(unit @ unit) < 1e-11 and
                           np.linalg.norm(proj - np.eye(len(h))) > 1e-10)
            if not (is_involution or is_lowering):
                raise EvolutionError("unsupported_seed_channel")
            # Upstream classifies each jump with an absolute 1e-12 test.
            # Refuse a lowering jump it would misidentify as an involution.
            if is_lowering and np.linalg.norm(rate*(proj-np.eye(len(h)))) < 1e-12:
                raise EvolutionError("unresolved_seed_channel_rate", rate=float(rate))
            for other in normalized:
                for adj in (other, other.conj().T):
                    if np.linalg.norm(unit @ adj - adj @ unit) > 1e-11:
                        raise EvolutionError("noncommuting_seed_layers")
            channels.append(jump)
            normalized.append(unit)
        # This legacy routine reads only the current amplitude and finite channel.
        root = _legacy_module().positive_start_root(matrix(state), h, channels, elapsed, self.substeps)
        return matrix_to_mpo(root, state.n), dict(
            seed_backend="dense_kraus_factor_svd", seed_substeps=self.substeps,
            seed_raw_min_eigenvalue=float(np.linalg.eigvalsh(root)[0]),
            seed_raw_norm=float(np.linalg.norm(root)),
        )


@dataclass
class PositiveWitness:
    """Fitted positive-factor endpoint correction with the repository's D/E audit.

    An accepted factor is committed separately, so a failed adaptive proposal
    cannot alter the next trial's starting factor. The certificate is conditional
    on the supplied engine's independent density and metric construction.
    """
    engine: object
    factor: object
    factor_bond: int
    work_bond: int
    tolerance: float = 1e-5
    iterations: int = 4

    def propose(self, predictor, physical_time, root_bond):
        import scalable_cp_witness as sw
        t0 = perf_counter()
        target = sw.channel(self.engine.physical, physical_time, self.engine.noise)
        density_seconds = perf_counter() - t0
        t0 = perf_counter()
        factor, distance = sw.fit_witness(self.factor, predictor, self.engine.factor_preconditioner,
                                         self.factor_bond, self.work_bond)
        fitting_seconds = perf_counter() - t0
        t0 = perf_counter()
        check, residual = self.engine.endpoint(predictor, distance, target, physical_time, self.tolerance)
        certification_seconds = perf_counter() - t0
        candidate = predictor
        raw_bound = float(check["bound"])
        corrections = 0
        correction_seconds = 0.
        for _ in range(self.iterations):
            if check["bound"] <= self.tolerance:
                break
            t0 = perf_counter()
            update = self.engine.correction(residual, self.work_bond)
            correction_seconds += perf_counter() - t0
            for length in (1., .5, .25):
                t0 = perf_counter()
                trial = hermitian_round(tn.add(factor, update, beta=length), self.factor_bond)
                trial, trial_root, trial_distance = sw.factor_to_root(trial, root_bond)
                correction_seconds += perf_counter() - t0
                t0 = perf_counter()
                proposal, r = self.engine.endpoint(trial_root, trial_distance, target, physical_time,
                                                    self.tolerance)
                certification_seconds += perf_counter() - t0
                if proposal["bound"] < check["bound"]:
                    candidate, factor, check, residual = trial_root, trial, proposal, r
                    corrections += 1
                    break
            else:
                break
        report = dict(
            predictor_bound=raw_bound, cp_bound=float(check["bound"]),
            witness_distance=float(check["witness"]), corrections=corrections,
            correction_distance=tn.distance(predictor, candidate),
            factor_bond=factor.bond_dimension(), witness_work_bond=self.work_bond,
            density_seconds=density_seconds, fitting_seconds=fitting_seconds,
            certification_seconds=certification_seconds, correction_seconds=correction_seconds,
            proof="conditional_jensen" if self.engine.jensen else "conditional_scalar",
        )
        if not np.isfinite(report["cp_bound"]) or report["cp_bound"] > self.tolerance:
            raise EvolutionError("witness_uncertified", **report)
        return candidate, factor, report


def integrate(initial, generator, times, chi, *, mode="one", clock="physical",
              solver_options=None, constrained=True, warm_start=False, dense_audit_max_n=6,
              initial_lower_bound=None, stage_lower_bound=None, seed=None, witness=None,
              step_doubling=False, step_tol=None, max_halvings=6, polar=False, on_event=None):
    """Evolve over strictly increasing physical times, including the initial time.

    Returns the last accepted MPO and a JSON-compatible history. A designated
    finite seed consumes the first interval when the initial root has unresolved
    support. With the sqrt clock the origin is times[0] and dt/du=2u. For large
    MPOs pass a lower bound on the supplied initial root and/or a callable
    providing a structural bound on each actual stored stage. Dense support and
    polar diagnostics are limited to dense_audit_max_n. Warm starts are physical-
    time velocity MPOs that the solver re-encodes at the new stage.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2 or np.any(~np.isfinite(times)) or not np.all(np.diff(times)>0):
        raise ValueError("times must include both endpoints and increase strictly")
    if clock not in ("physical","sqrt") or mode not in ("one","two") or chi < 1 or max_halvings < 0:
        raise ValueError("Invalid clock, tangent mode, bond cap, or halving limit")
    if step_tol is not None and (not step_doubling or step_tol <= 0):
        raise ValueError("A positive step_tol requires step doubling")
    if witness is not None and times[0] != 0.:
        raise ValueError("The benchmark witness evolves its initial physical MPO from t=0")
    options = dict(solver_options or {})
    history = []
    current = initial
    current_time = float(times[0])
    initial_accepted = False
    trace = dict(status="running", accepted_steps=0, rejected_steps=0, final_time=current_time,
                 history=history, clock=clock, constrained=constrained, warm_start=warm_start,
                 force_seconds=0., retraction_seconds=0., audit_seconds=0., witness_seconds=0.,
                 seed_seconds=0., normalization_retraction=True)
    all_start = perf_counter()
    def event(row):
        history.append(row)
        if on_event is not None:
            on_event(row)
    def timed(key, fn, *args, **kwargs):
        start = perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            trace[key] += perf_counter()-start
    def finish(status):
        trace.update(status=status, final_time=current_time, initial_accepted=initial_accepted,
                     integration_seconds=perf_counter()-all_start)
        return current, trace
    def stage(state, anchor=None, floor=None):
        return timed("audit_seconds", _stage_info, state, dense_audit_max_n,
                     anchor, floor, stage_lower_bound)
    def retract(raw):
        return timed("retraction_seconds", _retract, raw, chi, polar, dense_audit_max_n)
    def timed_velocity(state, audit, guess, record):
        if audit["lower_bound"] is None or audit["lower_bound"] <= 0:
            audit.update(timed("audit_seconds", _kernel_forcing, state, generator, audit))
        lower = _require_floor(audit)
        velocity, diagnostics = timed("force_seconds", solve_velocity, state, generator, mode=mode,
                                      lower_bound=lower, initial=guess if warm_start else None,
                                      constrained=constrained, **options)
        record["solves"].append(diagnostics)
        return velocity
    def coordinate(t):
        return float(np.sqrt(t-times[0])) if clock == "sqrt" else t-float(times[0])
    def physical(u):
        return float(times[0] + (u*u if clock == "sqrt" else u))
    def speed(u):
        return 2*u if clock == "sqrt" else 1.
    def midpoint(start_state, audit, u, h, guess, record):
        v0 = timed_velocity(start_state, audit, guess, record)
        mid, r = retract(tn.add(start_state, v0, beta=.5*h*speed(u)))
        record["retractions"].append(r)
        mid_info = stage(mid, start_state, audit["lower_bound"])
        record["stages"].append(mid_info)
        v_mid = timed_velocity(mid, mid_info, v0, record)
        candidate, r = retract(tn.add(start_state, v_mid, beta=h*speed(u+.5*h)))
        record["retractions"].append(r)
        return candidate, v_mid

    staged_errors = (EvolutionError, GalerkinError, np.linalg.LinAlgError,
                     FloatingPointError, MemoryError, ArithmeticError)
    def failure(exc):
        reason = getattr(exc, "status", "memory_limit" if isinstance(exc, MemoryError) else "numerical_failure")
        return dict(reason=reason, details=getattr(exc, "details", {}), error=str(exc),
                    traceback=traceback.format_exc())
    try:
        initial_info = stage(initial, None, initial_lower_bound)
        trace["input"] = initial_info
        if initial_info["trace_defect"] > 2e-9:
            raise EvolutionError("unnormalized_initial", **initial_info)
        current, retraction = retract(initial)
        trace["initial_representation_error"] = tn.distance(current, initial)
        audit = stage(current, initial, initial_info["lower_bound"])
        trace["initial"] = audit
        trace["initial_retraction"] = retraction
        index = 1
        if audit["lower_bound"] is None or audit["lower_bound"] <= 0:
            kernel = timed("audit_seconds", _kernel_forcing, current, generator, audit)
            audit.update(kernel)
            if seed is None or initial.n > dense_audit_max_n:
                _require_floor(audit)
            initial_accepted = True
            seed_state, seed_info = timed("seed_seconds", seed, current, times[1]-times[0])
            candidate, r = retract(seed_state)
            target_info = stage(candidate)
            if target_info["lower_bound"] is None or target_info["lower_bound"] <= 0:
                target_info.update(_kernel_forcing(candidate, generator, target_info))
                raise EvolutionError("rank_support_uncertain", endpoint=target_info,
                                     initial_support=kernel, seed_retraction=r, **seed_info)
            event(dict(kind="seed", accepted=True, time=float(times[1]), initial_support=kernel,
                       seed_retraction=r, endpoint=target_info, **seed_info))
            current, current_time, audit = candidate, float(times[1]), target_info
            trace["accepted_steps"] += 1
            index += 1
        initial_accepted = True
    except staged_errors as exc:
        entry = failure(exc)
        event(dict(kind="initial_or_seed", time=current_time, accepted=False, **entry))
        trace["rejected_steps"] += 1
        return finish(entry["reason"])

    guess = None
    endpoints = deque((float(t), 0) for t in times[index:])
    while endpoints:
        end_time, depth = endpoints.popleft()
        u = coordinate(current_time)
        h = coordinate(end_time)-u
        record = dict(kind="midpoint", from_time=current_time, time=end_time, clock_step=h,
                      halving_depth=depth, solves=[], stages=[], retractions=[])
        try:
            candidate, next_guess = midpoint(current, audit, u, h, guess, record)
            if step_doubling:
                half, half_guess = midpoint(current, audit, u, h/2, guess, record)
                half_info = stage(half, current, audit["lower_bound"])
                record["stages"].append(half_info)
                fine, next_guess = midpoint(half, half_info, u+h/2, h/2, half_guess, record)
                estimate = tn.distance(candidate, fine)/3
                record["step_doubling_estimate"] = estimate
                if step_tol is not None and estimate > step_tol:
                    raise EvolutionError("step_error_budget", estimate=estimate, tolerance=step_tol)
                candidate = fine
            factor = None
            if witness is not None:
                candidate, factor, check = timed("witness_seconds", witness.propose, candidate, end_time, chi)
                record["witness"] = check
            endpoint = stage(candidate, current, audit["lower_bound"])
            record["endpoint"] = endpoint
            _require_floor(endpoint)
        except staged_errors as exc:
            entry = failure(exc)
            event(dict(record, accepted=False, **entry))
            trace["rejected_steps"] += 1
            # Only a requested adaptive run retries. Unsupported initial data,
            # unavailable workspace, or a solver budget cannot be fixed by this rule.
            if step_tol is not None and depth < max_halvings and entry["reason"] in {
                    "step_error_budget", "negative_root", "rank_support_uncertain", "witness_uncertified"}:
                mid_time = physical(u+h/2)
                if current_time < mid_time < end_time:
                    endpoints.appendleft((end_time, depth+1))
                    endpoints.appendleft((mid_time, depth+1))
                    continue
            return finish(entry["reason"])
        current, current_time, audit, guess = candidate, end_time, endpoint, next_guess
        if witness is not None:
            witness.factor = factor
        trace["accepted_steps"] += 1
        event(dict(record, accepted=True))
    return finish("success")
