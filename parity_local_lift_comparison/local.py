"""Fixed-actual-width scheduling for the repository's two local lift kernels.

Input is a canonical-gauge, normalized purification at the initial time. The
physical cache is contracted once and subsequently propagated by onsite Kraus
maps, independently of the stored output. The parity-specific identity policy
is applied only when explicitly selected.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from factor_tdvp.tangent_solver import tn
# These standalone research scripts use unqualified imports for siblings.
# Load the requested source even if a predecessor added a namesake to sys.path.
def _native(name, directory, filename):
    path = Path(__file__).resolve().parents[1] / directory
    spec = importlib.util.spec_from_file_location(name, path / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path))
    return module


hp = _native("_parity_native_half_petz", "half_petz_evolution", "half_petz_evolution.py")
fit = _native("_parity_native_local_fit", "local_cp_large", "local_cp_large.py")


def site_window(site, width, n):
    return min(max(0, site - (width - 1) // 2), n - width), width


class _HalfPetzTiming:
    """Observe two existing phase boundaries without changing native arithmetic."""

    def __init__(self, solver):
        self.solver = solver

    def __getattr__(self, name):
        return getattr(self.solver, name)

    @property
    def state(self):
        # The kernel first reads state.n during validation, and next reads
        # state just before its center move and block application.
        self.apply_started = perf_counter()
        return self.solver.state

    @property
    def affected(self):
        # Native cache propagation starts after the block has been split.
        self.cache_started = perf_counter()
        return self.solver.affected

    @property
    def cache_discarded_mass(self):
        return self.solver.cache_discarded_mass

    @cache_discarded_mass.setter
    def cache_discarded_mass(self, value):
        self.solver.cache_discarded_mass = value


class _AlternateCompletion:
    """Reflect the orthogonal complement of the joint fitting span."""

    def __init__(self, gate):
        self.gate = gate

    def act(self, block):
        left, physical, right = block.shape
        flat = block.transpose(1, 0, 2).reshape(physical, -1)
        basis = self.gate.basis
        coords = basis.conj().T @ flat
        out = -flat + basis @ (self.gate.rotation @ coords + coords)
        return out.reshape(physical, left, right).transpose(1, 0, 2)


def _block(state, start, width):
    block = state.tensors[start]
    for tensor in state.tensors[start + 1:start + width]:
        block = np.tensordot(block, tensor, axes=(-1, 0))
    return block.copy()


class LocalLift:
    def __init__(self, initial, method, width, fit_width=None, protocol="active",
                 chi=4, max_elements=2_000_000, *, completion="native",
                 audit_inactive=False, inactive_sites=None, cutoff=1e-13):
        n = initial.n
        fit_width = width if fit_width is None else fit_width
        if not (n >= 2 and 1 <= width <= fit_width <= n and chi >= 1 and 0 < cutoff < 1):
            raise ValueError("Require N>=2, 1<=W<=R<=N, chi>=1, 0<cutoff<1.")
        if method not in ("Half-Petz", "Local-fit") or protocol not in ("active", "uniform", "uniform-skip"):
            raise ValueError("Unknown local lift or sweep protocol.")
        if completion not in ("native", "alternate") or (method == "Half-Petz" and completion != "native"):
            raise ValueError("Alternate completion applies only to Local-fit.")
        if method == "Half-Petz" and fit_width != width:
            raise ValueError("Half-Petz constructs both physical roots on W, requiring R=W.")
        self.method, self.protocol, self.width, self.fit_width = method, protocol, width, fit_width
        self.chi, self.cutoff, self.completion = chi, cutoff, completion
        self.inactive = frozenset(range(1, n) if inactive_sites is None else inactive_sites)
        if not self.inactive.issubset(range(n)):
            raise ValueError("Inactive site indices must lie inside the chain.")
        self.audit_inactive = audit_inactive
        self.scheduled = (0,) if protocol == "active" else tuple(range(n))
        self.applied = tuple(site for site in self.scheduled
                             if protocol != "uniform-skip" or site not in self.inactive)
        self.site_window = {site: site_window(site, width, n) for site in self.applied}
        self.region = {site: site_window(site, fit_width, n) for site in self.applied}
        keys = sorted(set(self.region.values()))
        # Bound the largest dense operand, the aggregate physical cache, and the
        # reduced cross matrix in the compact fit (no 4^W by 4^W unitary).
        bond = max(chi, initial.bond_dimension())
        training_columns = 2 * 4**(fit_width - width)
        operands = [bond**2 * 4**fit_width, 2 * len(keys) * 4**fit_width]
        if method == "Local-fit":
            operands.extend((2 * 4**fit_width, min(4**width, training_columns)**2))
        self.guarded_elements = max(operands) if keys else 0
        if self.guarded_elements > max_elements:
            raise MemoryError(f"Local operand/cache guard {self.guarded_elements} > {max_elements} elements")
        self.state = deepcopy(initial)
        self.state.max_bond = chi
        length = self.state.norm()
        if not np.isfinite(length) or abs(length - 1) > 1e-9:
            raise ValueError("Supply a normalized initial purification.")
        if method == "Half-Petz":
            self.state.tensors[self.state.center] /= length  # native initialization
        self.windows, self.factors, self.affected = {}, {}, [[] for _ in range(n)]
        self.cache_discarded_mass = 0.
        started = perf_counter()
        for key in keys:
            raw = hp.physical_factor(self.state, *key)
            if method == "Half-Petz":
                basis, singular, discarded = hp.factor_spectrum(raw, cutoff)
                raw = basis * singular
                self.factors[key] = raw
                self.cache_discarded_mass += discarded
            self.windows[key] = raw @ raw.conj().T
            start, size = key
            for site in range(start, start + size):
                self.affected[site].append(key)
        self.setup_cache_seconds = perf_counter() - started

    def step(self, h):
        if not np.isfinite(h) or h < 0.:
            raise ValueError("Supply a finite nonnegative time interval.")
        kraus = fit.channel_operators("Z", h)
        gate_seconds = apply_seconds = cache_seconds = audit_seconds = 0.
        local_fit = inverse = maximum_motion = None
        span = bound = cache_updates = peak_split = 0
        for site in self.applied:
            start, width = self.site_window[site]
            audit = self.audit_inactive and site in self.inactive
            if audit:
                tick = perf_counter()
                self.state.move_center(start)
                before = _block(self.state, start, width)
                audit_seconds += perf_counter() - tick
            left = self.state.tensors[start].shape[0]
            right = self.state.tensors[start + width - 1].shape[-1]
            # Maximum matrix's short dimension before the truncating SVD.
            for cut in range(1, width):
                peak_split = max(peak_split, min(4 * left, right * 4**(width - cut)))
                left = min(4 * left, self.chi, right * 4**(width - cut))
            tick = perf_counter()
            if self.method == "Half-Petz":
                timed = _HalfPetzTiming(self)
                info = hp.HalfPetzEvolution.update(timed, site, kraus)
                gate_seconds += timed.apply_started - tick
                apply_seconds += timed.cache_started - timed.apply_started
                cache_seconds += perf_counter() - timed.cache_started
                inverse = max(inverse or 0., info.inverse_norm)
                bound += info.truncation_bound
            else:
                region = self.region[site]
                edge, size = region
                gate = fit.fit_gate(self.windows[region], [site-edge], start-edge, width, kraus)
                span = max(span, gate.basis.shape[1])
                local_fit = max(local_fit or 0., gate.residual)
                if self.completion == "alternate":
                    gate = _AlternateCompletion(gate)
                applying = perf_counter()
                discarded = fit.apply_compact(self.state, start, width, gate)
                caching = perf_counter()
                bound += sum(np.sqrt(discarded))
                for key in self.affected[site]:
                    edge, size = key
                    self.windows[key] = fit.onsite_update(self.windows[key], size, site-edge, kraus)
                gate_seconds += applying - tick
                apply_seconds += caching - applying
                cache_seconds += perf_counter() - caching
            cache_updates += len(self.affected[site])
            if audit:
                tick = perf_counter()
                change = float(np.linalg.norm(_block(self.state, start, width) - before))
                maximum_motion = max(maximum_motion or 0., change)
                audit_seconds += perf_counter() - tick
        return dict(nominal_site_updates=len(self.scheduled), applied_site_updates=len(self.applied),
                    skipped_site_updates=len(self.scheduled)-len(self.applied), cache_updates=cache_updates,
                    gate_seconds=gate_seconds, apply_seconds=apply_seconds, cache_seconds=cache_seconds,
                    audit_seconds=audit_seconds, max_local_fit=local_fit, max_inverse_norm=inverse,
                    max_joint_span=span if self.method == "Local-fit" else None,
                    truncation_bound=float(bound), cache_discarded_mass=self.cache_discarded_mass,
                    peak_split_matrix_min_dim_bound=peak_split, max_inactive_motion=maximum_motion)
