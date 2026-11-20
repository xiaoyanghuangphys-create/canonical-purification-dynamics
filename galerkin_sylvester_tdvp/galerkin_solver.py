"""Tangent Galerkin solve for the positive canonical root of a density MPO.

The generator callback takes the represented *current* X². A positive lower
bound on the current Hermitian X is required from the caller. Production
routines use no dense Hilbert-space vectors or explicit tangent columns.
Only dense_oracle exports global matrices, behind an explicit size guard.
"""
from __future__ import annotations

import copy
from time import perf_counter

import numpy as np
from scipy.linalg import eigh_tridiagonal, solve_sylvester

from factor_tdvp.tangent_solver import mpo, tn
from .tangent_frame import build_frame


def _evidence(value):
    """Convert diagnostic data (including arithmetic failures) to strict JSON."""
    if isinstance(value, dict):
        return {str(key): _evidence(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _evidence(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_evidence(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, complex):
        return dict(real=_evidence(value.real), imag=_evidence(value.imag))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f'Unsupported generator diagnostic type {type(value).__name__}')


class GalerkinError(RuntimeError):
    """A rejected solve, carrying JSON-serializable numerical evidence."""

    def __init__(self, status, message, **details):
        super().__init__(message)
        self.status = status
        self.details = _evidence(dict(status=status, **details))


def bond(state):
    return max(a.shape[-1] for a in state.tensors)


def _geometric_reduce(state):
    """Exact QR reduction when a virtual bond exceeds the ambient cut rank."""
    n = state.n
    if any(a.shape[-1] > 4**min(j + 1, n - j - 1)
           for j, a in enumerate(state.tensors[:-1])):
        state = copy.deepcopy(state)
        state.center = 0
        state.move_center(n - 1)
        state.move_center(0)
    return state


def _fixed_round(state, cap):
    if cap is None or bond(state) <= cap:
        return state, 0.
    result, info = tn.compress(state, cap, tolerance=0.)
    return result, info['discarded_bound']


def _ratio(value, reference):
    if reference == 0:
        return 0. if value == 0 else None
    result = float(value / reference)
    return result if np.isfinite(result) else None


class GalerkinOperator:
    """v -> B†(X Bv + Bv X), with at most one exact MPO product held per term.

    work_bond optionally rounds the fixed coefficient once. Its SVD discard
    bounds the absolute coefficient perturbation. Decoding, products and
    encoding inside every Krylov application are lossless up to roundoff.
    max_elements limits the largest raw product site before allocation.
    """

    def __init__(self, root, frame, *, work_bond=None, max_elements=2_000_000):
        if work_bond is not None and (int(work_bond) != work_bond or work_bond < 1):
            raise ValueError('work_bond must be a positive integer')
        if max_elements < 1:
            raise ValueError('max_elements must be positive')
        self.root, self.frame = root, frame
        coefficient, self.discard = _fixed_round(root, work_bond)
        # An MPO with individually adjoint-fixed cores is exactly Hermitian.
        # In any other gauge take the Hermitian part once, irrespective of
        # the observed skew. This makes the represented Krylov map HPD under
        # the supplied bound even when the smallest eigenvalue is very small.
        self.manifest_hermitian = all(np.array_equal(a, a[:, [0, 2, 1, 3], :].conj())
                                      for a in coefficient.tensors)
        if self.manifest_hermitian:
            self.coefficient = coefficient
        else:
            hermitian_sizes = [a.size * (2 if j > 0 else 1) * (2 if j < root.n-1 else 1)
                               for j, a in enumerate(coefficient.tensors)]
            if max(hermitian_sizes) > max_elements:
                raise GalerkinError('memory_budget', 'Hermitian coefficient exceeds core element budget',
                                    requested_core_elements=max(hermitian_sizes), max_elements=max_elements)
            self.coefficient = _geometric_reduce(tn.add(coefficient, mpo.adjoint(coefficient), .5, .5))
        self.peak_product_bond = 0
        self.peak_product_core_elements = 0
        self.max_elements = max_elements
        self.applications = 0

    def product(self, a, b):
        sizes = [(x.shape[0] * y.shape[0], x.shape[-1] * y.shape[-1])
                 for x, y in zip(a.tensors, b.tensors)]
        largest = max(l * 4 * r for l, r in sizes)
        self.peak_product_core_elements = max(self.peak_product_core_elements, largest)
        self.peak_product_bond = max(self.peak_product_bond, max(r for _, r in sizes))
        if largest > self.max_elements:
            raise GalerkinError('memory_budget', 'Exact product core exceeds element budget',
                                requested_core_elements=largest, max_elements=self.max_elements)
        return _geometric_reduce(mpo.multiply(a, b))

    def matvec(self, vector):
        self.applications += 1
        v = _geometric_reduce(self.frame.decode(vector))
        return (self.frame.encode(self.product(self.coefficient, v))
                + self.frame.encode(self.product(v, self.coefficient)))

    def audit(self, seed=731):
        """Two deterministic complex probes of linearity, symmetry and curvature."""
        rng = np.random.default_rng(seed)
        u, v = rng.normal(size=(2, self.frame.dimension)) + 1j*rng.normal(size=(2, self.frame.dimension))
        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)
        mu, mv = self.matvec(u), self.matvec(v)
        scale = max(np.linalg.norm(mu), np.linalg.norm(mv), np.finfo(float).tiny)
        return dict(linearity_defect=float(np.linalg.norm(self.matvec(u + 1j*v) - mu - 1j*mv)/scale),
                    self_adjoint_defect=float(abs(np.vdot(u, mv) - np.vdot(mu, v))/scale),
                    sampled_rayleigh=[float(np.vdot(u, mu).real), float(np.vdot(v, mv).real)])


def _positive_curvature(p, applied, label, iteration):
    c = np.vdot(p, applied)
    scale = np.linalg.norm(p)*np.linalg.norm(applied)
    if not np.isfinite(c) or c.real <= 0 or abs(c.imag) > 1e-9 * scale:
        raise GalerkinError('curvature_failure', label + ' is not positive real',
                            iteration=iteration, real=float(c.real), imag=float(c.imag), scale=float(scale))
    return c.real


def _ritz_condition(alphas, betas):
    """Extremal Ritz values of the symmetrically preconditioned scaled system."""
    if not alphas:
        return None
    a = np.asarray(alphas)
    off = np.sqrt(np.asarray(betas[:len(a)-1])) / a[:-1]
    diag = 1/a
    diag[1:] += np.asarray(betas[:len(a)-1]) / a[:-1]
    ev = eigh_tridiagonal(diag, off, eigvals_only=True)
    return float(ev[-1] / ev[0]) if ev[0] > 0 else None


def _pcg(apply, pre, start, rhs, threshold, maxiter):
    rhs_norm = np.linalg.norm(rhs)
    if rhs_norm == 0:
        return np.zeros_like(rhs), [0.], None
    x = start.copy()
    r = rhs - apply(x) if np.any(x) else rhs.copy()
    residual = np.linalg.norm(r)
    history = [_ratio(residual, rhs_norm)]
    alphas, betas = [], []
    if residual <= threshold:
        return x, history, None
    z = pre(r)
    rz = _positive_curvature(r, z, 'Preconditioned residual', 0)
    p = z.copy()
    for k in range(1, maxiter + 1):
        ap = apply(p)
        alpha = rz / _positive_curvature(p, ap, 'Galerkin curvature', k)
        alphas.append(alpha)
        x += alpha*p
        r -= alpha*ap
        residual = np.linalg.norm(r)
        history.append(_ratio(residual, rhs_norm))
        if residual <= threshold:
            return x, history, _ritz_condition(alphas, betas)
        z = pre(r)
        rz_new = _positive_curvature(r, z, 'Preconditioned residual', k)
        beta = rz_new/rz
        betas.append(beta)
        p = z + beta*p
        rz = rz_new
    raise GalerkinError('nonconvergence', 'Iteration limit before the coordinate tolerance',
                        iterations=maxiter, history=history, condition_estimate=_ritz_condition(alphas, betas))


def solve_velocity(root, generator, *, mode='one', lower_bound=None, constrained=True,
                   rtol=1e-9, atol=1e-13, maxiter=120, work_bond=None,
                   preconditioner='none', initial=None, max_elements=2_000_000,
                   physical_rtol=None, hermitian_tol=2e-9, compression_rtol=1e-9,
                   audit_operator=False, rank_rtol=1e-13):
    """Return (V as an MPO, diagnostics) or raise GalerkinError.

    lower_bound certifies lambda_min of the Hermitian part of the current
    input, with its provenance established by the caller. None/zero refuses
    singular evolution before projecting any forcing away. The physical
    residual limit is optional because the chosen space can miss G. The
    free complex tangent is adjoint invariant at Hermitian X; project the
    computed solution onto its measured adjoint-fixed part before final audit.
    """
    t0 = perf_counter()
    if (not np.isfinite([rtol, atol, hermitian_tol, compression_rtol]).all()
            or rtol < 0 or atol < 0 or not (rtol or atol) or maxiter < 0
            or compression_rtol < 0 or hermitian_tol <= 0
            or (physical_rtol is not None and (not np.isfinite(physical_rtol) or physical_rtol < 0))):
        raise ValueError('Invalid solver tolerance or iteration cap')
    if lower_bound is None or not np.isfinite(lower_bound) or lower_bound <= 0:
        raise GalerkinError('rank_support_uncertain', 'A positive current-root lower bound is required',
                            spectral_lower_bound=lower_bound)
    root_norm = tn.norm(root)
    xdef = tn.distance(root, mpo.adjoint(root))
    if xdef > hermitian_tol * root_norm:
        raise GalerkinError('hermiticity_failure', 'Non-Hermitian current root', root_hermiticity_defect=xdef)
    scale = 2 * float(mpo.trace(root).real) * 2.**(-root.n)
    if not np.isfinite(scale) or scale <= 0 or scale * .5 < lower_bound*(1-1e-10):
        raise GalerkinError('rank_support_uncertain', 'Root trace contradicts supplied spectrum bound',
                            spectral_lower_bound=lower_bound, mean_root_eigenvalue=scale*.5)
    frame = build_frame(root, mode, rank_rtol=rank_rtol)
    op = GalerkinOperator(root, frame, work_bond=work_bond, max_elements=max_elements)
    # Hermitian-part projection is contractive in Frobenius norm. Relative
    # to Herm(X), the coefficient perturbation is at most its SVD discard.
    if op.discard >= lower_bound or op.discard > compression_rtol*root_norm:
        raise GalerkinError('compression_overflow', 'Fixed coefficient rounding exceeds its budget',
                            coefficient_discard=op.discard, spectral_lower_bound=lower_bound,
                            compression_budget=compression_rtol*root_norm)
    density = op.product(root, root)
    generated = generator(density)
    g, gen_info = generated if isinstance(generated, tuple) else (generated, {})
    if g.n != root.n:
        raise ValueError('Generator returned a different chain size')
    if not all(np.isfinite(a).all() for a in g.tensors):
        raise GalerkinError('nonfinite_generator', 'Generator produced a nonfinite MPO')
    g = _geometric_reduce(g)
    g_norm = tn.norm(g)
    gdef = tn.distance(g, mpo.adjoint(g))
    if gdef > max(atol, hermitian_tol * g_norm):
        raise GalerkinError('hermiticity_failure', 'Generator did not preserve Hermiticity', generator_defect=gdef)
    g_work, rhs_discard = _fixed_round(g, work_bond)
    if rhs_discard > max(atol, compression_rtol*g_norm):
        raise GalerkinError('compression_overflow', 'Fixed source rounding exceeds its budget',
                            rhs_discard=rhs_discard, compression_budget=max(atol, compression_rtol*g_norm))
    b = frame.encode(g_work)
    c = frame.encode(root)
    c_norm = np.linalg.norm(c)
    normal = c / c_norm

    def null(v):
        return v - normal*np.vdot(normal, v) if constrained else v

    target = null(b)
    threshold = max(atol, rtol*np.linalg.norm(target))
    pstart = perf_counter()
    if preconditioner == 'product':
        # Q B† [W (B·) W] B Q is HPD on the constraint nullspace.
        # W is a tensor product of local positive inverse quarter powers.
        local = mpo.product_preconditioner(root)

        def pre(v):
            state = frame.decode(v)
            tensors = []
            for a, w in zip(state.tensors, local):
                block = a.reshape(a.shape[0], 2, 2, a.shape[-1])
                tensors.append(np.einsum('pi,aijb,jq->apqb', w, block, w, optimize=True).reshape(a.shape))
            state.tensors = tensors
            return null(frame.encode(state))
    elif preconditioner == 'none':
        def pre(v):
            return v
    else:
        raise ValueError("preconditioner must be 'none' or 'product'")
    pre_seconds = perf_counter() - pstart
    v0 = null(frame.encode(initial)) if initial is not None else np.zeros_like(b)
    probes = op.audit() if audit_operator else {}
    t1 = perf_counter()
    v, history, condition = _pcg(lambda p: null(op.matvec(null(p))) / scale, pre,
                                 v0, target/scale, threshold/scale, maxiter)
    t2 = perf_counter()
    before = _geometric_reduce(frame.decode(v))
    adj = mpo.adjoint(before)
    adj_v = frame.encode(adj)
    vnorm = tn.norm(before)
    adj_leak = tn.distance(adj, _geometric_reduce(frame.decode(adj_v)))
    h_before = tn.distance(before, adj)
    if adj_leak > hermitian_tol * max(vnorm, np.finfo(float).tiny):
        raise GalerkinError('hermiticity_failure', 'Adjoint escaped active tangent',
                            adjoint_tangent_leakage=adj_leak, velocity_norm=vnorm)
    v = null((v + adj_v)*.5)
    velocity = _geometric_reduce(frame.decode(v))
    h_after = tn.distance(velocity, mpo.adjoint(velocity))
    left = op.product(root, velocity)
    right = op.product(velocity, root)
    res = _geometric_reduce(tn.add(_geometric_reduce(tn.add(left, right)), g, beta=-1.))
    physical = tn.norm(res)
    projected_vector = frame.encode(res)
    raw_projected = np.linalg.norm(projected_vector)
    stationarity = np.linalg.norm(null(projected_vector))
    # Directly measure R - BB†R. The difference of squared norms loses
    # precision when R is almost entirely tangent (including full tangents).
    orthogonal = tn.distance(res, _geometric_reduce(frame.decode(projected_vector)))
    b_exact = frame.encode(g)
    exact_target = np.linalg.norm(null(b_exact))
    audit_tol = max(atol, rtol*exact_target)
    # Relative stationarity of a roundoff-zero target is ill-conditioned;
    # the explicit absolute term covers that case. No physical residual is
    # used as a replacement for an unmet coordinate tolerance.
    multiplier = -np.vdot(c, projected_vector) / c_norm**2 if constrained else 0.
    diagnostics = dict(status='success', constrained=bool(constrained), mode=mode,
        dimension=frame.dimension, state_bonds=list(frame.ranks), frame_metadata=frame.metadata,
        work_bond=work_bond, coefficient_hermitian_construction=(
            'adjoint_fixed_cores' if op.manifest_hermitian else 'explicit_hermitian_part'),
        iterations=len(history)-1, history=history, condition_estimate=condition,
        preconditioner=preconditioner, preconditioner_setup_seconds=pre_seconds,
        preconditioner_local_density_floor=1e-12 if preconditioner == 'product' else None,
        spectral_lower_bound=float(lower_bound), effective_lower_bound=float(lower_bound-op.discard),
        setup_seconds=t1-t0, solve_seconds=t2-t1, audit_seconds=perf_counter()-t2,
        max_elements=max_elements, peak_product_bond=op.peak_product_bond,
        peak_product_core_elements=op.peak_product_core_elements, matvec_count=op.applications,
        coefficient_bond=bond(op.coefficient), rhs_work_bond=bond(g_work), generator_bond=bond(g),
        coefficient_discard=op.discard, rhs_discard=rhs_discard,
        compression_discard_bound=op.discard + rhs_discard, generator_diagnostics=gen_info,
        warm_start=initial is not None, multiplier_real=float(np.real(multiplier)),
        multiplier_imag=float(np.imag(multiplier)), galerkin_residual_abs=float(stationarity),
        galerkin_residual_rel=_ratio(stationarity, exact_target), projected_residual_abs=float(raw_projected),
        projected_residual_rel=_ratio(raw_projected, np.linalg.norm(b_exact)),
        physical_residual_abs=physical, physical_residual_rel=_ratio(physical, g_norm),
        orthogonal_residual_abs=orthogonal,
        generator_norm=g_norm, root_hermiticity_defect=xdef, generator_hermiticity_defect=gdef,
        hermiticity_defect_before=h_before, hermiticity_defect=h_after, adjoint_tangent_leakage=adj_leak,
        normalization_velocity_defect=float(abs(np.vdot(c, v))), operator_audit=probes)
    failure = None
    if not all(np.isfinite(val) for val in (h_after, physical, stationarity, raw_projected, orthogonal)):
        failure = 'nonfinite_residual'
    elif h_after > hermitian_tol*max(tn.norm(velocity), np.finfo(float).tiny):
        failure = 'hermiticity_failure'
    elif stationarity > audit_tol * (1+1e-6):
        failure = 'nonconvergence'
    elif physical_rtol is not None and physical > max(atol, physical_rtol*g_norm):
        failure = 'physical_residual_failure'
    if failure:
        diagnostics.pop('status')
        raise GalerkinError(failure, 'Final audit on original unrounded equation failed', **diagnostics)
    return velocity, _evidence(diagnostics)


def dense_oracle(root, g_matrix, *, mode='one', constrained=True, max_entries=2_000_000):
    """Assemble independent small-system A/B/C objectives in physical matrix order.

    B is the design isometry with columns vec(B_mu) in row-major operator
    order. The direct KKT solve is indefinite. A and C use their own metrics
    subject to the same optional normalization row as B.
    """
    frame = build_frame(root, mode)
    d, D = 2**root.n, frame.dimension
    if max(d*d*D, D*D) > max_entries:
        raise GalerkinError('memory_budget', 'Explicit oracle exceeds entry budget',
                            dimension=D, hilbert_dimension=d, max_entries=max_entries)
    x = tn.to_matrix(root.dense(), root.n)
    g = np.asarray(g_matrix, dtype=complex)
    if g.shape != x.shape:
        raise ValueError('Dense target shape differs from root')
    eigenvalues = np.linalg.eigvalsh(x)
    if eigenvalues[0] <= 0 or not np.allclose(x, x.conj().T, atol=1e-13, rtol=1e-12):
        raise GalerkinError('rank_support_uncertain', 'Dense oracle requires a positive Hermitian root')
    columns = []
    applied = []
    for j in range(D):
        e = np.zeros(D, dtype=complex)
        e[j] = 1
        y = tn.to_matrix(frame.decode(e).dense(), root.n)
        columns.append(y.ravel())
        applied.append((x@y + y@x).ravel())
    B = np.column_stack(columns)
    SB = np.column_stack(applied)
    M = B.conj().T @ SB
    b = B.conj().T @ g.ravel()
    c = B.conj().T @ x.ravel()
    exact = solve_sylvester(x, x, g)

    def direct(h, rhs):
        if constrained:
            kkt = np.block([[h, c[:, None]], [c.conj()[None, :], np.zeros((1, 1))]])
            return np.linalg.solve(kkt, np.r_[rhs, 0.])[:-1]
        return np.linalg.solve(h, rhs)

    v = direct(M, b)
    a = direct(np.eye(D), B.conj().T @ exact.ravel())
    # Use SVD least squares on the constraint nullspace for the C oracle.
    if constrained:
        from scipy.linalg import null_space
        free = null_space(c.conj()[None, :])
        csolve = free @ np.linalg.lstsq(SB @ free, g.ravel(), rcond=None)[0]
    else:
        csolve = np.linalg.lstsq(SB, g.ravel(), rcond=None)[0]
    return dict(B=B, SB=SB, M=M, b=b, c=c, v=v, exact_force=exact,
                A=(B@a).reshape(d,d), B_velocity=(B@v).reshape(d,d), C=(B@csolve).reshape(d,d),
                root_spectrum=eigenvalues, metric_spectrum=np.linalg.eigvalsh(M))
