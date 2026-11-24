"""Bounded small-chain fallback: real-linear damped LSQR in the MPO tangent.

The ambient residual is dense in this optional backend. No global tangent
columns, normal matrix, target roots or inverse density powers are formed.
"""
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr

from factor_tdvp.tangent_solver import matrix, tn
from galerkin_sylvester_tdvp.galerkin_solver import GalerkinError
from galerkin_sylvester_tdvp.tangent_frame import build_frame


def jacobian(w, k, power, x=None):
    dx = k @ w.conj().T + w @ k.conj().T
    if power == 2:
        return dx
    if power != 4:
        raise ValueError("power must be 2 or 4")
    x = w @ w.conj().T if x is None else x
    return x @ dx + dx @ x


def jacobian_adjoint(w, residual, power, x=None):
    rh = (residual + residual.conj().T) / 2
    if power == 4:
        x = w @ w.conj().T if x is None else x
        rh = x @ rh + rh @ x
    elif power != 2:
        raise ValueError("power must be 2 or 4")
    return 2 * rh @ w


def pack(z):
    return np.concatenate((z.real.ravel(), z.imag.ravel()))


def unpack(v):
    re, im = np.split(v, 2)
    return re + 1j * im


def _untruncated_mpo(value, n):
    """Represent a small dense cotangent with all geometric QR columns.

    LSQR needs the adjoint of the same linear map at every iteration. Even
    a relative machine-scale TT-SVD cutoff can erase a small Schmidt sector
    of the cotangent when an unrelated sector becomes large.
    """
    block = tn.to_interleaved(value, n)
    left = 1
    cores = []
    for _ in range(n-1):
        basis, block = np.linalg.qr(block.reshape(4*left, -1), mode='reduced')
        right = basis.shape[1]
        cores.append(basis.reshape(left, 4, right))
        left = right
    cores.append(block.reshape(left, 4, 1))
    out = tn.MPS(n, max_bond=max(core.shape[-1] for core in cores))
    out.tensors, out.center = cores, n-1
    return out


def solve_direction(factor, target, power, *, mode="two", damping=1.e-8,
                    maxiter=100, max_elements=2_000_000, rtol=1.e-8):
    """Zero-start LSQR; return only independently verified descent directions."""
    start = perf_counter()
    if power not in (2, 4) or factor.n != target.n:
        raise ValueError("inconsistent power or chain length")
    if not np.isfinite(damping) or damping < 0 or not 0 < rtol < 1 or maxiter < 1:
        raise ValueError("invalid LSQR limits")
    n = factor.n
    if n > 6 or 12 * 4**n > max_elements:
        raise GalerkinError("gn_dense_budget", "Small-chain LSQR fallback exceeds cap", n=n)
    frame = build_frame(factor, mode)
    w, t = matrix(factor), matrix(target)
    x = w @ w.conj().T
    rhs = t - (x if power == 2 else x @ x)
    dimensions = (2 * w.size, 2 * frame.dimension)

    def apply(c):
        k = matrix(frame.decode(unpack(c)))
        return pack(jacobian(w, k, power, x))

    def adjoint(u):
        residual = unpack(u).reshape(w.shape)
        jstar = jacobian_adjoint(w, residual, power, x)
        return pack(frame.encode(_untruncated_mpo(jstar, n)))

    design = LinearOperator(dimensions, matvec=apply, rmatvec=adjoint, dtype=float)
    b = pack(rhs)
    setup = perf_counter() - start
    answer = lsqr(design, b, damp=damping, atol=rtol * .05, btol=rtol * .05,
                  conlim=1 / np.finfo(float).eps, iter_lim=maxiter)
    c, stop, iterations = answer[:3]
    residual = apply(c) - b
    grad = adjoint(residual) + damping**2 * c
    rhs_grad = np.linalg.norm(adjoint(b))
    normal = np.linalg.norm(grad)
    threshold = rtol * rhs_grad + 128*np.finfo(float).eps * np.linalg.norm(b) * max(answer[5], 1.)
    before, after = np.linalg.norm(b), np.linalg.norm(residual)
    info = dict(backend="small_dense_codomain_real_lsqr", mode=mode, power=power,
                real_dimension=dimensions[1], ambient_real_dimension=dimensions[0],
                damping=damping, iterations=int(iterations), stop_code=int(stop),
                normal_residual=float(normal), normal_threshold=float(threshold),
                full_rhs_norm=float(before), full_linear_residual=float(after),
                condition_estimate=float(answer[6]), setup_seconds=setup,
                total_seconds=perf_counter()-start)
    if normal > threshold or not np.isfinite(normal):
        raise GalerkinError("gn_unresolved_stationarity", "LSQR failed original normal residual audit", **info)
    if before > 0 and (not np.isfinite(after) or after >= before):
        raise GalerkinError("gn_no_descent", "Unresolved forcing or no least-squares improvement", **info)
    return frame.decode(unpack(c)), info
