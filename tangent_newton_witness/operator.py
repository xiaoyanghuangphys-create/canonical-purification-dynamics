"""Current-witness projected Newton equations in real Hermitian MPO coordinates.

All products in the Krylov operator and its final audit are untruncated. QR
only changes tensor gauges. The rank tolerance affects the variation space,
never the factor powers in the equation. Memory limits count tensor entries
per allocation, separately from BLAS/LAPACK scratch and process RSS.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.sparse.linalg import LinearOperator, minres

from factor_tdvp.tangent_solver import mpo, tn
from galerkin_sylvester_tdvp.galerkin_solver import GalerkinError, _evidence, _pcg
from galerkin_sylvester_tdvp.tangent_frame import TangentFrame
from galerkin_sylvester_tdvp.witness_anchor_preconditioner import encode_action
from galerkin_sylvester_tdvp.witness_stage import _pauli_lower

PAULI = np.stack((mpo.I, mpo.X, np.array([[0, -1j], [1j, 0]]), mpo.Z)).reshape(4, 4)/np.sqrt(2)
DEFAULT_LIMIT = 2_000_000


def _train(cores):
    result = tn.MPS(len(cores), max_bond=max(c.shape[2] for c in cores))
    result.tensors, result.center = cores, result.n-1
    return result


def _basis(value, transform):
    return _train([np.einsum('ps,lsr->lpr', transform, c) for c in value.tensors])


def _real_train(pauli):
    """Real part of a whole complex train, allowing an arbitrary input gauge."""
    cores = []
    for j, c in enumerate(pauli.tensors):
        x, y = c.real, c.imag
        if pauli.n == 1:
            out = x
        elif j == 0:
            out = np.concatenate((x, y), axis=2)
        elif j == pauli.n - 1:
            out = np.concatenate((x, -y), axis=0)
        else:
            out = np.concatenate((np.concatenate((x, y), axis=2),
                                  np.concatenate((-y, x), axis=2)), axis=0)
        cores.append(out)
    return _train(cores)


class HermitianFrame:
    """A real isometry. Local Pauli strings are an orthonormal Hermitian basis."""
    def __init__(self, factor, mode='two', rank_rtol=1e-13):
        skew = tn.distance(factor, mpo.adjoint(factor))
        factor_norm = tn.norm(factor)
        if not np.isfinite(skew) or skew > 2e-12*factor_norm:
            raise GalerkinError('hermiticity_failure', 'Current factor has a non-Hermitian component',
                                factor_hermiticity=skew, factor_norm=factor_norm)
        if factor_norm == 0:
            raise GalerkinError('support_uncertain', 'Zero factor has no active regular tangent')
        pauli = _basis(factor, PAULI.conj())
        self.real = TangentFrame(_real_train(pauli), mode=mode, rank_rtol=rank_rtol)
        self.dimension = self.real.dimension
        self.ranks = self.real.ranks
        self.metadata = dict(self.real.metadata, input_ranks=[1, *(c.shape[2] for c in factor.tensors)],
                             factor_hermiticity=skew, factor_norm=factor_norm,
                             coordinates='real_orthonormal_pauli')

    def decode(self, coords):
        if np.iscomplexobj(coords):
            raise ValueError('Hermitian coordinates must be real')
        return _basis(self.real.decode(coords), PAULI.T)

    def encode(self, value):
        return self.real.encode(_basis(value, PAULI.conj())).real


class _Budget:
    def __init__(self, limit):
        if int(limit) != limit or limit < 1:
            raise ValueError('max_elements must be a positive integer')
        self.limit, self.peak, self.bond = int(limit), 0, 1

    def check(self, size, name):
        self.peak = max(self.peak, int(size))
        if size > self.limit:
            raise GalerkinError('memory_budget', name+' exceeds per-array element budget',
                                requested_elements=int(size), max_elements=self.limit)

    def product(self, a, b):
        # Forward QR: cap the left product rank by its exact geometric
        # dimension before expanding the next tensor. No SVD cutoff.
        if a.n != b.n:
            raise ValueError('Incompatible chain sizes')
        carry = np.ones((1, 1, 1), complex)
        cores = []
        for ac, bc in zip(a.tensors, b.tensors):
            al, _, ar = ac.shape
            bl, _, br = bc.shape
            dl = carry.shape[0]
            self.check(4*dl*max(bl*ar, ar*br), 'Product QR panel')
            left, right = ac.reshape(al, 2, 2, ar), bc.reshape(bl, 2, 2, br)
            tmp = np.tensordot(carry, left, axes=(1, 0))
            panel = np.tensordot(tmp, right, axes=((1, 3), (0, 1)))
            panel = panel.transpose(0, 1, 3, 2, 4).reshape(4*dl, ar*br)
            self.bond = max(self.bond, ar*br)
            if len(cores) == a.n-1:
                cores.append(panel.reshape(dl, 4, 1))
            else:
                q, r = np.linalg.qr(panel, mode='reduced')
                cores.append(q.reshape(dl, 4, q.shape[1]))
                carry = r.reshape(-1, ar, br)
        result = _train(cores)
        result.move_center(0)
        return result

    def add(self, a, b, sign):
        for x, y in zip(a.tensors, b.tensors):
            self.check(4*(x.shape[0]+y.shape[0])*(x.shape[2]+y.shape[2]), 'Sum core')
        value = tn.add(a, b, beta=sign)
        # Stable cancellation before restricting to coordinates. QR on both
        # sides also removes bonds larger than the exact geometric ranks.
        value.center = value.n-1
        value.move_center(0)
        value.move_center(value.n-1)
        return value


def factor_power(factor, power, max_elements=DEFAULT_LIMIT):
    """Exact MPO product up to floating point. Intended for a Hermitian factor."""
    if power not in (1, 2, 3, 4):
        raise ValueError('Supported powers are 1 through 4')
    budget = _Budget(max_elements)
    if power == 1:
        return factor
    square = budget.product(factor, factor)
    if power == 2:
        return square
    return budget.product(square, factor if power == 3 else square)


def residual_norm(factor, target, power, max_elements=DEFAULT_LIMIT):
    budget = _Budget(max_elements)
    return tn.norm(budget.add(target, factor_power(factor, power, max_elements), -1.))


def _sandwich(left, right, budget):
    """Local Pauli-coordinate superoperator for V -> left V right."""
    result = []
    identity = np.eye(2).reshape(1, 4, 1)
    for j, a in enumerate(left.tensors):
        b = identity if right is None else right.tensors[j]
        al, _, ar = a.shape
        bl, _, br = b.shape
        budget.check(16*al*bl*ar*br, 'Fixed superoperator core')
        u = np.einsum('aikb,cljd->acijklbd', a.reshape(al, 2, 2, ar),
                      b.reshape(bl, 2, 2, br)).reshape(al*bl, 4, 4, ar*br)
        u = np.einsum('ps,lsur->lpur', PAULI.conj(), u, optimize=True)
        result.append(np.einsum('lpur,qu->lpqr', u, PAULI, optimize=True))
    return result


class NewtonOperator:
    """Fixed frame and exact coefficient powers; no projections between products."""
    def __init__(self, factor, power, mode='two', rank_rtol=1e-13, max_elements=DEFAULT_LIMIT):
        if power not in (2, 4):
            raise ValueError('power must be 2 or 4')
        self.power = power
        self.budget = _Budget(max_elements)
        for c in factor.tensors:
            self.budget.check(max(4*c.size, (8*c.shape[0])**2), 'Real frame canonicalization')
        tick = perf_counter()
        self.frame = HermitianFrame(factor, mode, rank_rtol)
        self.frame_seconds = perf_counter()-tick
        tick = perf_counter()
        self.factor = factor
        self.square = self.budget.product(factor, factor)
        self.pairs = [(factor, None)]
        if power == 4:
            cube = self.budget.product(self.square, factor)
            self.pairs = [(cube, None), (self.square, factor)]
        self.physical = self.square if power == 2 else self.budget.product(self.square, self.square)
        self.superops = [_sandwich(left, right, self.budget) for left, right in self.pairs]
        # In encode_action the open-right intermediate can join an input-left
        # and output-right virtual leg. Check both this contraction and the
        # corresponding prefix panel before any Krylov allocations.
        f = self.frame.real
        for chain in self.superops:
            for j, core in enumerate(chain):
                kl, _, _, kr = core.shape
                dl, dr = f._cuts[j][-1].stop, f._cuts[j+1][-1].stop
                self.budget.check(4*max(f.ranks[j], f.ranks[j+1])*
                                  max(kl, kr)*max(dl, dr), 'Encoded derivative panel')
        self.setup_seconds = perf_counter()-tick
        self.applications, self.apply_seconds = 0, 0.

    def matvec(self, v):
        if np.iscomplexobj(v):
            raise ValueError('Hermitian coordinates must be real')
        t = perf_counter()
        value = self.frame.real.decode(v)
        # Tr(B_i L K R) plus its Hermitian conjugate for real coordinates.
        # Hermiticity of B_i,K,L,R implies these coefficients are conjugates.
        coords = sum(2*encode_action(self.frame.real, ops, value).real for ops in self.superops)
        self.applications += 1
        self.apply_seconds += perf_counter()-t
        return coords

    def diagonal(self):
        """All B_i^* D B_i using common canonical left/right environments."""
        f = self.frame.real
        out = np.zeros(f.dimension)
        for ops in self.superops:
            right = [None]*(f.n+1)
            right[-1] = np.ones((1, 1, 1))
            for j in range(f.n-1, f.width-1, -1):
                a = f.right.tensors[j]
                right[j] = np.einsum('isx,astb,jty,xby->iaj', a.conj(), ops[j],
                                     a, right[j+1], optimize=True)
            prefix = np.ones((1, 1, 1))
            for j, q in enumerate(f.complements):
                comp = q.reshape(f.ranks[j], 4, -1)
                left = np.einsum('isx,iaj,astb,jtx->xb', comp.conj(), prefix, ops[j],
                                 comp, optimize=True)
                suffix = np.einsum('kbk->bk', right[j+f.width])
                if f.width == 2:
                    suffix = np.einsum('assb,bk->ask', ops[j+1], suffix).reshape(ops[j+1].shape[0], -1)
                out[f.offsets[j]:f.offsets[j+1]] += 2*(left @ suffix).real.ravel()
                a = f.left.tensors[j]
                prefix = np.einsum('isx,iaj,astb,jty->xby', a.conj(), prefix, ops[j], a, optimize=True)
        return out

    def derivative(self, direction):
        """Independent ambient MPO action for recomputing the linear residual."""
        value = None
        for left, right in self.pairs:
            term = self.budget.product(left, direction)
            if right is not None:
                term = self.budget.product(term, right)
            term = self.budget.add(term, mpo.adjoint(term), 1.)
            value = term if value is None else self.budget.add(value, term, 1.)
        return value


def _factor_floor(factor, frame):
    """Pauli triangle bound on W, charging the actual frame trimming error."""
    # _pauli_lower expects coefficients in an unnormalized Pauli expansion.
    # A final scalar 2^(-N/2) converts orthonormal Pauli-string coefficients.
    info = _pauli_lower(frame.real.left)
    if info is None:
        return None
    represented = (info['identity_coefficient']-info['nonidentity_pauli_majorant'])*2.**(-factor.n/2)
    return represented-frame.metadata['rank_discard_bound']-.5*frame.metadata['factor_hermiticity']


def solve_direction(factor, target, power, *, mode='two', lower_bound=None, rtol=1e-8,
                    atol=0., maxiter=80, max_elements=DEFAULT_LIMIT, rank_rtol=1e-13,
                    preconditioner='none'):
    """Return (Hermitian K, diagnostics); reject unresolved original equations.

    A supplied positive lower_bound must apply to the actual current W.
    With no bound the solver first tries a current-factor Pauli enclosure.
    The indefinite MINRES alternative has the same Galerkin objective and
    must pass the original residual check. Endpoint merit is a separate test.
    """
    if (factor.n != target.n or not np.isfinite([rtol, atol]).all()
            or min(rtol, atol) < 0 or not (rtol or atol) or maxiter < 1):
        raise ValueError('Invalid target size, Krylov tolerance or iteration cap')
    if preconditioner not in ('none', 'jacobi'):
        raise ValueError('Choose none or jacobi')
    op = NewtonOperator(factor, power, mode, rank_rtol, max_elements)
    frame, budget = op.frame, op.budget
    t = perf_counter()
    source = budget.add(target, op.physical, -1.)
    target_norm = tn.norm(target)
    target_skew = tn.distance(target, mpo.adjoint(target))
    if not np.isfinite(target_skew) or target_skew > 2e-12*target_norm:
        raise GalerkinError('hermiticity_failure', 'Newton target is not Hermitian', target_hermiticity=target_skew)
    norm_source = tn.norm(source)
    b = frame.encode(source)
    norm_b = float(np.linalg.norm(b))
    source_seconds = perf_counter()-t
    t = perf_counter()
    bound_source = 'caller_current_witness' if lower_bound is not None else 'current_pauli_triangle'
    floor = lower_bound if lower_bound is not None else _factor_floor(factor, frame)
    if lower_bound is not None and (not np.isfinite(lower_bound) or lower_bound <= 0
            or lower_bound > mpo.trace(factor).real*2.**(-factor.n)*(1+1e-10)):
        raise GalerkinError('invalid_floor', 'Supplied current-factor floor is invalid', lower_bound=lower_bound)
    # A triangle bound at cancellation scale cannot distinguish a singular
    # factor from a positive one in floating point. Treat this case as unknown.
    enclosure_margin = (64*np.finfo(float).eps*factor.n*frame.metadata['factor_norm']
                        if lower_bound is None else 0.)
    spd = floor is not None and np.isfinite(floor) and floor > enclosure_margin
    scale = power*(frame.metadata['factor_norm']*2.**(-factor.n/2))**(power-1)
    if not np.isfinite(scale) or scale <= 0:
        raise GalerkinError('numerical_range', 'Derivative scaling is outside floating point range')
    support_seconds = perf_counter()-t
    t = perf_counter()
    pre = lambda v: v
    if preconditioner == 'jacobi':
        if not spd:
            raise GalerkinError('preconditioner_failure', 'Jacobi requires a verified positive current factor')
        diagonal = op.diagonal()/scale
        if not np.isfinite(diagonal).all() or min(diagonal) <= 0:
            raise GalerkinError('preconditioner_failure', 'Current-factor diagonal lost positivity')
        pre = lambda v: v/diagonal
    precond_seconds = perf_counter()-t
    apply = lambda v: op.matvec(v)/scale
    solver = 'cg' if spd else 'minres'
    limit = max(atol, rtol*norm_b)
    t = perf_counter()
    if norm_b <= limit:
        v = np.zeros_like(b)
        history, cond, iterations = [0. if norm_b == 0 else 1.], None, 0
    elif spd:
        v, history, cond = _pcg(apply, pre, np.zeros_like(b), b/scale, limit/scale, maxiter)
        iterations = len(history)-1
    else:
        operator = LinearOperator((len(b), len(b)), matvec=apply, dtype=float)
        steps = []
        # MINRES' built-in stopping rule uses backward error; audit the
        # original stationarity separately below before releasing a direction.
        v, info = minres(operator, b/scale, rtol=max(np.finfo(float).eps, .01*limit/norm_b),
                         maxiter=maxiter, callback=lambda unused: steps.append(1))
        iterations, history, cond = len(steps), None, None
        if info:
            raise GalerkinError('nonconvergence', 'MINRES did not reach its stopping criterion',
                                iterations=iterations, info=info, factor_lower_bound=floor)
    solve_seconds = perf_counter()-t
    t = perf_counter()
    direction = frame.decode(v)
    ambient = budget.add(op.derivative(direction), source, -1.)
    projected = frame.encode(ambient)
    stationarity = float(np.linalg.norm(projected))
    full = tn.norm(ambient)
    outside = tn.norm(budget.add(ambient, frame.decode(projected), -1.))
    diag = dict(power=power, mode=mode, solver=solver, factor_lower_bound=floor,
                floor_provenance=bound_source, enclosure_margin=enclosure_margin, verified_spd=bool(spd),
                target_hermiticity=target_skew, scale=scale,
                preconditioner=preconditioner, regularization_shift=0., iterations=iterations,
                history=history, condition_estimate=cond, frame_metadata=frame.metadata,
                projected_rhs_norm=norm_b, rhs_norm=norm_source, projected_residual=stationarity,
                full_linear_residual=full, outside_tangent_residual=outside,
                linear_tolerance=limit, hermiticity_defect=tn.distance(direction, mpo.adjoint(direction)),
                dimension=frame.dimension, max_elements=max_elements, peak_guarded_elements=budget.peak,
                peak_product_bond=budget.bond, factor_power_bond=op.physical.bond_dimension(),
                direction_bond=direction.bond_dimension(), matvec_count=op.applications,
                frame_seconds=op.frame_seconds, operator_setup_seconds=op.setup_seconds,
                source_seconds=source_seconds, support_seconds=support_seconds,
                preconditioner_setup_seconds=precond_seconds, solve_seconds=solve_seconds,
                matvec_seconds_inclusive_in_solve=op.apply_seconds, audit_seconds=perf_counter()-t)
    if (not np.isfinite([stationarity, full, outside]).all()
            or stationarity > (1+1e-5)*limit):
        raise GalerkinError('nonconvergence', 'Original projected residual failed (possible unresolved null forcing)',
                            **diag)
    return direction, _evidence(diag)
