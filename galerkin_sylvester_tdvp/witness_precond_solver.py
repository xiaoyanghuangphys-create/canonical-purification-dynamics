"""Cold-start tangent Sylvester PCG with the positive-witness preparation anchor."""
from __future__ import annotations

import copy
from time import perf_counter

import numpy as np

from . import galerkin_solver as gs
from .models import mpo, tn
from .tangent_frame import build_frame
from .witness_anchor_preconditioner import encode_action, preparation_preconditioner
from .witness_streaming import ProductCores, _sum_norms


def product_operators(x):
    """Fixed superoperator cores for V -> X V and V -> V X separately."""
    left, right = [], []
    eye = np.eye(2)
    for core in x.tensors:
        dl, _, dr = core.shape
        a = core.reshape(dl, 2, 2, dr)
        left.append(np.einsum('aikb,jl->aijklb', a, eye).reshape(dl, 4, 4, dr))
        right.append(np.einsum('aljb,ik->aijklb', a, eye).reshape(dl, 4, 4, dr))
    return left, right


def projected_sylvester(frame, operators, velocity):
    left, right = operators
    return encode_action(frame, left, velocity) + encode_action(frame, right, velocity)


def _canonical_right(value):
    out = copy.deepcopy(value)
    out.center = out.n - 1
    out.move_center(0)
    return out


def bounded_product(a, b, absolute_budget, max_elements, max_rank=192):
    """Round a product with a verified absolute Frobenius error allowance.

    Both operands are right canonical. Past every cut, each suffix indexed by
    (r,s) is a product A_r B_s, hence has Frobenius norm at most one. The
    suffix map has norm at most sqrt(number of such pairs). Multiplying each
    discarded prefix Frobenius norm by this bound and summing charges every
    SVD truncation. Retained prefix factors are isometries. Product cores
    themselves are never formed. The rank guard can reject an unverified fit.
    """
    if a.n != b.n or absolute_budget <= 0 or max_rank < 1:
        raise ValueError('Invalid product approximation budget or dimensions')
    a, b = _canonical_right(a), _canonical_right(b)
    carry = np.ones((1, 1, 1), complex)
    tensors, allowance = [], 0.
    for j, (ac, bc) in enumerate(zip(a.tensors, b.tensors)):
        ar, br = ac.shape[-1], bc.shape[-1]
        needed = 4*carry.shape[0]*max(ar*br, ar*bc.shape[0])
        if needed > max_elements:
            raise gs.GalerkinError('memory_budget', 'Bounded product panel exceeds budget',
                                   requested_core_elements=needed, max_elements=max_elements)
        left = ac.reshape(ac.shape[0], 2, 2, ar)
        right = bc.reshape(bc.shape[0], 2, 2, br)
        partial = np.tensordot(carry, left, axes=(1, 0))  # prefix,b_left,i,k,a_right
        local = np.tensordot(partial, right, axes=((1, 3), (0, 1)))
        local = local.transpose(0, 1, 3, 2, 4).reshape(4*carry.shape[0], ar*br)
        if j == a.n-1:
            tensors.append(local.reshape(carry.shape[0], 4, 1))
            break
        u, sig, vh = np.linalg.svd(local, full_matrices=False)
        tail = np.sqrt(np.cumsum(sig[::-1]**2))
        local_budget = (absolute_budget-allowance)/(a.n-1-j)
        discarded = int(np.searchsorted(tail, local_budget/np.sqrt(ar*br), side='right'))
        rank = max(1, len(sig)-discarded)
        if rank > max_rank:
            raise gs.GalerkinError('memory_budget', 'Required bounded product rank exceeds audit budget',
                                   required_rank=rank, max_rank=max_rank, site=j,
                                   absolute_budget=absolute_budget)
        allowance += np.linalg.norm(sig[rank:])*np.sqrt(ar*br)
        tensors.append(u[:, :rank].reshape(carry.shape[0], 4, rank))
        carry = (sig[:rank, None]*vh[:rank]).reshape(rank, ar, br)
    result = tn.MPS(a.n, max_bond=max(c.shape[-1] for c in tensors))
    result.tensors, result.center = tensors, a.n-1
    return result, float(allowance)


def _audit_norms(root, velocity, g, projected, frame, g_norm, rtol, max_elements):
    """Use lossless QR whenever its panels fit, otherwise certified product bounds."""
    projected_state = frame.decode(projected)
    product_dims = [a.shape[-1]*b.shape[-1] for a,b in zip(root.tensors, velocity.tensors)]
    previous = 1
    exact_panels = True
    for j, dr in enumerate(product_dims):
        dl = 1 if j == 0 else product_dims[j-1]
        width = 2*dr + g.tensors[j].shape[-1] + projected_state.tensors[j].shape[-1]
        if 4*dl*dr > max_elements or 4*previous*width > 6_000_000:
            exact_panels = False
            break
        previous = min(4*previous, width)
    if exact_panels:
        left = ProductCores(root, velocity, max_elements)
        right = ProductCores(velocity, root, max_elements)
        radius, method = 0., 'exact_streamed_qr'
    else:
        # The projected stationarity check always uses the full product. Only
        # the two reported physical diagnostic norms share this error radius.
        per_product_budget = max(1e-12, min(rtol*.01, 1e-6))*g_norm/2
        left, e_left = bounded_product(root, velocity, per_product_budget, max_elements)
        right, e_right = bounded_product(velocity, root, per_product_budget, max_elements)
        radius, method = e_left+e_right, 'bounded_products_qr'
    physical, outside = _sum_norms([(1.,left),(1.,right),(-1.,g),(-1.,projected_state)],
                                    root.n, [range(3), range(4)])
    return physical, outside, radius, method


def solve_velocity(root, generator, *, mode='two', lower_bound, constrained=True,
                   rtol=1e-9, atol=1e-13, maxiter=80, work_bond=None,
                   max_elements=2_000_000, anchor=None, preconditioner='preparation'):
    """Solve the scaled normalized Galerkin equation and audit the original source."""
    if not constrained or mode not in ('one','two'):
        raise ValueError('The normalized tangent and a one-/two-site frame are required')
    if not np.isfinite([rtol,atol]).all() or min(rtol,atol)<0 or not (rtol or atol) or maxiter<0:
        raise ValueError('Invalid Krylov tolerance or cap')
    if preconditioner not in ('preparation','none') or (preconditioner=='preparation' and anchor is None):
        raise ValueError('A preparation MPO is required for preparation preconditioning')
    start = perf_counter()
    herm_tol = 2e-9
    x_norm = tn.norm(root)
    x_skew = tn.distance(root, mpo.adjoint(root))
    if x_skew > herm_tol*x_norm:
        raise gs.GalerkinError('hermiticity_failure','Non-Hermitian current root',root_hermiticity_defect=x_skew)
    scale = 2*mpo.trace(root).real*2.**(-root.n)
    if lower_bound is None or not np.isfinite([scale,lower_bound]).all() or lower_bound<=0 or scale*.5<lower_bound*(1-1e-10):
        raise gs.GalerkinError('rank_support_uncertain','Invalid current spectral floor',spectral_lower_bound=lower_bound)
    frame = build_frame(root, mode)
    op = gs.GalerkinOperator(root,frame,work_bond=work_bond,max_elements=max_elements)
    if op.discard>=lower_bound or op.discard>1e-9*x_norm:
        raise gs.GalerkinError('compression_overflow','Fixed coefficient rounding exceeds budget',coefficient_discard=op.discard)
    generated = generator(op.product(root,root))
    g, g_info = generated if isinstance(generated, tuple) else (generated,{})
    if g.n!=root.n or not all(np.isfinite(x).all() for x in g.tensors):
        raise gs.GalerkinError('nonfinite_generator','Invalid current-density generator output')
    g = gs._geometric_reduce(g)
    g_norm = tn.norm(g)
    g_skew = tn.distance(g,mpo.adjoint(g))
    if g_skew>max(atol,herm_tol*g_norm):
        raise gs.GalerkinError('hermiticity_failure','Generator is not Hermitian',generator_hermiticity_defect=g_skew)
    g_fixed, rhs_discard = gs._fixed_round(g, work_bond)
    if rhs_discard>max(atol,1e-9*g_norm):
        raise gs.GalerkinError('compression_overflow','Fixed source rounding exceeds budget',rhs_discard=rhs_discard)
    c = frame.encode(root)
    normal = c/np.linalg.norm(c)
    def null(v):
        return v-normal*np.vdot(normal,v)
    target = null(frame.encode(g_fixed))
    limit = max(atol,rtol*np.linalg.norm(target))
    fixed_operators = product_operators(op.coefficient)
    def matvec(v):
        op.applications += 1
        velocity = frame.decode(null(v))
        return null(projected_sylvester(frame, fixed_operators, velocity))/scale

    begin_pre = perf_counter()
    inverse = preparation_preconditioner(frame, anchor, normal, scale) if preconditioner=='preparation' else None
    pre_setup = perf_counter()-begin_pre
    pre_calls, pre_time = 0, 0.
    def pre(v):
        nonlocal pre_calls, pre_time
        if inverse is None:
            return v
        before = perf_counter()
        out = inverse(v)
        pre_calls += 1
        pre_time += perf_counter()-before
        return out
    tick = perf_counter()
    coef, history, condition = gs._pcg(matvec,pre,np.zeros_like(target),target/scale,limit/scale,maxiter)
    tock = perf_counter()
    before = gs._geometric_reduce(frame.decode(coef))
    adj = mpo.adjoint(before)
    adj_coords = frame.encode(adj)
    leakage = tn.distance(adj,gs._geometric_reduce(frame.decode(adj_coords)))
    if leakage>herm_tol*max(tn.norm(before),np.finfo(float).tiny):
        raise gs.GalerkinError('hermiticity_failure','Adjoint escaped active tangent',adjoint_tangent_leakage=leakage)
    coef = null((coef+adj_coords)/2)
    velocity = gs._geometric_reduce(frame.decode(coef))
    h_after = tn.distance(velocity,mpo.adjoint(velocity))
    if h_after>herm_tol*max(tn.norm(velocity),np.finfo(float).tiny):
        raise gs.GalerkinError('hermiticity_failure','Postsolve velocity is not Hermitian',hermiticity_defect=h_after)
    exact_b = frame.encode(g)
    exact_ops = fixed_operators if op.coefficient is root else product_operators(root)
    projected = projected_sylvester(frame,exact_ops,velocity)-exact_b
    stationarity = float(np.linalg.norm(null(projected)))
    exact_target = np.linalg.norm(null(exact_b))
    raw = float(np.linalg.norm(projected))
    physical, orthogonal, radius, audit_method = _audit_norms(root,velocity,g,projected,frame,g_norm,rtol,max_elements)
    diag = dict(status='success', backend='factored_sylvester_pcg',mode=mode,constrained=True,
                iterations=len(history)-1,history=history,condition_estimate=condition,
                dimension=frame.dimension,state_bonds=list(frame.ranks),work_bond=work_bond,
                frame_metadata=frame.metadata,preconditioner=preconditioner,warm_start=False,
                coefficient_bond=gs.bond(op.coefficient),rhs_work_bond=gs.bond(g_fixed),generator_bond=gs.bond(g),
                coefficient_hermitian_construction=('adjoint_fixed_cores' if op.manifest_hermitian else 'explicit_hermitian_part'),
                setup_seconds=tick-start,solve_seconds=tock-tick,audit_seconds=perf_counter()-tock,
                preconditioner_setup_seconds=pre_setup,preconditioner_apply_seconds=pre_time,
                preconditioner_calls=pre_calls,rtol=rtol,atol=atol,
                galerkin_residual_abs=stationarity,galerkin_residual_rel=gs._ratio(stationarity,exact_target),
                projected_residual_abs=raw,projected_residual_rel=gs._ratio(raw,np.linalg.norm(exact_b)),
                physical_residual_abs=physical,physical_residual_rel=gs._ratio(physical,g_norm),
                orthogonal_residual_abs=orthogonal,residual_audit_method=audit_method,residual_norm_radius=radius,
                physical_residual_interval=[max(0.,physical-radius),physical+radius],
                orthogonal_residual_interval=[max(0.,orthogonal-radius),orthogonal+radius],
                adjoint_tangent_leakage=leakage,hermiticity_defect=h_after,
                normalization_velocity_defect=abs(np.vdot(c,coef)),spectral_lower_bound=lower_bound,
                effective_lower_bound=lower_bound-op.discard,root_hermiticity_defect=x_skew,
                generator_hermiticity_defect=g_skew,generator_norm=g_norm,generator_diagnostics=g_info,
                coefficient_discard=op.discard,rhs_discard=rhs_discard,
                compression_discard_bound=op.discard+rhs_discard,max_elements=max_elements,
                peak_product_bond=op.peak_product_bond,peak_product_core_elements=op.peak_product_core_elements,
                matvec_count=op.applications)
    if not np.isfinite([physical,orthogonal,stationarity,raw,radius]).all() or stationarity>max(atol,rtol*exact_target)*(1+1e-6):
        raise gs.GalerkinError('nonconvergence','Original Galerkin equation failed its residual audit',diagnostics=diag)
    return velocity,gs._evidence(diag)
