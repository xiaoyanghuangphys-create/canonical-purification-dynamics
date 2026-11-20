"""Lossless streamed products and residual norms for the positive-witness benchmark.

The two-site frame and nullspace PCG are identical to the original implementation.
Only the constrained, cold-start, unpreconditioned comparison setting is exposed.
"""
from __future__ import annotations

from functools import lru_cache
from time import perf_counter
import numpy as np

from . import galerkin_solver as gs
from .models import mpo, tn
from .tangent_frame import build_frame


class ProductCores:
    """MPO product with a two-core cache; the existing frame.encode reads by index."""
    def __init__(self, a, b, max_elements):
        if a.n != b.n:
            raise ValueError('MPO chain lengths differ')
        self.n = a.n
        self.right = [x.shape[-1]*y.shape[-1] for x,y in zip(a.tensors,b.tensors)]
        self.peak_core = max(4*l*r for l,r in zip([1]+self.right[:-1],self.right))
        if self.peak_core > max_elements:
            raise gs.GalerkinError('memory_budget','Streamed product core exceeds budget',
                                   max_elements=max_elements,requested_core_elements=self.peak_core)
        @lru_cache(maxsize=2)
        def core(j):
            x,y=a.tensors[j],b.tensors[j]
            dl,_,dr=x.shape
            el,_,er=y.shape
            prod=np.einsum('aikb,ckjd->acijbd',x.reshape(dl,2,2,dr),y.reshape(el,2,2,er),optimize=True)
            return prod.reshape(dl*el,4,dr*er)
        self.tensors = _IndexedCores(core)


class _IndexedCores:
    def __init__(self, core):
        self.core=core

    def __getitem__(self,j):
        return self.core(j)


def _sum_norms(terms, n, groups, max_elements=6_000_000):
    """Stable norms of several linear combinations using one common prefix QR.

    sectors[i] carries the uncontracted columns of summand i. At the last
    site every summand shares the same open boundary and is summed before
    taking the norm. The same prefix isometry is valid for any subset.
    """
    sectors = [np.array([[weight]], complex) for weight, _ in terms]
    for j in range(n):
        cores = [term.tensors[j] for _, term in terms]
        right_dims = [core.shape[-1] for core in cores]
        needed = 4 * sectors[0].shape[0] * sum(right_dims)
        if needed > max_elements:
            raise gs.GalerkinError('memory_budget', 'Residual QR panel exceeds budget',
                                   requested_core_elements=needed, max_elements=max_elements)
        blocks = [np.tensordot(sector, core, axes=(1, 0)).reshape(-1, dim)
                  for sector, core, dim in zip(sectors, cores, right_dims)]
        if j == n - 1:
            return [float(np.linalg.norm(sum(blocks[i] for i in group))) for group in groups]
        r = np.linalg.qr(np.concatenate(blocks, axis=1), mode='r')
        sectors = np.split(r, np.cumsum(right_dims)[:-1], axis=1)


def sum_norm(terms, n, max_elements=6_000_000):
    """Frobenius norm of a signed sum of MPOs without forming the global sum."""
    return _sum_norms(terms, n, [range(len(terms))], max_elements)[0]


def solve_velocity(root, generator, *, mode='two', lower_bound, constrained=True,
                   rtol=1e-9, atol=1e-13, maxiter=80, work_bond=None,
                   max_elements=2_000_000):
    """Same fixed-coefficient Galerkin objective and stopping tests, streamed audits."""
    if not constrained or mode not in ('one','two'):
        raise ValueError('Streamed benchmark solve requires the normalized root tangent')
    if not np.isfinite([rtol,atol]).all() or min(rtol,atol)<0 or not (rtol or atol) or maxiter<0:
        raise ValueError('Invalid Krylov tolerance or cap')
    start=perf_counter()
    herm_tol=2e-9
    x_norm=tn.norm(root)
    x_skew=tn.distance(root,mpo.adjoint(root))
    if x_skew>herm_tol*x_norm:
        raise gs.GalerkinError('hermiticity_failure','Non-Hermitian current root',root_hermiticity_defect=x_skew)
    scale=2*mpo.trace(root).real*2.**(-root.n)
    if lower_bound is None or not np.isfinite([scale,lower_bound]).all() or lower_bound<=0 or scale*.5<lower_bound*(1-1e-10):
        raise gs.GalerkinError('rank_support_uncertain','Invalid current spectral floor',spectral_lower_bound=lower_bound)
    frame=build_frame(root,mode)
    op=gs.GalerkinOperator(root,frame,work_bond=work_bond,max_elements=max_elements)
    if op.discard>=lower_bound or op.discard>1e-9*x_norm:
        raise gs.GalerkinError('compression_overflow','Fixed coefficient rounding exceeds budget',coefficient_discard=op.discard)
    generated=generator(op.product(root,root))
    g,g_info=generated if isinstance(generated,tuple) else (generated,{})
    if g.n!=root.n or not all(np.isfinite(x).all() for x in g.tensors):
        raise gs.GalerkinError('nonfinite_generator','Invalid current-density generator output')
    g=gs._geometric_reduce(g)
    g_norm=tn.norm(g)
    g_skew=tn.distance(g,mpo.adjoint(g))
    if g_skew>max(atol,herm_tol*g_norm):
        raise gs.GalerkinError('hermiticity_failure','Generator is not Hermitian',generator_hermiticity_defect=g_skew)
    g_fixed,rhs_discard=gs._fixed_round(g,work_bond)
    if rhs_discard>max(atol,1e-9*g_norm):
        raise gs.GalerkinError('compression_overflow','Fixed source rounding exceeds budget',rhs_discard=rhs_discard)
    c=frame.encode(root)
    normal=c/np.linalg.norm(c)
    def null(v):
        return v-normal*np.vdot(normal,v)
    target=null(frame.encode(g_fixed))
    limit=max(atol,rtol*np.linalg.norm(target))
    def terms(x,v):
        prods=(ProductCores(x,v,max_elements),ProductCores(v,x,max_elements))
        op.peak_product_bond=max(op.peak_product_bond,max(prods[0].right))
        op.peak_product_core_elements=max(op.peak_product_core_elements,prods[0].peak_core)
        return prods
    def matvec(p):
        op.applications+=1
        v=frame.decode(null(p))
        left,right=terms(op.coefficient,v)
        return null(frame.encode(left)+frame.encode(right))/scale
    tick=perf_counter()
    coef,history,condition=gs._pcg(matvec,lambda v:v,np.zeros_like(target),target/scale,limit/scale,maxiter)
    tock=perf_counter()
    before=gs._geometric_reduce(frame.decode(coef))
    adj=mpo.adjoint(before)
    adj_coords=frame.encode(adj)
    leakage=tn.distance(adj,gs._geometric_reduce(frame.decode(adj_coords)))
    if leakage>herm_tol*max(tn.norm(before),np.finfo(float).tiny):
        raise gs.GalerkinError('hermiticity_failure','Adjoint escaped active tangent',adjoint_tangent_leakage=leakage)
    coef=null((coef+adj_coords)/2)
    velocity=gs._geometric_reduce(frame.decode(coef))
    h_after=tn.distance(velocity,mpo.adjoint(velocity))
    if h_after>herm_tol*max(tn.norm(velocity),np.finfo(float).tiny):
        raise gs.GalerkinError('hermiticity_failure','Postsolve velocity is not Hermitian',hermiticity_defect=h_after)
    left,right=terms(root,velocity)
    exact_b=frame.encode(g)
    projected=frame.encode(left)+frame.encode(right)-exact_b
    stationarity=float(np.linalg.norm(null(projected)))
    exact_target=np.linalg.norm(null(exact_b))
    raw=float(np.linalg.norm(projected))
    residual=[(1.,left),(1.,right),(-1.,g)]
    physical, orthogonal = _sum_norms(residual + [(-1., frame.decode(projected))],
                                      root.n, [range(3), range(4)])
    diag=dict(status='success', backend='streamed_products_qr',mode=mode, constrained=True,
              iterations=len(history)-1, history=history, condition_estimate=condition,
              dimension=frame.dimension, state_bonds=list(frame.ranks), work_bond=work_bond,
              frame_metadata=frame.metadata, preconditioner='none', warm_start=False,
              coefficient_bond=gs.bond(op.coefficient),rhs_work_bond=gs.bond(g_fixed),generator_bond=gs.bond(g),
              coefficient_hermitian_construction=('adjoint_fixed_cores' if op.manifest_hermitian else 'explicit_hermitian_part'),
              setup_seconds=tick-start,solve_seconds=tock-tick,audit_seconds=perf_counter()-tock,
              galerkin_residual_abs=stationarity,galerkin_residual_rel=gs._ratio(stationarity,exact_target),
              projected_residual_abs=raw,projected_residual_rel=gs._ratio(raw,np.linalg.norm(exact_b)),
              physical_residual_abs=physical,physical_residual_rel=gs._ratio(physical,g_norm),
              orthogonal_residual_abs=orthogonal,adjoint_tangent_leakage=leakage,
              hermiticity_defect=h_after,normalization_velocity_defect=abs(np.vdot(c,coef)),
              spectral_lower_bound=lower_bound,effective_lower_bound=lower_bound-op.discard,
              root_hermiticity_defect=x_skew,generator_hermiticity_defect=g_skew,generator_norm=g_norm,
              generator_diagnostics=g_info,coefficient_discard=op.discard,rhs_discard=rhs_discard,
              compression_discard_bound=op.discard+rhs_discard, max_elements=max_elements,
              peak_product_bond=op.peak_product_bond,peak_product_core_elements=op.peak_product_core_elements,
              matvec_count=op.applications)
    if not np.isfinite([physical,orthogonal,stationarity,raw,h_after]).all() or stationarity>max(atol,rtol*exact_target)*(1+1e-6):
        raise gs.GalerkinError('nonconvergence','Streamed residual failed the original equation audit',
                               diagnostics=diag)
    return velocity,gs._evidence(diag)
