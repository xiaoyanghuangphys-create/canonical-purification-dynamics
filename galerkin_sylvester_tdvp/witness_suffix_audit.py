"""Optional high-bond physical residual audit with tighter product-suffix bounds.

All norms and truncation enclosures assume accurate floating-point contractions.
The full Galerkin equation and the endpoint certificate use their original exact
MPO expressions and are independent of this diagnostic.
"""
import numpy as np

from . import galerkin_solver as gs
from . import witness_precond_solver as base
from .models import mpo, tn
from .witness_streaming import _sum_norms


def suffix_factors(canonical_root):
    """Fourth-moment bounds for the suffix root's left/right multiplication.

    E_abcd = Tr(X_a† X_b X_c† X_d), using the root suffix at the current cut.
    Direction L contracts E_iijj and direction R contracts E_ijji. The fourth
    roots of these traces are sqrt(||sum_i X_i† X_i||_F) and its reversed
    counterpart. Multiplying either by sqrt(r_v) bounds the norm of the map
    with suffix rows X_i V_j or V_j X_i, for normalized suffixes V_j.
    """
    n = canonical_root.n
    factors = np.ones((n+1, 2))
    environment = np.ones((1,1,1,1),complex)
    log_scale = 0.
    for j in range(n-1, -1, -1):
        core = canonical_root.tensors[j]
        w = core.reshape(core.shape[0],2,2,core.shape[-1])
        # Fixed contraction order avoids a naive eight-virtual-index einsum.
        t = np.tensordot(w.conj(),environment,axes=(3,0))       # a i j q r s
        t = np.tensordot(w,t,axes=((1,3),(1,3)))               # b k a j r s
        t = np.tensordot(w.conj(),t,axes=((2,3),(1,4)))        # c l b a j s
        t = np.tensordot(w,t,axes=((1,2,3),(1,4,5)))          # d c b a
        environment = t.transpose(3,2,1,0)
        norm = np.max(np.abs(environment))
        if not np.isfinite(norm) or norm <= 0:
            raise ArithmeticError('Nonfinite/zero root suffix fourth moment')
        environment /= norm
        log_scale += np.log(norm)
        traces = [np.einsum('aacc->', environment), np.einsum('abba->', environment)]
        for direction, trace in enumerate(traces):
            if not np.isfinite(trace) or trace.real <= 0 or abs(trace.imag) > 1e-8*trace.real:
                raise ArithmeticError('Invalid positive root suffix fourth moment')
            factors[j,direction] = np.exp((log_scale+np.log(trace.real))*.25)
    # Ordinary floating point. Charge a numerical margin to evaluated moments.
    factors *= 1.+64*np.finfo(float).eps*n*max(c.shape[0] for c in canonical_root.tensors)**2
    return factors


def _zip_product(a, b, bounds, budget, max_elements, max_rank=192):
    """Absolute allowance by left Schmidt discards times upper bounds for suffix maps."""
    carry = np.ones((1,1,1),complex)
    cores, radius = [], 0.
    for j,(am,bm) in enumerate(zip(a.tensors,b.tensors)):
        ar,br = am.shape[-1],bm.shape[-1]
        kept = carry.shape[0]
        panel_elements = 4*kept*max(ar*br,ar*bm.shape[0])
        if panel_elements > max_elements:
            raise gs.GalerkinError('memory_budget','Suffix-audit panel exceeds cap',
                                   requested_core_elements=panel_elements,max_elements=max_elements)
        ac = am.reshape(am.shape[0],2,2,ar)
        bc = bm.reshape(bm.shape[0],2,2,br)
        t = np.tensordot(carry,ac,axes=(1,0))
        t = np.tensordot(t,bc,axes=((1,3),(0,1)))
        panel=t.transpose(0,1,3,2,4).reshape(4*kept,ar*br)
        if j==a.n-1:
            cores.append(panel.reshape(kept,4,1))
            break
        u,s,vh = np.linalg.svd(panel,full_matrices=False)
        tails = np.sqrt(np.cumsum(s[::-1]**2))
        threshold = (budget-radius)/((a.n-1-j)*bounds[j+1])
        rank = max(1,len(s)-np.searchsorted(tails,threshold,side='right'))
        if rank > max_rank:
            raise gs.GalerkinError('memory_budget','Suffix-audit product rank exceeds cap',
                                   required_rank=rank,max_rank=max_rank,site=j,
                                   suffix_bound=bounds[j+1],absolute_budget=budget)
        radius += np.linalg.norm(s[rank:])*bounds[j+1]
        cores.append(u[:,:rank].reshape(kept,4,rank))
        carry = (s[:rank,None]*vh[:rank]).reshape(rank,ar,br)
    out = tn.MPS(a.n,max_bond=max(c.shape[-1] for c in cores))
    out.tensors,out.center = cores,a.n-1
    return out,radius


def suffix_products(root, velocity, budget, max_elements, max_rank=192):
    """Approximate both ordered products, charging both truncation and skew."""
    x = base._canonical_right(root)
    v = base._canonical_right(velocity)
    factors = suffix_factors(x)
    x_ranks = np.array([c.shape[0] for c in x.tensors]+[1])
    v_ranks = np.array([c.shape[0] for c in v.tensors]+[1])
    suffix = np.sqrt(v_ranks[:,None])*np.minimum(np.sqrt(x_ranks[:,None]),factors)
    # If the skew bound is negligible, use (XV)† as the second product.
    # The Schatten-four norm of the full root bounds its operator norm.
    manifest_root = all(np.array_equal(c,c[:,[0,2,1,3],:].conj()) for c in root.tensors)
    x_skew = 0. if manifest_root else tn.distance(root,mpo.adjoint(root))
    v_skew = tn.distance(velocity,mpo.adjoint(velocity))
    skew_allowance = v_skew*factors[0,0]+tn.norm(velocity)*x_skew
    if skew_allowance <= .05*budget:
        left,err = _zip_product(x,v,suffix[:,0],(budget-skew_allowance)/2,max_elements,max_rank)
        return left,mpo.adjoint(left),2*err+skew_allowance,'suffix_fourth_moment_adjoint_qr'
    left,err_l = _zip_product(x,v,suffix[:,0],budget/2,max_elements,max_rank)
    right,err_r = _zip_product(v,x,suffix[:,1],budget/2,max_elements,max_rank)
    return left,right,err_l+err_r,'suffix_fourth_moment_qr'


def refined_audit_norms(root, velocity, g, projected, frame, g_norm, rtol, max_elements, *, max_rank=192, diagnostic_relative_width=1e-6):
    """Use baseline exact QR at small bonds, tighten the high-bond audit."""
    projected_state = frame.decode(projected)
    previous = 1
    products = [a.shape[-1]*b.shape[-1] for a,b in zip(root.tensors,velocity.tensors)]
    for j,dr in enumerate(products):
        dl = 1 if j == 0 else products[j-1]
        width = 2*dr + g.tensors[j].shape[-1]+projected_state.tensors[j].shape[-1]
        if 4*dl*dr>max_elements or 4*previous*width>6_000_000:
            break
        previous=min(4*previous,width)
    else:
        return base_audit_norms(root,velocity,g,projected,frame,g_norm,rtol,max_elements)
    budget = max(1e-12,min(rtol*.01,diagnostic_relative_width))*g_norm
    left,right,radius,method = suffix_products(root,velocity,budget,max_elements,max_rank=max_rank)
    physical,outside = _sum_norms([(1.,left),(1.,right),(-1.,g),(-1.,projected_state)],
                                   root.n,[range(3),range(4)])
    return physical,outside,radius,method


# Retain the unpatched implementation for workers injecting this callback.
base_audit_norms = base._audit_norms
