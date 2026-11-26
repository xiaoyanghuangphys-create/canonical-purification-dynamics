"""Analytic parity controls and independent endpoint scoring in interleaved order.

The production predictors and lift adapters must only use root_mpo at initialization.
score constructs a fresh reference and contracts the actual stored amplitude.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from galerkin_sylvester_tdvp.models import tn, mpo

I, X, Z = mpo.I, mpo.X, mpo.Z
SQRT2 = np.sqrt(2.)


def root_coefficients(p, b):
    radius = np.hypot(p, b)
    if not np.isfinite(radius) or radius >= 1:
        raise ValueError("Parity benchmark requires p^2+b^2 < 1.")
    denom = np.sqrt(1+radius) + np.sqrt(1-radius)
    c = 1./denom
    return np.array([denom/2, c*p, c*b])


def _strings(n, identity, parity, transverse):
    """Right-canonical bond-two tensor train in a normalized Pauli-string basis."""
    if n < 2:
        raise ValueError("At least two spins are required.")
    ii, xx, zz = (a.reshape(4)/SQRT2 for a in (I, X, Z))
    first = identity*ii + transverse*xx
    if parity == 0.:
        state = tn.MPS(n, max_bond=1)
        state.tensors = [first.reshape(1,4,1)] + [ii.reshape(1,4,1).copy() for _ in range(n-1)]
    else:
        state = tn.MPS(n, max_bond=2)
        middle = np.zeros((2,4,2),complex)
        middle[0,:,0], middle[1,:,1] = ii, zz
        state.tensors = [np.column_stack((first,parity*zz)).reshape(1,4,2)]
        state.tensors += [middle.copy() for _ in range(n-2)]
        state.tensors.append(np.stack((ii,zz)).reshape(2,4,1))
    state.center = 0
    return state


def root_mpo(n, p, b):
    return _strings(n, *root_coefficients(p,b))


def _entropy_from_bias(x):
    probs = np.array([(1.+x)/2, (1.-x)/2])
    return -np.dot(probs,np.log(probs))


def cmi(p, b):
    root_coefficients(p,b)
    return float(_entropy_from_bias(b) - _entropy_from_bias(np.hypot(p,b)))


def unitary_floor(p, b_before, b_after):
    """Optimal proper-window error for exact incoming and target roots."""
    first, last = [root_coefficients(p,b) for b in (b_before,b_after)]
    weights = [[np.sqrt(1.-v[1]**2),abs(v[1])] for v in (first,last)]
    return float(np.linalg.norm(np.subtract(*weights)))


def _pauli_power(b, exponent):
    """Power of I+bQ in the faithful Clifford representation Q=X, P=Z."""
    hi,lo = (1.+b)**exponent, (1.-b)**exponent
    return ((hi+lo)*I+(hi-lo)*X)/2


def _clifford_root(p,b):
    a,pc,bc = root_coefficients(p,b)
    return a*I+pc*Z+bc*X


def half_petz_matrix(p,b_before,b_after):
    """Faithful 2-by-2 representative with normalized trace, proper W.

    Conjugation by site-1 Z fixes P and negates Q and QP, represented by Z.
    """
    if b_before == 0.:
        if b_after != 0.:
            raise ValueError("Dephasing cannot create transverse polarization.")
        damping = 1.
    else:
        damping = b_after/b_before
    if not 0 <= damping <= 1.+2.e-15:
        raise ValueError("Expected a forward dephasing interval.")
    initial = _pauli_power(b_before,.5) @ _clifford_root(p,b_before)
    evolved = ((1+damping)*initial + (1-damping)*Z@initial@Z)/2
    return _pauli_power(b_after,-.5) @ evolved


def half_petz_metrics(p,b_before,b_after):
    y = half_petz_matrix(p,b_before,b_after)
    target = _clifford_root(p,b_after)
    length = float(np.linalg.norm(y)/SQRT2)
    return dict(raw_error=float(np.linalg.norm(y-target)/SQRT2),
                normalized_error=float(np.linalg.norm(y/length-target)/SQRT2),
                norm=length, hermiticity_defect=float(np.linalg.norm(y-y.conj().T)/SQRT2),
                qp_coefficient=float((np.vdot(X@Z,y)/2).real))


def _pauli_projection(state,first,rest):
    env=np.ones(1,complex)
    for j,core in enumerate(state.tensors):
        bra=(first if j==0 else rest).conj().reshape(4)/SQRT2
        env=env@np.einsum('s,lsr->lr',bra,core)
    return env.item()


def score(state,t,p=.6,b0=.4,*,max_product_elements=2_000_000):
    """Score actual, unnormalized Y. MPS products use raw physical YY*, not Y².

    The density residual is formed after scaling BOTH terms by 2^(N/2).
    The scaled residual and its relative form remain interpretable when the
    unscaled value is tiny. All norms subtract trains before orthogonalizing.
    """
    n=state.n
    b=b0*np.exp(-t)
    exact=root_mpo(n,p,b)
    adj=mpo.adjoint(state)
    length=tn.norm(state)
    canonical=deepcopy(state)
    canonical.center=n-1
    canonical.move_center(0)
    first=canonical.tensors[0].reshape(2,2,-1)
    local=np.einsum('ier,jer->ij',first,first.conj())
    result=dict(global_cp_error=tn.distance(state,exact), norm=float(length),
                norm_drift=float(abs(length-1.)), density_trace_error=float(abs(length**2-1)),
                hermiticity_defect=tn.distance(state,adj), actual_root_bond=state.bond_dimension(),
                site1_physical_marginal_error=float(np.linalg.norm(local-(I+b*X)/2)),
                scored_time=float(t))
    for key,head,tail in [('i',I,I),('p',Z,Z),('q',X,I),('qp',X@Z,Z)]:
        coeff=_pauli_projection(state,head,tail)
        result['root_'+key+'_re'],result['root_'+key+'_im']=float(coeff.real),float(coeff.imag)
    elements=sum(4*(core.shape[0]**2+2)*(core.shape[2]**2+2) for core in state.tensors)
    result['density_scoring_product_elements']=elements
    if elements > max_product_elements:
        result.update(density_residual=None,density_residual_scaled=None,density_residual_relative=None,
                      density_score_status='allocation_guard')
        return result
    represented=mpo.multiply(state,adj)
    represented.tensors=[core*SQRT2 for core in represented.tensors]
    rho_scaled=_strings(n,1.,p,b)
    delta=tn.distance(represented,rho_scaled)
    result.update(density_residual=float(np.exp2(-.5*n)*delta),
                  density_residual_scaled=float(delta), density_residual_relative=float(delta/np.sqrt(1+p*p+b*b)),
                  density_score_status='completed')
    return result


def dense_power(a, exponent):
    vals, vecs=np.linalg.eigh((a+a.conj().T)/2)
    if vals.min()<=0:
        raise ValueError("Dense parity power requires full rank.")
    return (vecs*vals**exponent)@vecs.conj().T


def _kron(items):
    out=np.ones((1,1),complex)
    for item in items:
        out=np.kron(out,item)
    return out


def dense_operators(n,*,max_dimension=256):
    """Small-chain physical matrices only, checked before allocation."""
    if n<2 or 2**n>max_dimension:
        raise ValueError("Dense physical dimension exceeds oracle budget.")
    return np.eye(2**n), _kron([Z]*n), _kron([X]+[I]*(n-1))


def dense_density(n,p,b,*,max_dimension=256):
    root_coefficients(p,b)
    ident,parity,transverse=dense_operators(n,max_dimension=max_dimension)
    return (ident+p*parity+b*transverse)/2**n


def marginal(rho,keep):
    n=int(np.log2(len(rho)))
    keep=list(keep)
    if len(set(keep))!=len(keep) or any(j<0 or j>=n for j in keep):
        raise ValueError("Invalid marginal sites")
    rest=[j for j in range(n) if j not in keep]
    order=keep+rest+[j+n for j in keep+rest]
    dim=2**len(keep)
    block=rho.reshape([2]*(2*n)).transpose(order).reshape(dim,2**len(rest),dim,2**len(rest))
    return np.trace(block,axis1=1,axis2=3)


def dense_cmi(rho,a,b,c):
    def entropy(sites):
        vals=np.linalg.eigvalsh(marginal(rho,sites))
        return -np.dot(vals,np.log(vals))
    return float(entropy(a+b)+entropy(b+c)-entropy(b)-entropy(a+b+c))


def dense_dephase(rho,h,site=0):
    """Finite Kraus update via diagonal signs, avoiding full Kraus products."""
    n=int(np.log2(len(rho)))
    signs=1-2*((np.arange(len(rho))>>(n-1-site))&1)
    q=-np.expm1(-h)/2
    return (1-q)*rho+q*(signs[:,None]*rho*signs[None,:])


def interleaved(matrix):
    n=int(np.log2(len(matrix)))
    return matrix.reshape([2]*(2*n)).transpose([k for j in range(n) for k in (j,j+n)]).reshape(-1)


def dense_hp_update(y,rho,h,width):
    """Direct dense physical Kraus contractions on an arbitrary full amplitude."""
    n=int(np.log2(len(rho)))
    updated=dense_dephase(rho,h)
    r=marginal(rho,range(width))
    s=marginal(updated,range(width))
    eye=np.eye(2**(n-width))
    right=np.kron(dense_power(r,.5),eye)
    left=np.kron(dense_power(s,-.5),eye)
    return left@dense_dephase(right@y,h),updated


def dense_optimum(y0,y1,width,*,max_gate_elements=65536):
    """Independent dense polar cross-matrix optimum; returns attained error."""
    dim=4**width
    if dim**2>max_gate_elements:
        raise ValueError("Dense unitary exceeds oracle budget.")
    a,b=interleaved(y0).reshape(dim,-1),interleaved(y1).reshape(dim,-1)
    u,_,vh=np.linalg.svd(b@a.conj().T,full_matrices=False)
    gate=u@vh
    return float(np.linalg.norm(gate@a-b)),gate


def explicit_optimal_action(first,last,width,p,b_before,b_after):
    """Apply an attaining block rotation without forming a dense local gate.

    Proper windows rotate the normalized I/Q branches and leave P unchanged.
    On the whole chain a Householder maps the full (normalized) roots.
    Here both endpoints are oracle data. This is never a production lift.
    """
    n=int(np.log2(len(first)))
    a=interleaved(first).reshape(4**width,-1)
    if width==n:
        target=interleaved(last)[:,None]
        difference=a-target
        norm=np.linalg.norm(difference)
        if norm==0: return a.copy()
        unit=difference/norm
        return a-2*unit@(unit.conj().T@a)
    init,final=(root_coefficients(p,b) for b in (b_before,b_after))
    theta=np.arctan2(final[2],final[0])-np.arctan2(init[2],init[0])
    ident=interleaved(np.eye(2**width))/2**(.5*width)
    trans=interleaved(_kron([X]+[I]*(width-1)))/2**(.5*width)
    v=np.column_stack((ident,trans))
    rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    return a+v@((rotation-np.eye(2))@(v.conj().T@a))
