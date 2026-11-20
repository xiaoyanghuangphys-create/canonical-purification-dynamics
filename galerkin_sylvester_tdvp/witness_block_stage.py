"""Positivity bound using block marginals of the numerical stage in a whitened basis.

This is an optional refinement of witness_stage's initial-circuit congruence.
It depends only on the represented trial MPO and the initial preparation.
"""
import numpy as np
from scipy.linalg import null_space
from .models import mpo
from .witness_stage import _whitened_pauli, make_stage_floor


def _block_core(train, left, right):
    """Virtual-operator matrix for a consecutive block (physical basis order i...,k...)."""
    sigma=np.stack([mpo.I,mpo.X,np.array([[0,-1j],[1j,0]]),mpo.Z]).reshape(4,4)
    block=np.einsum('apb,ps->asb',train.tensors[left],sigma,optimize=True)
    for site in range(left+1,right):
        local=np.einsum('apb,ps->asb',train.tensors[site],sigma,optimize=True)
        block=np.tensordot(block,local,axes=(-1,0))
    width=right-left
    l,r=block.shape[0],block.shape[-1]
    order=[0,2*width+1]+list(range(1,2*width+1,2))+list(range(2,2*width+1,2))
    return block.reshape([l]+[2,2]*width+[r]).transpose(order).reshape(l,r,2**width,2**width)


def block_relative_floor(train,width=3):
    """Bound Herm(R) by a positive scalar via a positive product of block operators.

    Set the first virtual component to the exact identity-traced environment
    at every block boundary. The all-zero path contributes a product of local
    operators. Congruence by each positive local Hermitian part whitens that
    product. Bound the remainder with the sum of spectral-norm products of all
    virtual paths visiting a nonzero sector; charge the skew of the zero path.
    """
    if not 1<=width<=4:
        raise ValueError('choose a block of one to four spins')
    n=train.n
    left=[np.ones(1,complex)]
    for a in train.tensors:
        left.append(left[-1] @ a[:,0,:])
    right=[None]*(n+1)
    right[-1]=np.ones(1,complex)
    for j in range(n-1,-1,-1):
        right[j]=train.tensors[j][:,0,:] @ right[j+1]
    if left[-1].item().real<=0:
        return dict(status='identity_trace_nonpositive')
    inv=np.ones((1,1),complex)
    marked=np.zeros(1)
    zero_path=1.
    skew_log=0.
    log_floor=0.
    for j in range(0,n,width):
        end=min(n,j+width)
        if end==n:
            gauge=inv_next=np.ones((1,1),complex)
        else:
            u=right[end]/np.linalg.norm(right[end])
            pivot=left[end] @ u
            if abs(pivot)<1e-9*np.linalg.norm(left[end]):
                return dict(status='identity_gauge_ill_conditioned')
            dual=left[end]/pivot
            q=null_space(left[end][None,:])
            gauge=np.column_stack((u,q))
            inv_next=np.vstack((dual,q.conj().T-np.outer(q.conj().T@u,dual)))
        raw=_block_core(train,j,end)
        block=np.einsum('ai,ijpq,jb->abpq',inv,raw,gauge,optimize=True)
        inv=inv_next
        anchor=(block[0,0]+block[0,0].conj().T)/2
        vals,U=np.linalg.eigh(anchor)
        if vals[0]<=0:
            return dict(status='nonpositive_local_anchor',block=j,block_min=float(vals[0]))
        whiten=(U/np.sqrt(vals))@U.conj().T
        log_floor+=np.log(vals[0])
        # Batch physical spectral norms after whitening.
        normalized=(whiten@block)@whiten
        skew_log+=np.log1p(np.linalg.norm(normalized[0,0]-np.eye(len(vals)),2))
        norms=np.linalg.svd(normalized,compute_uv=False)[...,0]
        marked=marked@norms
        marked[1:]+=zero_path*norms[0,1:]
        zero_path*=norms[0,0]
    drift=float(marked[0]+np.expm1(skew_log))
    scale=float(np.exp(log_floor))
    return dict(status='bounded',width=width,block_offpath_bound=float(marked[0]),
                block_zero_path_error=float(np.expm1(skew_log)),product_floor=scale,
                relative_floor=scale*(1-drift) if drift<1 else None)


def make_block_stage_floor(family,n,model,record=None,width=3):
    """Refine an inconclusive Pauli/Frobenius witness_stage floor on the D family."""
    first=make_stage_floor(family,n,model,record=record)
    if family!='D':
        return first
    def floor(state):
        lower=first(state)
        if lower>0:
            return lower
        train, info=_whitened_pauli(state,model.parameters['polarization'],model.parameters['angle'])
        details=block_relative_floor(train,width)
        proposal=details.get('relative_floor')
        allowance=info['discarded_bound']*2.**(n/2)
        if proposal is not None and proposal>allowance:
            lower=max(lower,model.floor*(proposal-allowance))
        if record:
            record(dict(details,source='zero_path_block_congruence',pauli_discard_op_allowance=allowance,
                        lower_bound=lower))
        return lower
    return floor
