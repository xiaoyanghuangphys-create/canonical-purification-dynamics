"""Current-state positivity audits for the witnessed D/E trajectory benchmarks.

For the controlled-phase preparation D, undo the known initial preparation
unitary and whiten with the product of initial single-spin eigenvalues. In
the resulting frame the initial root is the identity. Bound the stored
stage's smallest eigenvalue by expanding its transformed operator in Pauli
strings. Neither the physical time nor a target endpoint enters this bound.
"""
import numpy as np
from scipy.linalg import null_space

from .models import tn, mpo, parity_floor_setup


def _whitened_pauli(state, polarization, angle):
    """Exact Pauli coefficient train of A^-1/2 X A^-1/2 in A's eigenbasis.

    Here A is the prepared positive initial root. A nearest-neighbor phase
    circuit acts diagonally on matrix entries, so a four-valued memory of the
    preceding ket/bra label suffices. No MPO root inverse or dense state is
    formed. Local coefficients use Tr(P†X)/2.
    """
    h = np.array([[1., 1.], [1., -1.]])/np.sqrt(2)
    rotate = np.kron(h, h)
    label = np.arange(4)
    ket, bra = label//2, label%2
    phase = np.exp(-1j*angle*(np.outer(ket, ket)-np.outer(bra, bra)))
    w = (np.array([1+polarization,1-polarization])/2)**(-.25)
    paulis = np.stack([mpo.I,mpo.X,np.array([[0.,-1j],[1j,0.]]),mpo.Z])
    convert = (paulis.conj().reshape(4,4)*np.outer(w,w).ravel()) @ rotate / 2
    tensors = []
    for j, a in enumerate(state.tensors):
        rotated = np.einsum('us,lsr->lur', rotate, a, optimize=True)
        phases = np.ones((1,4)) if j==0 else phase
        memory = (np.eye(4) if j+1<state.n else np.ones((4,1)))
        core = np.einsum('lur,ku,ub,pu->lkprb', rotated, phases, memory, convert, optimize=True)
        tensors.append(core.reshape(a.shape[0]*len(phases),4,a.shape[-1]*memory.shape[-1]))
    train=tn.MPS(state.n,max_bond=max(t.shape[2] for t in tensors))
    train.tensors, train.center = tensors, state.n-1
    # QR/SVD at the full geometric rank removes redundant memory gauges.
    return tn.compress(train,train.max_bond,tolerance=0.)


def _pauli_lower(train):
    """Lower bound Herm(R) by the identity coefficient less all nonidentity weights.

    The two sectors in the identity transfer, the traced environment and its
    complement, are put in separate virtual coordinates. This keeps cancellations
    in identity-only segments before absolute values bound virtual-path sums.
    """
    left = [np.ones(1,complex)]
    for core in train.tensors:
        left.append(left[-1] @ core[:,0,:])
    scalar = left[-1].item()
    if not np.isfinite(scalar) or scalar.real <= 0:
        return None
    right=[None]*(train.n+1)
    right[-1]=np.ones(1,complex)
    for j in range(train.n-1,-1,-1):
        right[j]=train.tensors[j][:,0,:] @ right[j+1]
    gauge=[np.ones((1,1),complex)]
    inverse=[np.ones((1,1),complex)]
    for prefix,suffix in zip(left[1:-1],right[1:-1]):
        unit=suffix/np.linalg.norm(suffix)
        pivot=prefix @ unit
        if abs(pivot)<1e-9*np.linalg.norm(prefix):
            return None
        null=null_space(prefix[None,:])
        dual=prefix / pivot
        gauge.append(np.column_stack((unit,null)))
        inverse.append(np.vstack((dual,null.conj().T-np.outer(null.conj().T @ unit,dual))))
    gauge.append(np.ones((1,1),complex))
    # Sum nonidentity string weights, allowing cancellation on the complete
    # identity prefix. No numerical equality of virtual transfer entries is
    # assumed when bounding the remaining strings.
    prefix=np.ones(1,complex)
    remainder=np.zeros(1)
    for j,core in enumerate(train.tensors):
        local=np.einsum('al,lpr,rb->apb', inverse[j],core,gauge[j+1],optimize=True)
        nonid=np.abs(np.einsum('l,lpr->pr',prefix,local[:,1:,:])).sum(axis=0)
        remainder=remainder @ np.abs(local).sum(axis=1) + nonid
        prefix=prefix @ local[:,0,:]
    return dict(identity_coefficient=float(scalar.real),nonidentity_pauli_majorant=float(remainder[0]))


def make_stage_floor(family, n, model, record=None):
    """Return a numerical-stage lower bound, conditional on floating-point contractions."""
    if family=='E':
        return parity_floor_setup(n)
    if family!='D':
        raise ValueError('expected the D or E preparation')
    def floor(state):
        if state.n!=n:
            raise ValueError('chain length differs from the initial preparation')
        movement=tn.distance(state,model.root)
        result=model.floor-movement
        row=dict(source='initial_frobenius', lower_bound=result)
        if result<=0:
            train,info=_whitened_pauli(state,model.parameters['polarization'],model.parameters['angle'])
            check=_pauli_lower(train)
            if check is not None:
                # A Pauli-coefficient 2-norm discard bounds the physical
                # operator norm after multiplication by sqrt(2**N).
                allowance=info['discarded_bound']*2.**(n/2)
                relative=check['identity_coefficient']-check['nonidentity_pauli_majorant']-allowance
                row.update(check, relative_floor=relative,pauli_discard_op_allowance=allowance)
                if relative>0:
                    result=model.floor*relative
                    row.update(source='unwound_pauli_congruence',lower_bound=result)
        if record is not None: record(row)
        return result
    return floor
