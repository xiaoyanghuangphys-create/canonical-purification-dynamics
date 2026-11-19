"""Shared physical models. Only initialization and independent scoring use reference roots."""
from dataclasses import dataclass
import numpy as np
from scipy.sparse.linalg import expm_multiply
from factor_tdvp import tangent_solver as ts
from factor_tdvp.baseline import _legacy_module
from factor_tdvp_comparison import factor_benchmarks as common
import scalable_cp_witness as witness

tn, mpo = ts.tn, ts.mpo


def kron_state(matrices):
    state = tn.MPS(len(matrices), max_bond=1)
    state.tensors = [np.asarray(a, complex).reshape(1, 4, 1) for a in matrices]
    state.center = state.n - 1
    return state


def parity_floor_setup(n):
    """Fit the STORED state in the normalized I,Z...Z,X1 Pauli subspace.

    Weyl and ||remainder||op <= ||remainder||F apply even off that subspace.
    This audit uses no target coefficients and no physical time.
    """
    paulis = [[mpo.I]*n, [mpo.Z]*n, [mpo.X]+[mpo.I]*(n-1)]
    basis = [kron_state([p/np.sqrt(2) for p in word]) for word in paulis]
    def lower(state):
        coeff = [tn.overlap(p, state).real for p in basis]
        fit = tn.add(tn.add(basis[0], basis[1], coeff[0], coeff[1]), basis[2], beta=coeff[2])
        return (coeff[0]-np.hypot(*coeff[1:]))*2.**(-n/2) - tn.distance(state, fit)
    return lower


@dataclass
class Model:
    root: object
    generator: object
    floor: float
    dense_physics: object
    reference: object
    parameters: dict
    pure: bool = False


class Generator:
    """Current-density MPO action. Compression in the RHS is separately audited."""
    def __init__(self, n, kind, rates, ham=None, work=32):
        self.n, self.kind, self.rates = n, kind, rates
        self.ham, self.work = ham, work
        self.discards = []

    def __call__(self, rho):
        def rounded(x):
            y, info = tn.compress(x, self.work, tolerance=1e-14)
            self.discards.append(info['discarded_bound'])
            if info['discarded_bound'] > 1e-10*max(tn.norm(x), np.finfo(float).tiny):
                raise ArithmeticError('RHS compression exceeded its relative allowance')
            return y
        rho = rounded(rho)
        if self.kind == 'pair':
            total = tn.scale(rho, 0.)
            for j, rate in enumerate(self.rates):
                transformed = tn.scale(rho, 1.)
                for k in (j, j+1):
                    transformed.tensors[k] = transformed.tensors[k] * np.array([1,-1,-1,1])[None,:,None]
                total = rounded(tn.add(total, tn.add(transformed, rho, rate, -rate)))
            return total
        result = mpo.onsite_dissipation(rho, self.rates, self.kind)
        if self.ham is not None:
            coherent = tn.add(mpo.multiply(self.ham,rho),mpo.multiply(rho,self.ham),-1j,1j)
            result = tn.add(result, coherent)
        return rounded(result)


def make_model(family, n, work=32):
    if family in ('D','E'):
        power = witness.prepared_power if family=='D' else witness.parity_power
        root = power(n, .5)
        floor = .44**(n/2) if family=='D' else np.sqrt(1-np.hypot(.6,.4))*2.**(-n/2)
        def dense_physics():
            return (np.zeros((2**n,2**n),complex),
                    [common.fd.embed(mpo.Z,j,n)/np.sqrt(2) for j in range(n)])
        def reference(t):
            if family=='E':
                return (power(n,.5,transverse=.4*np.exp(-t)),power(n,1.,transverse=.4*np.exp(-t)))
            rho = witness.channel(power(n,1.),t)
            if n>8:
                return None,rho
            vals, U = np.linalg.eigh(ts.matrix(rho))
            return ts.matrix_to_mpo((U*np.sqrt(vals))@U.conj().T,n),rho
        return Model(root, Generator(n,'dephasing',np.full(n,.5),work=work),
                     floor, dense_physics, reference,
                     dict(coherence='exp(-t)',polarization=.12,angle=.35) if family=='D'
                     else dict(coherence='exp(-t)',parity=.6,transverse=.4))
    if family in ('product','cluster','coherent','commuting'):
        pure = family=='cluster'
        angle = np.pi if pure else 0.
        p = 1. if pure else .31
        phase = common.phase_vector(n,np.full(n-1,angle))
        axis = mpo.Z if family in ('coherent','commuting') else mpo.X
        def root_at(t):
            if family=='coherent':
                angle_h=.43*t
                u=np.cos(angle_h)*mpo.I-1j*np.sin(angle_h)*mpo.X
                local=u@((mpo.I+p*axis)/2)@u.conj().T
            else:
                local=(mpo.I+p*(1 if family=='commuting' else np.exp(-t))*axis)/2
            # Use the analytic small eigenvalue on the initially pure path.
            if pure:
                pop=-np.expm1(-t)/2
                little=(np.sqrt(1-pop)*(mpo.I+mpo.X)+np.sqrt(pop)*(mpo.I-mpo.X))/2
            else:
                vals,U=np.linalg.eigh(local)
                little=(U*np.sqrt(vals))@U.conj().T
            return phase[:,None]*common.kron([little]*n)*phase.conj()[None,:]
        def reference(t):
            x=root_at(t)
            return ts.matrix_to_mpo(x,n),ts.matrix_to_mpo(x@x,n)
        rates=np.full(n,0. if family=='coherent' else .5)
        ham=mpo.hamiltonian(n,jzz=0.,hx=.43,hz=0.) if family=='coherent' else None
        def dense_physics():
            h=ts.matrix(ham) if ham else np.zeros((2**n,2**n),complex)
            return h,[common.fd.embed(mpo.Z,j,n)/np.sqrt(2) for j in range(n)] if family!='coherent' else []
        return Model(ts.matrix_to_mpo(root_at(0),n),Generator(n,'dephasing',rates,ham,work=work),
                     ((1-p)/2)**(n/2),dense_physics,reference,dict(polarization=p,angle=angle,axis='Z' if family in ('coherent','commuting') else 'X'),pure)
    kind, variant = family.split('_')
    pure=variant=='pure'
    h,jumps,liouv,pure_rho=_legacy_module().spin_model(n,kind)
    rho=pure_rho if pure else .8*pure_rho+.2*np.eye(2**n)/2**n
    initial=pure_rho if pure else common.fd.positive_power(rho,.5)
    def reference(t):
        density=expm_multiply(t*liouv,rho.reshape(-1,order='F')).reshape(rho.shape,order='F')
        density=(density+density.conj().T)/2
        # Raw eigensolver on rho cannot resolve a relative density eigenvalue below roundoff.
        vals,U=np.linalg.eigh(density)
        if min(vals)<-1e-12:
            raise ArithmeticError('independent reference is not positive')
        return ts.matrix_to_mpo((U*np.sqrt(np.maximum(vals,0)))@U.conj().T,n),ts.matrix_to_mpo(density,n)
    rates=.3+.12*np.cos(np.arange(n))**2 if kind=='loss' else .4+.1*np.sin(np.arange(n-1))**2
    return Model(ts.matrix_to_mpo(initial,n), Generator(n,kind,rates,mpo.hamiltonian(n) if kind=='loss' else None,work=work),
                 0. if pure else np.sqrt(.2/2**n),lambda:(h,jumps),reference,
                 dict(kind=kind,pure_weight=1. if pure else .8,rates=rates.tolist(),
                      jzz=.7 if kind=='loss' else 0.,hx=.43 if kind=='loss' else 0.,hz=.19 if kind=='loss' else 0.),pure)
