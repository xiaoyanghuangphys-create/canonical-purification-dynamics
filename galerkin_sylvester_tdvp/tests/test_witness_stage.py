"""Independent finite-chain checks of the witnessed stage audit."""
import unittest
from itertools import product

import numpy as np

from galerkin_sylvester_tdvp.models import make_model, mpo, tn, ts, witness
from galerkin_sylvester_tdvp.witness_stage import make_stage_floor, _whitened_pauli, _pauli_lower


class WitnessStageTests(unittest.TestCase):
    def test_unwinds_gate_and_whitens_with_correct_pauli_indexing(self):
        rng = np.random.default_rng(167)
        n=3
        model=make_model('D',n)
        x=rng.normal(size=(8,8))+1j*rng.normal(size=(8,8))
        local_h=np.array([[1,1],[1,-1]])/np.sqrt(2)
        all_h=np.kron(local_h,np.kron(local_h,local_h))
        cp=np.diag([np.exp(1j*.35*(i*j+j*k)) for i,j,k in product(range(2),repeat=n)])
        u=all_h@cp@all_h
        anchor=u.conj().T@ts.matrix(model.root)@u
        inv=np.diag(np.diag(anchor).real**-.5)
        rel=inv@u.conj().T@x@u@inv
        paulis=[mpo.I,mpo.X,np.array([[0,-1j],[1j,0]]),mpo.Z]
        expected=np.array([np.trace(np.kron(p,np.kron(q,r))@rel)/8
                           for p,q,r in product(paulis,repeat=n)])
        train,info=_whitened_pauli(ts.matrix_to_mpo(x,n),.12,.35)
        np.testing.assert_allclose(train.dense(),expected,atol=2e-13)
        self.assertLess(info['discarded_bound'],1e-13)

    def test_relative_and_absolute_floors_are_conservative(self):
        model=make_model('D',4)
        x=ts.matrix(model.root)
        rng=np.random.default_rng(618)
        test_dir=rng.normal(size=x.shape)+1j*rng.normal(size=x.shape)
        test_dir=(test_dir+test_dir.conj().T)/2
        for scale in (0.0001,0.02,0.8):
            a=x+scale*test_dir
            state=ts.matrix_to_mpo(a,4)
            train,_=_whitened_pauli(state,.12,.35)
            coeff=train.dense()
            check=_pauli_lower(train)
            if check is not None:
                self.assertGreaterEqual(check['nonidentity_pauli_majorant']+1e-12,np.abs(coeff[1:]).sum())
            low=make_stage_floor('D',4,model)(state)
            self.assertLessEqual(low,np.linalg.eigvalsh(a)[0]+1e-12)

    def test_native_size_positive_and_wrong_sign_controls(self):
        n=24
        m=make_model('D',n)
        x=witness.prepared_power(n,.5,polarization=.1195)
        events=[]
        f=make_stage_floor('D',n,m,record=events.append)
        self.assertGreater(f(x),0)
        self.assertEqual(events[-1]['source'],'unwound_pauli_congruence')
        self.assertLess(f(tn.scale(x,-1)),0)

    def test_parity_uses_actual_state(self):
        m=make_model('E',4)
        f=make_stage_floor('E',4,m)
        self.assertAlmostEqual(f(m.root),m.floor,places=12)
        wrong=tn.scale(m.root,-1.)
        self.assertLess(f(wrong),0)
