"""Dense lower-spectrum comparisons for the optional block-product certificate."""
import unittest

import numpy as np
from galerkin_sylvester_tdvp.models import make_model, ts, tn
from galerkin_sylvester_tdvp.witness_stage import _whitened_pauli
from galerkin_sylvester_tdvp.witness_block_stage import block_relative_floor, make_block_stage_floor


class BlockStageTests(unittest.TestCase):
    def test_conservative_over_complex_perturbations_and_incomplete_last_block(self):
        n=5
        model=make_model('D',n)
        base=ts.matrix(model.root)
        rng=np.random.default_rng(73)
        a=rng.normal(size=base.shape)+1j*rng.normal(size=base.shape)
        l,u=np.linalg.eigh(base)
        w=(u/np.sqrt(l))@u.conj().T
        had_certificate=False
        for eps in (0.,.000013,.08):
            candidate=base+eps*a
            train,info=_whitened_pauli(ts.matrix_to_mpo(candidate,n),.12,.35)
            oracle=np.linalg.eigvalsh(w@((candidate+candidate.conj().T)/2)@w)[0]
            for width in (2,3):
                check=block_relative_floor(train,width)
                floor=check.get('relative_floor')
                if floor is not None:
                    self.assertLessEqual(floor-info['discarded_bound']*2**(n/2),oracle+2e-13)
                    had_certificate=had_certificate or eps>0
        self.assertTrue(had_certificate)

    def test_wrong_sign_rejected_and_identity_control_exact(self):
        m=make_model('D',4)
        t,_=_whitened_pauli(m.root,.12,.35)
        self.assertAlmostEqual(block_relative_floor(t)['relative_floor'],1,places=12)
        f=make_block_stage_floor('D',4,m)
        self.assertLess(f(tn.scale(m.root,-1)),0)
