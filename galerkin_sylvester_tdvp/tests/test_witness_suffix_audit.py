import unittest
import numpy as np
from galerkin_sylvester_tdvp import witness_suffix_audit as audit
from galerkin_sylvester_tdvp.models import make_model,mpo,tn,ts


class SuffixBounds(unittest.TestCase):
    def test_every_cut_fourth_moment_matches_dense(self):
        rng = np.random.default_rng(927)
        n=4
        x=ts.matrix_to_mpo(rng.normal(size=(16,16))+1j*rng.normal(size=(16,16)),n)
        x=audit.base._canonical_right(x)
        factors=audit.suffix_factors(x)
        for j in range(n):
            cut=[]
            for row in range(x.tensors[j].shape[0]):
                a=tn.MPS(n-j,max_bond=16)
                a.tensors=[x.tensors[j][row:row+1]]+x.tensors[j+1:]
                cut.append(ts.matrix(a))
            l=sum(a.conj().T@a for a in cut)
            r=sum(a@a.conj().T for a in cut)
            np.testing.assert_allclose(factors[j],[np.sqrt(np.linalg.norm(l)),np.sqrt(np.linalg.norm(r))],rtol=2e-11)

    def test_noncommuting_complex_bounds_with_and_without_adjoint_reuse(self):
        rng=np.random.default_rng(505)
        n=4
        a,b=[rng.normal(size=(16,16))+1j*rng.normal(size=(16,16)) for _ in range(2)]
        for herm in (False,True):
            x,v=(a+a.conj().T,b+b.conj().T) if herm else (a,b)
            left,right,rad,method=audit.suffix_products(ts.matrix_to_mpo(x,n),ts.matrix_to_mpo(v,n),1000.,2_000_000)
            self.assertEqual('adjoint' in method,herm)
            self.assertGreater(rad, .1)
            self.assertLessEqual(rad,1000.)
            fullerror=np.linalg.norm(x@v-ts.matrix(left))+np.linalg.norm(v@x-ts.matrix(right))
            self.assertLessEqual(fullerror,rad+3e-11)

    def test_diffuse_suffix_bound_is_sharper_than_rank_only(self):
        n=12
        x=tn.MPS(n,max_bond=1)
        x.tensors=[np.eye(2).reshape(1,4,1)/np.sqrt(2) for _ in range(n)]
        f=audit.suffix_factors(audit.base._canonical_right(x))
        np.testing.assert_allclose(f[:,0],2.**(-np.arange(n,-1,-1)/4))
        self.assertLess(f[1,0], .16)
