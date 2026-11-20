import unittest
import numpy as np
from galerkin_sylvester_tdvp import galerkin_solver as original, witness_streaming as streaming
from galerkin_sylvester_tdvp.models import make_model, mpo, tn, ts


class StreamingTests(unittest.TestCase):
    def test_product_encode_and_qr_norm_on_cancelling_terms(self):
        rng=np.random.default_rng(112)
        for n in (2,3,4):
            d=2**n
            a,b=[rng.normal(size=(d,d))+1j*rng.normal(size=(d,d)) for _ in range(2)]
            am,bm=[ts.matrix_to_mpo(x,n) for x in (a,b)]
            small=1e-9*a
            cm=ts.matrix_to_mpo(a@b-small,n)
            p=streaming.ProductCores(am,bm,2_000_000)
            for mode in ('one','two'):
                frame=original.build_frame(make_model('E',n).root,mode)
                np.testing.assert_allclose(frame.encode(p),frame.encode(mpo.multiply(am,bm)),atol=1e-13)
            value=streaming.sum_norm([(1.,p),(-1.,cm)],n)
            self.assertAlmostEqual(value,np.linalg.norm(small),delta=2e-12)

    def test_constrained_velocity_and_residual_agree_with_dense_and_original(self):
        for f in ('D','E'):
            for mode in ('one','two'):
                model=make_model(f,3)
                x=ts.matrix(model.root)
                g=model.generator(mpo.multiply(model.root,model.root))
                oracle=original.dense_oracle(model.root,ts.matrix(g),mode=mode)
                kwargs=dict(lower_bound=model.floor,mode=mode,rtol=1e-10,work_bond=16)
                v,info=streaming.solve_velocity(model.root,model.generator,**kwargs)
                ref,details=original.solve_velocity(model.root,model.generator,**kwargs)
                np.testing.assert_allclose(ts.matrix(v),oracle['B_velocity'],atol=1e-10)
                self.assertLess(tn.distance(v,ref),1e-11)
                for key in ('physical_residual_abs','orthogonal_residual_abs','galerkin_residual_abs'):
                    self.assertAlmostEqual(info[key],details[key],delta=1e-12)
                self.assertAlmostEqual(info['physical_residual_abs'],
                                       np.linalg.norm(x@ts.matrix(v)+ts.matrix(v)@x-ts.matrix(g)),delta=1e-13)

    def test_nonstationary_midpoint_residual_at_restricted_frame(self):
        from galerkin_sylvester_tdvp.evolution import _retract
        from galerkin_sylvester_tdvp.tests.test_solver import problem
        state, generator, _, lower = problem()
        initial, _ = original.solve_velocity(state, generator, lower_bound=lower)
        mid, _ = _retract(tn.add(state, initial, beta=.01), 2, False, 0)
        x = ts.matrix(mid)
        current_floor = float(np.linalg.eigvalsh(x)[0] * .99)
        g = ts.matrix(generator(mpo.multiply(mid, mid)))
        opts = dict(mode='one', lower_bound=current_floor, rtol=1e-11)
        oracle = original.dense_oracle(mid, g, mode='one')
        v, info = streaming.solve_velocity(mid, generator, **opts)
        v_matrix = ts.matrix(v)
        np.testing.assert_allclose(v_matrix, oracle['B_velocity'], atol=2e-11, rtol=0.)
        residual = (x @ v_matrix + v_matrix @ x - g).ravel()
        projection = oracle['B'] @ (oracle['B'].conj().T @ residual)
        self.assertGreater(np.linalg.norm(residual - projection), .01)
        self.assertAlmostEqual(info['physical_residual_abs'], np.linalg.norm(residual), delta=1e-12)
        self.assertAlmostEqual(info['orthogonal_residual_abs'], np.linalg.norm(residual-projection), delta=1e-12)
