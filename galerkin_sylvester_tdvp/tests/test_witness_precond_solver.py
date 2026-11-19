"""Checks the projected preparation solve, tolerances and bounded audit."""
import unittest
import numpy as np

from galerkin_sylvester_tdvp import galerkin_solver as gs, witness_precond_solver as solver
from galerkin_sylvester_tdvp.models import make_model, tn, ts, mpo, witness
from galerkin_sylvester_tdvp.tangent_frame import build_frame


class PreparedSolverTests(unittest.TestCase):
    def test_exact_factorized_projected_products(self):
        rng = np.random.default_rng(843)
        n = 4
        a, b = [rng.normal(size=(2**n,2**n))+1j*rng.normal(size=(2**n,2**n)) for _ in range(2)]
        am, bm = [ts.matrix_to_mpo(v,n) for v in (a,b)]
        ref = ts.matrix_to_mpo(a@b+b@a,n)
        for mode in ('one','two'):
            frame = build_frame(make_model('E',n).root,mode)
            coords = solver.projected_sylvester(frame,solver.product_operators(am),bm)
            np.testing.assert_allclose(coords,frame.encode(ref),atol=2e-12)

    def test_dense_equation_and_original_residual_after_pcg(self):
        for family in ('D','E'):
            model = make_model(family,3)
            power = witness.prepared_power if family=='D' else witness.parity_power
            anchor = power(3,-.25)
            g = model.generator(mpo.multiply(model.root,model.root))
            expected = gs.dense_oracle(model.root,ts.matrix(g),mode='two')['B_velocity']
            for mode in ('none', 'preparation'):
                v, info = solver.solve_velocity(model.root,model.generator,lower_bound=model.floor,
                                                anchor=anchor,preconditioner=mode,rtol=1e-10,work_bond=16)
                np.testing.assert_allclose(ts.matrix(v),expected,atol=1e-11)
                self.assertEqual(info['preconditioner_calls'], 0 if mode=='none' else info['iterations'])
                self.assertLessEqual(info['galerkin_residual_rel'],1.00001e-10)
                self.assertEqual(info['residual_norm_radius'],0.)
                x, vel = ts.matrix(model.root), ts.matrix(v)
                self.assertAlmostEqual(np.linalg.norm(x@vel+vel@x-ts.matrix(g)),info['physical_residual_abs'],delta=1e-12)

    def test_zip_up_product_under_deliberate_nonzero_rounding(self):
        rng = np.random.default_rng(304)
        n = 4
        a,b = [rng.normal(size=(16,16))+1j*rng.normal(size=(16,16)) for _ in range(2)]
        for budget in (.01,75.,1000.):
            p,rad=solver.bounded_product(ts.matrix_to_mpo(a,n),ts.matrix_to_mpo(b,n),budget,2_000_000)
            self.assertLessEqual(np.linalg.norm(a@b-ts.matrix(p)),rad+5e-12)
            self.assertLessEqual(rad,budget+1e-12)
        self.assertGreater(rad,.1)

    def test_endpoint_relative_tolerance_controls_iteration_count(self):
        model = make_model('D',4)
        anchor = witness.prepared_power(4,-.25)
        kwargs = dict(lower_bound=model.floor,anchor=anchor,atol=1e-16)
        _, core = solver.solve_velocity(model.root,model.generator,rtol=1e-10,**kwargs)
        _, loose = solver.solve_velocity(model.root,model.generator,rtol=.03,**kwargs)
        self.assertLess(loose['iterations'],core['iterations'])
        self.assertLessEqual(loose['galerkin_residual_rel'],.03)
        self.assertGreater(loose['galerkin_residual_rel'],1e-10)
