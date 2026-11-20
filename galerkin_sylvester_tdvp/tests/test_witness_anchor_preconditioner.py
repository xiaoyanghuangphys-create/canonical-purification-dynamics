"""Dense comparison and normalized-tangent positivity of the preparation map."""
import unittest

import numpy as np

from galerkin_sylvester_tdvp.models import make_model, ts, witness
from galerkin_sylvester_tdvp.tangent_frame import build_frame
from galerkin_sylvester_tdvp.witness_anchor_preconditioner import preparation_preconditioner


class PreparationPreconditionerTests(unittest.TestCase):
    def test_against_dense_congruence_on_complex_coordinates(self):
        rng = np.random.default_rng(314)
        for family in ('D', 'E'):
            for n in (1, 3, 5):
                root = make_model(family, n).root
                anchor = (witness.prepared_power(n, -.25) if family == 'D'
                          else witness.parity_power(n, -.25))
                w = ts.matrix(anchor)
                for mode in ('one', 'two'):
                    frame = build_frame(root, mode)
                    normal = frame.encode(root)
                    normal /= np.linalg.norm(normal)
                    scale = 2 * ts.matrix(root).trace().real / 2**n
                    fn = preparation_preconditioner(frame, anchor, normal, scale)
                    vectors = rng.normal(size=(2, frame.dimension)) + 1j*rng.normal(size=(2, frame.dimension))
                    u, v = vectors
                    qv = v - normal*np.vdot(normal, v)
                    y = ts.matrix(frame.decode(qv))
                    exact = frame.encode(ts.matrix_to_mpo(w @ y @ w, n))
                    exact = .5 * scale * (exact - normal*np.vdot(normal, exact))
                    np.testing.assert_allclose(fn(v), exact, atol=2e-12, rtol=2e-12)
                    np.testing.assert_allclose(fn(u + 1j*v), fn(u) + 1j*fn(v), atol=4e-12)
                    self.assertAlmostEqual(np.vdot(u, fn(v)), np.vdot(fn(u), v), delta=2e-10)
                    self.assertGreater(np.vdot(qv, fn(qv)).real, 0.)
                    self.assertLess(np.linalg.norm(fn(normal)), 2e-12)
