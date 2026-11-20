"""Deterministic complex-coordinate, legacy, and bond-opening checks."""
import copy
import unittest

import numpy as np

from cp_tdvp import tdvp_core as legacy
from factor_tdvp import tangent_solver as ts
from galerkin_sylvester_tdvp.tangent_frame import build_frame, tangent_dimension


def random_mps(rng, ranks):
    state = ts.tn.MPS(len(ranks)-1, max_bond=max(ranks))
    state.tensors = [rng.normal(size=(l, 4, r)) + 1j*rng.normal(size=(l, 4, r))
                     for l, r in zip(ranks, ranks[1:])]
    return ts.tn.scale(state, 1/ts.tn.norm(state))


def basis_matrix(frame):
    return np.column_stack([frame.decode(e).dense() for e in np.eye(frame.dimension)])


class FrameTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(541)

    def test_ten_sites_and_irregular_complex_ranks(self):
        for ranks in [(1, *([2]*9), 1), (1, 2, 3, 1, 1), (1, 4, 4, 1), (1, 1)]:
            state = random_mps(self.rng, ranks)
            for mode in ("one", "two"):
                with self.subTest(ranks=ranks, mode=mode):
                    f = build_frame(state, mode)
                    self.assertEqual(f.ranks, ranks)
                    self.assertEqual(f.dimension, tangent_dimension(ranks, mode))
                    if len(ranks) == 11:
                        self.assertEqual(f.dimension, 108 if mode == "one" else 384)
                    for j, q in enumerate(f.complements):
                        np.testing.assert_allclose(q.conj().T @ q, np.eye(q.shape[1]), atol=2e-14)
                        if j < f.n-f.width:
                            a = f.left.tensors[j].reshape(q.shape[0], -1)
                            np.testing.assert_allclose(a.conj().T @ q, 0., atol=2e-14)
                    v = self.rng.normal(size=f.dimension)+1j*self.rng.normal(size=f.dimension)
                    np.testing.assert_allclose(f.encode(f.decode(v)), v, atol=2e-13)

    def test_dense_isometry_encode_and_legacy_projectors(self):
        # The second fixture is strictly positive with active bond two:
        # the identity plus a tensor product of local positive matrices.
        n = 3
        positive = np.eye(2**n) + np.kron(np.kron(np.diag([.4, 1.]),
                          np.array([[1., .2j], [-.2j, .6]])), np.diag([.3, .8]))
        for state in [random_mps(self.rng, (1, 2, 2, 1)), ts.matrix_to_mpo(positive, n)]:
            vec = self.rng.normal(size=4**n)+1j*self.rng.normal(size=4**n)
            target = ts.tn.from_dense(vec, n)
            for mode, original in [("one", legacy.tangent_project), ("two", legacy.tangent_project_two)]:
                with self.subTest(mode=mode):
                    f = build_frame(state, mode)
                    basis = basis_matrix(f)
                    p = basis @ basis.conj().T
                    np.testing.assert_allclose(basis.conj().T @ basis, np.eye(f.dimension), atol=3e-14)
                    np.testing.assert_allclose(f.encode(target), basis.conj().T @ vec, atol=3e-14)
                    np.testing.assert_allclose(p @ p, p, atol=3e-14)
                    np.testing.assert_allclose(p.conj().T, p, atol=3e-14)
                    out = f.project(target)
                    np.testing.assert_allclose(out.dense(), p @ vec, atol=3e-14)
                    np.testing.assert_allclose(original(state, target).dense(), out.dense(), atol=1e-13)

    def test_nonunitary_virtual_gauge_and_padded_sectors(self):
        x = random_mps(self.rng, (1, 2, 2, 1))
        gauged = copy.deepcopy(x)
        for j in range(x.n-1):
            g = np.array([[1.2, .12j], [.22+.04j, .74]])
            gauged.tensors[j] = gauged.tensors[j] @ g
            gauged.tensors[j+1] = np.tensordot(np.linalg.inv(g), gauged.tensors[j+1], axes=1)
        padded = ts.tn.add(x, x, alpha=.25, beta=.75)
        target = random_mps(self.rng, (1, 4, 4, 1))
        for mode in ("one", "two"):
            f = build_frame(x, mode)
            for y in (gauged, padded):
                alternate = build_frame(y, mode)
                self.assertEqual(f.ranks, alternate.ranks)
                self.assertLess(alternate.metadata["rank_discard_bound"], 1e-13)
                if y is padded:
                    self.assertNotEqual(f.ranks, alternate.metadata["stored_ranks"])
                np.testing.assert_allclose(f.project(target).dense(), alternate.project(target).dense(), atol=1e-13)

    def test_bond_opening_at_adjacent_pair(self):
        x = ts.tn.vacuum(3)
        direction = ts.tn.vacuum(3)
        # |11><11| on the first two qubits, |0><0| on the third.
        direction.tensors[:2] = [np.eye(4)[3].reshape(1, 4, 1)]*2
        self.assertLess(ts.tn.norm(build_frame(x).project(direction)), 1e-14)
        np.testing.assert_allclose(build_frame(x, "two").project(direction).dense(), direction.dense(), atol=1e-14)
        perturbed, _ = ts.tn.compress(ts.tn.add(x, direction, beta=1e-4), 4)
        self.assertEqual(build_frame(perturbed).ranks, (1, 2, 1, 1))

    def test_adjoint_involution_and_hermitian_operator_interleaving(self):
        # The tangent of a Hermitian MPS is closed under Hermitian adjoint.
        root = ts.mpo.product_root(4, polarization=.41)
        for mode in ("one", "two"):
            f = build_frame(root, mode)
            v = self.rng.normal(size=f.dimension) + 1j*self.rng.normal(size=f.dimension)
            mapped = f.decode(v)
            flipped = ts.mpo.adjoint(mapped)
            np.testing.assert_allclose(ts.matrix(flipped), ts.matrix(mapped).conj().T, atol=1e-13)
            jv = f.encode(flipped)
            np.testing.assert_allclose(f.decode(jv).dense(), flipped.dense(), atol=1e-13)
            np.testing.assert_allclose(f.encode(ts.mpo.adjoint(f.decode(jv))), v, atol=1e-13)

    def test_small_schmidt_values_and_invalid_inputs(self):
        x, y = ts.tn.vacuum(2), ts.tn.vacuum(2)
        for t in y.tensors:
            t[:, 0, :] = 0
            t[:, 3, :] = 1
        z = ts.tn.add(x, y, beta=1e-10)
        self.assertEqual(build_frame(z).ranks, (1, 2, 1))
        trimmed = build_frame(z, rank_rtol=1e-8)
        self.assertEqual(trimmed.ranks, (1, 1, 1))
        self.assertAlmostEqual(trimmed.metadata["rank_discard_bound"], 1e-10, delta=1e-22)
        for invalid in ("bad", "", None):
            with self.assertRaises(ValueError):
                build_frame(x, invalid)
        with self.assertRaises(ValueError):
            build_frame(ts.tn.scale(x, 0))
        with self.assertRaises(ValueError):
            trimmed.decode(np.zeros(trimmed.dimension+1))
        with self.assertRaises(ValueError):
            trimmed.encode(ts.tn.vacuum(3))


if __name__ == "__main__":
    unittest.main()
