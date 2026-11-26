"""Check stored amplitudes and evolved physical data against separate references."""
import unittest
from unittest.mock import patch

import numpy as np
from parity_local_lift_comparison import oracles as o
from parity_local_lift_comparison.local import LocalLift, fit, hp, site_window
from parity_local_lift_comparison.local_controls import run_case


class LocalKernelTests(unittest.TestCase):
    def test_half_petz_native_doubled_amplitude_and_grid(self):
        p, b = .6, .4
        predicted = o.half_petz_metrics(p, b, b*np.exp(-1.))
        for n in (2, 3, 4, 6, 8):
            rho = o.dense_density(n, p, b)
            y = o.dense_power(rho, .5)
            for width in sorted({1, min(3, n-1)}):
                dense, _ = o.dense_hp_update(y, rho, 1., width)
                for steps in (1, 3):
                    with self.subTest(n=n, width=width, steps=steps):
                        lift = LocalLift(o.root_mpo(n, p, b), "Half-Petz", width)
                        trunc = sum(lift.step(1./steps)["truncation_bound"] for _ in range(steps))
                        np.testing.assert_allclose(lift.state.dense(), o.interleaved(dense), rtol=0, atol=2e-13)
                        self.assertLess(trunc, 2e-12)
                        self.assertLess(lift.cache_discarded_mass, 1e-12)
                        report = o.score(lift.state, 1.)
                        for measured, target in (("global_cp_error", "raw_error"), ("norm", "norm"),
                                                 ("root_qp_re", "qp_coefficient"), ("hermiticity_defect", "hermiticity_defect")):
                            self.assertAlmostEqual(report[measured], predicted[target], delta=2e-13)
                        self.assertGreater(report["hermiticity_defect"], 1e-3)

    def test_generic_complex_input_and_independent_cache(self):
        n, width = 4, 2
        rho = o.dense_density(n, .37, .51)
        phase = np.kron(np.diag([1., np.exp(.47j)]), np.eye(2**(n-1)))
        rho = phase @ rho @ phase.conj().T
        y = o.dense_power(rho, .5)
        initial = o.tn.from_dense(o.interleaved(y), n)
        lift = LocalLift(initial, "Half-Petz", width)
        # Once initialized, a step must not contract physical input from the
        # evolving amplitude, nor evaluate an analytic model reference.
        with patch.object(hp, "physical_factor", side_effect=AssertionError), \
             patch.object(o, "root_mpo", side_effect=AssertionError):
            for h in (.13, .39, .48):
                lift.step(h)
                y, rho = o.dense_hp_update(y, rho, h, width)
        np.testing.assert_allclose(lift.state.dense(), o.interleaved(y), rtol=0, atol=1e-13)
        self.assertGreater(np.linalg.norm(y.imag), .01)
        np.testing.assert_allclose(lift.windows[0,width], o.marginal(rho, range(width)), atol=1e-14)

    def test_fit_rank_completion_and_whole_controls(self):
        for width, region in ((2, 2), (2, 3), (2, 5), (2, 6), (6, 6)):
            with self.subTest(width=width, region=region):
                checks = run_case(dict(kind="fit-data", n=6, width=width, fit_width=region))["checks"]
                self.assertLess(checks["max_physical_cache_error"], 1e-13)
                for key in ("native_local_residual", "alternate_local_residual"):
                    if width == 2 and region == 6:
                        self.assertAlmostEqual(checks[key], o.unitary_floor(.6, .4, .4*np.exp(-1)), delta=5e-13)
                    else:
                        self.assertLess(checks[key], 8e-13)
                if region < 6:
                    self.assertEqual(checks["source_rank"], 1)
                    self.assertEqual(checks["cross_rank"], 1)
                    self.assertEqual(checks["reduced_cross_dim"], 2)
                    self.assertLess(checks["fit_cache_difference_p_vs_minus_p"], 1e-13)
                    self.assertGreater(checks["global_cp_difference_p_vs_minus_p"], .65)
                    self.assertAlmostEqual(checks["global_cp_error"], .01928021377671, delta=2e-12)
                elif width < 6:
                    self.assertEqual(checks["cross_rank"], 2)
                    self.assertEqual(checks["reduced_cross_dim"], 3)
                    self.assertAlmostEqual(checks["global_cp_error"], o.unitary_floor(.6, .4, .4*np.exp(-1)), delta=5e-13)
                else:
                    self.assertLess(checks["global_cp_error"], 5e-13)
        for n in (2, 3, 4, 6):
            lift = LocalLift(o.root_mpo(n, .6, .4), "Half-Petz", n)
            for _ in range(2):
                lift.step(.5)
            self.assertLess(o.score(lift.state, 1.)["global_cp_error"], 5e-13)

    def test_full_sweep_inactive_motion_and_correct_caches(self):
        for method, region in (("Half-Petz", 2), ("Local-fit", 4)):
            with self.subTest(method=method):
                n = 6
                initial = o.root_mpo(n, .6, .4)
                sweep = LocalLift(initial, method, 2, region, "uniform", audit_inactive=True)
                active = LocalLift(initial, method, 2, region, "active")
                skip = LocalLift(initial, method, 2, region, "uniform-skip")
                for h in (.4, .6):
                    active.step(h)
                    info = sweep.step(h)
                    s = skip.step(h)
                    self.assertEqual(info["applied_site_updates"], n)
                    self.assertEqual(s["skipped_site_updates"], n-1)
                    self.assertLess(info["max_inactive_motion"], 1e-13)
                    self.assertLess(o.tn.distance(sweep.state, active.state), 1e-13)
                    self.assertLess(o.tn.distance(skip.state, active.state), 1e-13)
                rho = o.dense_density(n, .6, .4*np.exp(-1.))
                for (edge, size), actual in sweep.windows.items():
                    np.testing.assert_allclose(actual, o.marginal(rho, range(edge, edge+size)), atol=3e-14)
                # A proper fixed support preserves Schmidt weights even after
                # repeated fits; the full-sweep equality above is numerical.
                if method == "Local-fit":
                    self.assertGreater(o.score(active.state, 1.)["global_cp_error"], o.unitary_floor(.6,.4,.4*np.exp(-1.)))

    def test_stationary_alt_completion_and_explicit_identity(self):
        initial = o.root_mpo(5, .6, 0.)
        for completion, protocol, expected in (("native", "uniform", 0.),
                                               ("alternate", "uniform", np.sqrt(.4)),
                                               ("alternate", "uniform-skip", 0.)):
            lift = LocalLift(initial, "Local-fit", 2, 3, protocol, completion=completion,
                             audit_inactive=True, inactive_sites=range(5))
            result = lift.step(1.)
            self.assertAlmostEqual(o.tn.distance(initial, lift.state), expected, delta=1e-13)
            if protocol == "uniform":
                self.assertAlmostEqual(result["max_inactive_motion"], expected, delta=1e-13)
                self.assertLess(result["max_local_fit"], 1e-13)
            else:
                self.assertEqual(result["applied_site_updates"], 0)

    def test_stationary_and_product_controls(self):
        for n, p, b in ((3,.6,0.), (8,0.,.4)):
            for method in ("Half-Petz", "Local-fit"):
                lift = LocalLift(o.root_mpo(n, p, b), method, 2, protocol="uniform")
                for _ in range(2):
                    lift.step(.5)
                self.assertLess(o.score(lift.state, 1., p, b)["global_cp_error"], 1e-12)

    def test_existing_cluster_positive_control(self):
        for method, width, fit_width in (("Half-Petz", 3, 3), ("Local-fit", 3, 7)):
            row = run_case(dict(kind="cluster", method=method, n=8, width=width,
                                fit_width=fit_width, intervals=2, horizon=.6))
            self.assertLess(row["checks"]["global_cp_error"], 2e-9)
            self.assertTrue(all(step["applied_site_updates"] == 8 for step in row["steps"]))

    def test_preallocation_guard_and_windows(self):
        init = o.root_mpo(32, .6, .4)
        with patch.object(hp, "physical_factor", side_effect=AssertionError):
            with self.assertRaises(MemoryError):
                LocalLift(init, "Local-fit", 12, 32)
            with self.assertRaises(MemoryError):
                LocalLift(init, "Half-Petz", 24)
        for n in range(2, 12):
            for w in range(1, n+1):
                for r in range(w, n+1):
                    for site in range(n):
                        a, _ = site_window(site, w, n)
                        b, _ = site_window(site, r, n)
                        self.assertTrue(b <= a <= site < a+w <= b+r <= n)


if __name__ == "__main__":
    unittest.main()
