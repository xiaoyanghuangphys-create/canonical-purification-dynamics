"""Integration, rank/branch diagnostics, reference independence and failure tests."""
import unittest

import numpy as np
from scipy.linalg import expm

from factor_tdvp.tangent_solver import matrix, matrix_to_mpo, mpo, tn
from galerkin_sylvester_tdvp.evolution import (EvolutionError, KrausStartup, PositiveWitness,
                                               integrate, root_diagnostics)


def exact_root(rho):
    values, vectors = np.linalg.eigh(rho)
    return (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.conj().T


def dephase(rho):
    return mpo.onsite_dissipation(rho, np.full(rho.n, .5), "dephasing")


class TestEvolution(unittest.TestCase):
    def test_coherent_refinement_and_density(self):
        h = np.array([[.2, .7j],[-.7j,-.2]])
        x = exact_root(np.array([[.65, .22], [.22, .35]]))
        initial = matrix_to_mpo(x, 1)
        h_mpo = matrix_to_mpo(h, 1)
        def coherent(rho):
            return tn.add(mpo.multiply(h_mpo,rho), mpo.multiply(rho,h_mpo), -1j, 1j)
        final = .25
        u = expm(-1j*final*h)
        exact = u @ x @ u.conj().T
        errors = []
        for steps in [2,4,8]:
            end, info = integrate(initial, coherent, np.linspace(0, final, steps+1), 1)
            self.assertEqual(info["status"], "success", info["history"][-1])
            errors.append(np.linalg.norm(matrix(end)-exact))
            last = info["history"][-1]["endpoint"]
            self.assertLess(last["trace_defect"], 2e-12)
            self.assertLess(last["polar_defect"], 2e-12)
        self.assertGreater(min(np.array(errors[:-1])/errors[1:]), 3.7)
        self.assertLess(np.linalg.norm(matrix(end) @ matrix(end) - exact @ exact), 1e-4)

    def test_step_doubling_rejects_retries_and_preserves_last_state(self):
        initial = mpo.product_root(1, .75)
        _, blocked = integrate(initial,dephase,[0,.5],1,step_doubling=True,step_tol=1e-14,
                               max_halvings=0)
        self.assertEqual(blocked["status"], "step_error_budget")
        self.assertEqual(blocked["accepted_steps"], 0)
        self.assertGreater(blocked["history"][0]["step_doubling_estimate"], 1e-14)
        self.assertIn("traceback", blocked["history"][0])
        end, passed = integrate(initial,dephase,[0,.5],1,step_doubling=True,step_tol=1e-4)
        self.assertEqual(passed["status"],"success",passed["history"][-1])
        self.assertGreater(passed["rejected_steps"],0)
        exact = matrix(mpo.product_root(1,.75*np.exp(-.5)))
        self.assertLess(np.linalg.norm(matrix(end)-exact),3e-4)

    def test_rank_opening_and_small_populations(self):
        sm = np.array([[0.,1.],[0.,0.]])
        seed = KrausStartup(np.zeros((2,2)), [sm])
        excited = matrix_to_mpo(np.diag([0.,1.]),1)
        def loss(rho):
            return mpo.onsite_dissipation(rho, [1.], "loss")
        seed_time = 1e-18
        early, info = integrate(excited,loss,[0,seed_time],1,clock="sqrt",seed=seed)
        self.assertEqual(info["status"],"success",info)
        self.assertAlmostEqual(matrix(early)[0,0].real, np.sqrt(-np.expm1(-seed_time)), places=14)
        self.assertGreater(info["history"][0]["initial_support"]["kernel_forcing"], .99)
        self.assertLess(abs(info["history"][0]["seed_raw_min_eigenvalue"]-1e-9),1e-14)
        _, blocked = integrate(excited,loss,[0,.01],1,clock="sqrt")
        self.assertEqual(blocked["status"], "rank_support_uncertain")
        self.assertGreater(blocked["history"][0]["details"]["kernel_forcing"],.99)
        errors = []
        for intervals in (4,8):
            times = np.linspace(0,np.sqrt(.04),intervals+1)**2
            end, info = integrate(excited,loss,times,1,clock="sqrt",seed=seed)
            self.assertEqual(info["status"],"success",info["history"][-1])
            self.assertEqual(info["history"][0]["kind"],"seed")
            exact = np.diag([np.sqrt(-np.expm1(-.04)),np.exp(-.02)])
            errors.append(np.linalg.norm(matrix(end)-exact))
        self.assertLess(errors[-1],errors[0])
        self.assertLess(errors[-1],5e-5)

    def test_seed_scope_and_branch(self):
        x = matrix_to_mpo(np.diag([1.,0.]),1)
        with self.assertRaises(EvolutionError):
            KrausStartup(np.zeros((2,2)),[np.diag([1.,.3])])(x, .01)
        with self.assertRaisesRegex(EvolutionError, "unresolved_seed_channel_rate"):
            KrausStartup(np.zeros((2,2)), [1e-8*mpo.SM])(x, .01)
        signed = matrix_to_mpo(np.diag([.8,-.6]), 1)
        diag = root_diagnostics(signed)
        self.assertAlmostEqual(diag["negative_part"],.6)
        self.assertAlmostEqual(diag["polar_defect"],1.2)
        _, fail=integrate(signed,dephase,[0,.1],1,polar=True)
        self.assertEqual(fail["status"],"negative_root")
        self.assertEqual(fail["accepted_steps"],0)
        self.assertFalse(fail["initial_accepted"])

    def test_warm_start_mpo_only_and_structural_floor(self):
        x = mpo.product_root(3,.2)
        times = [0.,.01,.02]
        runs = []
        floor = ((1-.2)/2)**(x.n/2)
        for warm in (False,True):
            state, info = integrate(x,dephase,times,2,dense_audit_max_n=0,initial_lower_bound=floor,
                                    warm_start=warm)
            self.assertEqual(info["status"],"success",info["history"][-1])
            self.assertIsNone(info["history"][-1]["endpoint"]["min_eigenvalue"])
            runs.append(state)
        self.assertLess(tn.distance(*runs),2e-10)
        _, fail = integrate(x,dephase,times,2,dense_audit_max_n=0)
        self.assertEqual(fail["status"],"rank_support_uncertain")
        # Valid lower bound on Hermitian part measured in this test by dense oracle.
        def structural(stage):
            a = matrix(stage)
            return float(np.linalg.eigvalsh((a+a.conj().T)/2)[0] - 1e-13)
        _, info = integrate(x,dephase,times,2,dense_audit_max_n=0,stage_lower_bound=structural)
        self.assertEqual(info["status"],"success",info["history"][-1])
        self.assertEqual(info["history"][-1]["endpoint"]["floor_source"],"structural_stage_bound")

    def test_two_site_cap_and_global_error_despite_stationarity(self):
        z = np.diag([1.,-1.])
        h = np.kron(z,z)
        h_mpo = matrix_to_mpo(h,2)
        x = mpo.product_root(2,.65)
        def rhs(rho):
            return tn.add(mpo.multiply(h_mpo,rho),mpo.multiply(rho,h_mpo),-1j,1j)
        u = expm(-.04j*h)
        reference = u @ matrix(x) @ u.conj().T
        errors = {}
        for cap in (1,4):
            end, info = integrate(x,rhs,[0.,.02,.04],cap,mode="two")
            self.assertEqual(info["status"],"success",info["history"][-1])
            errors[cap] = np.linalg.norm(matrix(end)-reference)
            if cap == 1:
                self.assertGreater(info["history"][-1]["retractions"][-1]["svd_discard_norm"],1e-4)
                self.assertLess(max(s["galerkin_residual_abs"] for row in info["history"]
                                    for s in row["solves"]),1e-7)
        self.assertGreater(errors[1],1e-2)
        self.assertLess(errors[4],errors[1]/1000)

    def test_witness_certifies_or_rejects(self):
        import scalable_cp_witness as sw
        x = sw.parity_power(2,.5)
        def run(tol, iterations):
            witness = PositiveWitness(sw.WitnessAudit.parity_chain(2),sw.parity_power(2,.25),
                                      4,8,tol,iterations)
            return integrate(x,dephase,[0.,.15],4,mode="two",witness=witness)
        state, passed = run(1e-4,8)
        self.assertEqual(passed["status"],"success",passed["history"][-1])
        check = passed["history"][0]["witness"]
        exact = sw.parity_power(2,.5,transverse=.4*np.exp(-.15))
        self.assertLessEqual(tn.distance(state,exact),check["cp_bound"]+5e-12)
        self.assertLessEqual(check["cp_bound"],1e-4)
        _, failed = run(1e-12,0)
        self.assertEqual(failed["status"],"witness_uncertified")
        self.assertGreater(failed["history"][-1]["details"]["cp_bound"],1e-12)


if __name__ == "__main__":
    unittest.main()
