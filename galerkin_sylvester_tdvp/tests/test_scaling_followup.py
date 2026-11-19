"""Independent dense checks of the optional large-chain bounds."""
import unittest
import numpy as np
from factor_tdvp import tangent_solver as ts
from galerkin_sylvester_tdvp.models import make_model, witness, tn
from galerkin_sylvester_tdvp.scaling_followup import AnchoredFloor, endpoint_bound, followup_grid


class GroupedStageAudit(unittest.TestCase):
    def test_three_branches_and_unresolved_positivity(self):
        model = make_model("D", 4)
        ref, density = model.reference(.1)
        reference = ts.matrix(ref)
        for excess, wanted in [(1e-4, "frobenius"), (.34, "grouped"), (.8, "congruence"), (1.1, "grouped")]:
            with self.subTest(excess=excess):
                record = []
                audit = AnchoredFloor(model.root, model.floor, witness.prepared_power(4, -.25),
                                     width=2, record=record.append)
                candidate = tn.scale(ref, 1+excess)
                floor = audit(candidate)
                x = ts.matrix(candidate)
                self.assertLessEqual(floor, np.linalg.eigvalsh(x)[0]+2e-14)
                self.assertEqual(record[0]["source"], wanted)
                if excess >= 1:
                    self.assertLess(floor, 0)
                    with self.assertRaises(ValueError):
                        endpoint_bound(candidate, density, floor, model.floor)
                else:
                    self.assertGreater(floor, 0)
                    bound = endpoint_bound(candidate, density, floor, model.floor)
                    self.assertGreaterEqual(bound["global_cp_bound"]+4e-13, np.linalg.norm(x-reference))

    def test_nonhermitian_movement_and_aggressive_discard(self):
        n=3
        model = make_model("D", n)
        rho = ts.matrix(model.root)
        rng = np.random.default_rng(718)
        delta = rng.normal(size=rho.shape) + 1j*rng.normal(size=rho.shape)
        delta *= .52/np.linalg.norm(delta)
        candidate = ts.matrix_to_mpo(rho+delta,n)
        history = []
        anchor = AnchoredFloor(model.root,model.floor,witness.prepared_power(n,-.25),bond=1,width=1,record=history.append)
        floor = anchor(candidate)
        actual = np.linalg.eigvalsh(rho+(delta+delta.conj().T)/2)[0]
        self.assertLessEqual(floor, actual + 3e-14)
        self.assertGreater(history[-1]["initial_delta_discard"], .01)
        self.assertGreaterEqual(history[-1]["grouped_movement_bound"]+2e-14,np.linalg.norm(delta,2))
        neg = anchor(tn.scale(model.root,-1))
        self.assertLess(neg, 0)

    def test_endpoint_with_skew_and_prespecified_sizes(self):
        model = make_model("D", 3)
        reference, density = model.reference(.1)
        a = ts.matrix(reference)
        pert = np.diag(.014j*np.arange(len(a))/len(a))
        z = ts.matrix_to_mpo(a+pert,3)
        floor = np.linalg.eigvalsh(a)[0]
        bound = endpoint_bound(z, density, floor, model.floor)
        self.assertGreater(bound["positive_bound_skew"], 0)
        self.assertGreaterEqual(bound["global_cp_bound"]+1e-14, np.linalg.norm(pert))
        grid = followup_grid()
        self.assertEqual(len(grid),8)
        self.assertEqual({r["n"] for r in grid}, {8,16,32,64})
        self.assertTrue(all(r["work"]==64 if r["family"]=="D" else r["steps"]==8 for r in grid))
