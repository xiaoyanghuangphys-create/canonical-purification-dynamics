"""Check the production generator against an independent dense physical definition."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from galerkin_sylvester_tdvp.benchmarks import config, execute, grid, score
from galerkin_sylvester_tdvp.models import make_model, parity_floor_setup, ts


class PhysicalModelTests(unittest.TestCase):
    def test_generators_and_reference_derivatives(self):
        rng=np.random.default_rng(922)
        q=rng.normal(size=(8,8))+1j*rng.normal(size=(8,8))
        rho=q@q.conj().T
        rho/=np.trace(rho)
        for name in ('D','E','cluster','product','commuting','coherent','loss_mixed','pair_pure'):
            with self.subTest(family=name):
                m=make_model(name,3)
                h,jumps=m.dense_physics()
                actual=ts.matrix(m.generator(ts.matrix_to_mpo(rho,3)))
                exact=-1j*(h@rho-rho@h)
                for jump in jumps:
                    a=jump.conj().T@jump
                    exact+=jump@rho@jump.conj().T-.5*(a@rho+rho@a)
                np.testing.assert_allclose(actual,exact,atol=2e-14)
                dt=2e-5
                r1,r2=m.reference(.03+dt)[1],m.reference(.03-dt)[1]
                approx=(ts.matrix(r1)-ts.matrix(r2))/(2*dt)
                np.testing.assert_allclose(approx,ts.matrix(m.generator(m.reference(.03)[1])),atol=2e-9)
                s=score(m.root,m,0,name)
                self.assertLess(s['global_cp_error'],5e-8)
                self.assertLess(s['density_error'],1e-12)

    def test_parity_stage_audit_off_subspace(self):
        m=make_model('E',3)
        a=ts.matrix(m.root)
        perturb=ts.matrix(make_model('product',3).root)*.001
        s=ts.matrix_to_mpo(a+perturb,3)
        self.assertLessEqual(parity_floor_setup(3)(s),np.linalg.eigvalsh(a+perturb)[0]+2e-15)

    def test_worker_budgets_are_failures_with_absent_score(self):
        for timeout,memory,expected in ((.001,512,'timeout'),(10.,.01,'memory_limit')):
            with self.subTest(expected=expected),tempfile.TemporaryDirectory() as out:
                c=config('D',3,'B_two',steps=20)
                c.update(memory_mib=memory,repeat=1)
                r=execute(c,Path(out),timeout)
                self.assertEqual(r['status'],expected)
                self.assertIsNone(r['global_cp_error'])
                self.assertEqual(json.loads((Path(out)/'measurement.json').read_text())['status'],expected)

class EvidenceTests(unittest.TestCase):
    def test_frontier_requires_all_repeats_at_the_final_time(self):
        from galerkin_sylvester_tdvp.analyze_results import frontier
        c=config('E',3,'B_one',repeats=3)
        c.update(status='completed',reached_time=.1,global_cp_error=0.000002,total_seconds=.07,
                 wall_seconds=.22,setup_seconds=.01,integration_seconds=.05,peak_rss_mib=88)
        rows=[dict(c,repeat=i) for i in (1,2,3)]
        def pass_at(rows):
            return [w['target'] for w in frontier(rows)[1] if w['attained']]
        self.assertEqual(pass_at(rows),[1e-3,1e-5])
        for override in (dict(status='timeout'),dict(reached_time=.09),dict(global_cp_error=None),
                         dict(global_cp_error=float('nan')),dict(repeat=2)):
            self.assertEqual(pass_at(rows[:2]+[dict(rows[2],**override)]),[])
