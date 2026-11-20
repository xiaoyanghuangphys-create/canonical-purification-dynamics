import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from galerkin_sylvester_tdvp import witness_preconditioned_suffix_run as study


class SuffixScheduling(TestCase):
    def test_fixed_two_step_confirmations_and_option_fingerprint(self):
        def simulated(out,c,repeat):
            is_best=c['rtol']==.01
            return dict(c,repeat=repeat,status='completed',accepted_steps=2,reached_time=.1,
                        production_seconds=1. if is_best else 2.,global_cp_bound=1e-8,
                        parent_complete=True)
        with TemporaryDirectory() as tmp, patch.object(study.supplement,'run_case',side_effect=simulated):
            out=Path(tmp)/'run'
            args=SimpleNamespace(out=out,timeout=180,memory_mib=1536,resume=False,
                                 audit_rank_cap=384,diagnostic_relative_width=1e-4,allow_wider_diagnostic=True,
                                 residual_bond=24,extended_witness=True)
            study.run(args)
            rows=json.loads((out/'measurements.json').read_text())
            self.assertTrue(all(r['steps']==2 and r['n']==64 and r['cert_tol']==1e-7 for r in rows))
            confirmed=[r for r in rows if r['phase']=='confirm']
            self.assertEqual(len(confirmed),9)
            self.assertEqual({r['preconditioner'] for r in confirmed},{'none','preparation','ambient'})
            self.assertTrue(all(r['rtol']==.01 for r in confirmed if r['method']=='B'))
            self.assertTrue(all(r['diagnostic_relative_width']==1e-4 for r in rows))
            args.resume=True
            args.diagnostic_relative_width=1e-6
            with self.assertRaisesRegex(ValueError,'settings changed'):
                study.run(args)

    def test_worker_records_widest_stage_without_changing_acceptance(self):
        class Passthrough:
            def solve(self):
                radii=iter([2e-7,3e-5])
                def stage(**kwargs):
                    radius=next(radii)
                    return None,dict(residual_norm_radius=radius,generator_norm=1.,
                                     physical_residual_interval=[.01-radius,.01+radius],
                                     projected_residual_abs=.009)
                return stage
        calls=[]
        def primary_worker(cfg,out):
            for _ in range(2):
                _,diag=study.solver.solve_velocity()
                calls.append(diag)
            return dict(status='completed',global_cp_bound=3e-8)
        cfg=dict(rtol=.01,diagnostic_relative_width=1e-6,allow_wider_diagnostic=True,
                 audit_rank_cap=32)
        with TemporaryDirectory() as f, patch.object(study.solver,'solve_velocity',Passthrough().solve()), \
                patch.object(study.primary,'worker',side_effect=primary_worker):
            row=study.worker(cfg,Path(f))
            self.assertEqual(row,json.loads((Path(f)/'measurement.json').read_text()))
            self.assertFalse(row['width_met'])
            self.assertEqual(row['max_rel_radius'],3e-5)
            self.assertEqual(row['diagnostic_wide_stages'],1)
            self.assertEqual(row['global_cp_bound'],3e-8)
            self.assertEqual([r['width_met'] for r in calls],[True,False])
