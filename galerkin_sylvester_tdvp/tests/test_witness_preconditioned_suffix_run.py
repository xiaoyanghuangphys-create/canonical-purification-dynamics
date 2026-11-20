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
                                 audit_rank_cap=384,diagnostic_relative_width=1e-4,
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
