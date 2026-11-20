import json
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from galerkin_sylvester_tdvp import witness_preconditioned_run as study


class StudyTests(TestCase):
    def test_paired_fallback_and_same_tolerance_control(self):
        def simulate(out, c, rep):
            failed = c['steps']==2 and c['preconditioner']=='preparation'
            return dict(c, repeat=rep, status='failed' if failed else 'completed',
                        reached_time=0. if failed else c['total'], accepted_steps=c['steps'],
                        global_cp_error=1e-8, global_cp_bound=1e-7,
                        production_seconds=1.0 if c['rtol']==0.01 else 2.0)

        with TemporaryDirectory() as tmp, patch.object(study,'source_hashes',return_value={'test':'fixed'}), \
                patch.object(study.supplement,'run_case',side_effect=simulate):
            out=Path(tmp)/'cohort'
            args=Namespace(out=out, resume=False, sizes=[8], families=['E'], targets=[1e-5],
                           steps=2, one_step_all=False, timeout=10., memory_mib=512., extended_witness=True,residual_bond=24)
            study.run(args)
            rows=json.loads((out/'measurements.json').read_text())
            ids=[(r['config_id'],r['phase'],r['repeat']) for r in rows]
            self.assertEqual(len(ids),len(set(ids)))
            confirm=[r for r in rows if r['phase']=='confirm' and r['steps']==1]
            self.assertEqual(len(confirm),9)
            for mode in ('preparation','none'):
                self.assertEqual({r['rtol'] for r in confirm if r['preconditioner']==mode},{0.01})
            selection=json.loads((out/'selection.json').read_text())
            self.assertEqual(set(selection['E08_s2_tol1e-05']),{'A'})

    def test_endpoint_requires_target_bound_even_if_measured_score_is_small(self):
        row=dict(status='completed',family='E',n=8,steps=2,accepted_steps=2,global_cp_error=1e-8,
                 reached_time=1.,total=1.,production_seconds=2.,cert_tol=1e-5,global_cp_bound=1e-2)
        self.assertFalse(study.valid(row))
        self.assertTrue(study.valid(dict(row,global_cp_bound=1e-6)))
