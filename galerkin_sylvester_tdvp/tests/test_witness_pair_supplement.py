"""Check supplemental scheduling, process-local injections and independent provenance."""
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
import json

from galerkin_sylvester_tdvp import witness_pair_supplement as supp, witness_streaming


class SupplementTests(unittest.TestCase):
    def test_worker_changes_only_intended_callbacks(self):
        orig=supp.paired.solve_velocity
        orig_floor=supp.witness_stage.make_stage_floor
        orig_audit=supp.paired.witness.WitnessAudit
        for method,variant in [('A','streamed-block'),('B','block'),('B','streamed-block')]:
            def inspect(config,folder):
                self.assertIs(supp.paired.solve_velocity,witness_streaming.solve_velocity
                              if method=='B' and variant=='streamed-block' else orig)
                self.assertIs(supp.witness_stage.make_stage_floor,supp.make_block_stage_floor
                              if method=='B' else orig_floor)
                self.assertEqual(supp.paired.witness.WitnessAudit(2).residual_bond,32)
            with patch.object(supp.paired,'worker',inspect):
                supp.worker(dict(method=method,variant=variant,residual_bond=32),None)
            self.assertIs(supp.paired.solve_velocity,orig)
            self.assertIs(supp.witness_stage.make_stage_floor,orig_floor)
            self.assertIs(supp.paired.witness.WitnessAudit,orig_audit)

    def test_extensions_apply_only_at_tight_d_tolerances(self):
        for enabled in (False,True):
            args=Namespace(timeout=180,memory_mib=1536,extended_witness=enabled,residual_bond=32)
            for family in ('D','E'):
                for tol in (1e-5,1e-7,1e-8):
                    active=enabled and family=='D' and tol<=1e-7
                    for method in ('A','B'):
                        cfg=supp.configuration(family,64,method,2,tol,args)
                        levels=supp.primary.configuration(family,64,method,2,tol,args)['levels']
                        self.assertEqual(cfg['levels'],levels+([[10,14,20],[12,16,24]] if active else []))
                        self.assertEqual(cfg['residual_bond'],32 if active else 24)
                        self.assertEqual(cfg['extended_witness_active'],active)

    def test_shared_screen_and_paired_confirmations(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'supplement'
            args=Namespace(out=out,resume=False,sizes=[64],families=['D'],steps=[1,2],
                           tolerances=[1e-5,1e-7],variant='streamed-block',timeout=180,memory_mib=1536,
                           extended_witness=True,residual_bond=32)
            def fake_worker(out,c,rep):
                speed=c['steps'] if c['method']=='A' else 1/c['steps']
                return dict(c,repeat=rep,status='completed',reached_time=c['total'],
                            global_cp_error=None,global_cp_bound=5e-8,production_seconds=speed,
                            total_seconds=1+speed,scoring_seconds=.1,wall_seconds=2+speed)
            with patch.object(supp,'run_case',fake_worker):
                supp.run(args)
            grid=json.loads((out/'grid.json').read_text())
            rows=json.loads((out/'measurements.json').read_text())
            for r in grid+rows:
                self.assertEqual(r['levels'][-1],[12,16,24] if r['cert_tol']<=1e-7 else [8,12,16])
                self.assertEqual(r['residual_bond'],32 if r['cert_tol']<=1e-7 else 24)
                self.assertTrue(r['extended_witness'])
            self.assertEqual(len([r for r in rows if r['phase']=='screen']),8)
            confirms=[r for r in rows if r['phase']=='confirm']
            self.assertEqual(len({r['steps'] for r in confirms}),2)
            self.assertEqual({r['method'] for r in confirms},{'A','B'})
            self.assertEqual({r['repeat'] for r in confirms},{1,2,3})
            from galerkin_sylvester_tdvp.witness_pair_analysis import summarize_configs, frontiers
            configs=summarize_configs(grid,rows,json.loads((out/'environment.json').read_text()))
            for r in configs:
                self.assertTrue(r['qualified_endpoints'],r)
            self.assertTrue(all(r['matched'] for r in frontiers(configs)[0]))
            self.assertIn('galerkin_sylvester_tdvp/witness_streaming.py',rows[0]['source_files_sha256'])
