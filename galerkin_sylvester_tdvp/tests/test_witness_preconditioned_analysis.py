"""Selection and timing evidence for target-tuned Galerkin witness benchmarks."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from galerkin_sylvester_tdvp.tests.test_witness_pair_analysis import cfg, observations
from galerkin_sylvester_tdvp.witness_pair_analysis import summarize_configs
from galerkin_sylvester_tdvp.witness_preconditioned_analysis import check_trace, comparison, frontiers, tighter_references


def setting(method, pc, rtol, phase, elapsed, fail=False):
    c = dict(cfg(method, phase=phase), rtol=rtol, atol=1e-18, residual_bond=24, preconditioner=pc)
    c['config_id'] = f'{method}_{pc}_{rtol}'
    records = observations(c, elapsed)
    if fail:
        records[0].update(status='memory_budget', global_cp_bound=None, global_cp_error=None)
    e = summarize_configs([c], records, {})[0]
    e.update(role='A' if method == 'A' else 'B_pre' if pc == 'preparation' else 'B_none', residual_bond=24)
    return e


class TargetTuningAnalysisTests(unittest.TestCase):
    def test_select_screen_first_and_compare_same_tolerance_control(self):
        cases = [('A','ambient',1e-9,1.,3.), ('B','preparation',.02,2.,4.),
                 ('B','none',.02,1.7,3.5)]
        screen = [setting(m,pc,t,'screen',s) for m,pc,t,s,c in cases]
        confirm = [setting(m,pc,t,'confirm',c) for m,pc,t,s,c in cases]
        strict = setting('B','preparation',1e-9,'screen',3.)
        missed = setting('B','preparation',.1,'screen',.1,fail=True)
        nominations = {'group':{c['role']:c['config_id'] for c in screen}}
        rows, strict_rows = comparison(screen + [strict, missed] + confirm, nominations)
        self.assertEqual(len(rows),1)
        self.assertAlmostEqual(rows[0]['B_pre_over_A'],4/3)
        self.assertAlmostEqual(rows[0]['B_pre_over_B_none'],4/3.5)
        self.assertEqual(rows[0]['accuracy_kind'],'global_cp_bound')
        self.assertEqual(rows[0]['B_pre_confirmed_repeats'],3)
        self.assertEqual(strict_rows[0]['phase'],'screen')

        with self.assertRaisesRegex(ValueError, 'fastest qualifying'):
            bad_nom = dict(nominations['group'], B_pre=strict['config_id'])
            comparison(screen+[strict], {'group':bad_nom})
        with self.assertRaisesRegex(ValueError, 'solver budget'):
            other = setting('B','none',1e-9,'screen',1.)
            bad_nom = dict(nominations['group'], B_none=other['config_id'])
            comparison(screen+[other], {'group':bad_nom})
        # A failed repeat leaves the confirmed comparison unavailable.
        fail_confirm = setting('B','preparation',.02,'confirm',4.,fail=True)
        bad_rows,_ = comparison(screen+[confirm[0],fail_confirm,confirm[2]], nominations)
        self.assertIsNone(bad_rows[0]['B_pre_over_A'])

    def test_frontier_uses_control_at_winning_b_step_count(self):
        def option(steps,a,b,u):
            row = dict(family='E',n=64,cert_tol=1e-7,total=1.,steps=steps,accuracy_kind='global_cp_error',
                       B_pre_over_B_none=b/u if u else None)
            for role,cost in [('A',a),('B_pre',b),('B_none',u)]:
                row[role+'_config_id'] = f'{role}-{steps}' if cost else None
                row[role+'_median_production_seconds'] = cost
            return row
        r, = frontiers([option(1,2.,6.,1.),option(2,3.,4.,5.)])
        self.assertEqual((r['A_steps'],r['B_pre_steps'],r['B_none_steps']), (1,2,2))
        self.assertEqual(r['B_pre_over_A'],2.)
        self.assertEqual(r['B_pre_over_B_none'],.8)

    def test_brackets_share_certified_reference_and_validate_exact_control(self):
        from galerkin_sylvester_tdvp import witness_pair_brackets as brackets
        with TemporaryDirectory() as temp:
            p = Path(temp)
            hashes = dict.fromkeys(brackets.PHYSICS_SOURCES,'abc')
            (p/'environment.json').write_text(json.dumps(dict(source_files_sha256=hashes)))
            physical = dict(family='E',n=8,total=1.)
            ref = dict(physical,config_id='ref',repeat=1,directory='tight',global_cp_bound=4e-8)
            setting = dict(physical,cert_tol=1e-5,steps=1)
            measured = []
            for role in ('A','B_pre'):
                setting.update({role+'_rtol':.01,role+'_config_id':role,
                               role+'_confirmed_repeats':3,role+'_max_global_cp_error':3e-6})
                measured.extend(dict(physical,phase='confirm',repeat=i,config_id=role,
                                     directory=f'{role}_{i}',global_cp_error=3e-6,global_cp_bound=9e-6)
                                for i in (1,2,3))
            refs = ('old',tuple(hashes.values()),[],[ref])
            with patch.object(brackets,'cohort',return_value=refs), patch.object(brackets,'checkpoint'), \
                 patch.object(brackets,'interval',return_value=dict(lower=2.9e-6,upper=3.1e-6)):
                rows, summary = tighter_references(p,p/'prior',[setting],measured)
                self.assertEqual(len(rows),6)
                self.assertEqual(len(summary),2)
                self.assertTrue(all(x['measured_inside'] for x in rows))
                self.assertEqual({x['reference_directory'] for x in rows},{'tight'})
                measured[0]['global_cp_error'] = 4e-6
                with self.assertRaisesRegex(ValueError,'Independent score'):
                    tighter_references(p,p/'prior',[setting],measured)

    def test_worker_history_checks_intermediate_witness_and_aggregates_stages(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp)
            worker = run/'candidate'
            worker.mkdir()
            row = dict(directory=worker.name, n=16,family='D',total=.1,steps=2,accepted_steps=2,
                       method='B',preconditioner='preparation',parent_complete=True,
                       global_cp_bound=8e-6,cert_tol=1e-5,rtol=.02,cg_iterations=8)
            accepted = [dict(time=t, bound=row['global_cp_bound'], accepted=True) for t in (.05,.1)]
            solves = [dict(start=t,stage=s,status='success',iterations=2,matvec_count=3,rtol=.02,preconditioner='preparation',
                       setup_seconds=.001,solve_seconds=.004,audit_seconds=.002,
                       preconditioner_setup_seconds=.0001,preconditioner_apply_seconds=.0007,
                       preconditioner_calls=2) for t in (0,.05) for s in ('start','midpoint')]
            def put(name, data):
                (worker/name).write_text(''.join(json.dumps(d)+'\n' for d in data))
            put('solves.jsonl', solves)
            put('events.jsonl', accepted)
            times = check_trace(run,row)
            self.assertEqual(times['iterations'],8)
            self.assertAlmostEqual(times['preconditioner_apply_seconds'],.0028)
            (run/'raw').mkdir()
            with ZipFile(run/'raw'/'D16.zip','w') as archive:
                for path in worker.iterdir():
                    archive.write(path,path.relative_to(run))
                    path.unlink()
            self.assertEqual(check_trace(run,row),times)
            accepted[0]['bound']=2e-5
            put('events.jsonl', accepted)
            with self.assertRaisesRegex(ValueError,'endpoint trace'):
                check_trace(run,row)
