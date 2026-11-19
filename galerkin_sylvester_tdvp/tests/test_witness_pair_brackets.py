"""Independent final-state brackets, provenance, and archive-only replay."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

import numpy as np

from galerkin_sylvester_tdvp import witness_pair_brackets as b


SOURCES = {name: 'shared' for name in b.PHYSICS_SOURCES}


def cohort(root, name, settings, variant='primary', family='D', n=8, score_offset=0.):
    run = root / name
    run.mkdir()
    grid, rows = [], []
    for method, tol, delta, radius, seconds, phase in settings:
        key = f'{method}-{tol:g}'
        cfg = dict(config_id=key, family=family, n=n, method=method, phase=phase, total=.1,
                   steps=1, chi=2, work=5, levels=[[2, 4, 5]], passes=4, cert_tol=tol,
                   requested_repeats=3 if phase == 'confirm' else 1)
        grid.append(cfg)
        for rep in range(1, cfg['requested_repeats']+1):
            folder = f'{phase}_{key}_r{rep}'
            path = run / folder
            path.mkdir()
            value = delta + rep*1e-12
            score = value + score_offset if family == 'E' or n <= 8 else None
            row = dict(cfg, repeat=rep, directory=folder, parent_complete=True, status='completed',
                       reached_time=.1, global_cp_error=score, global_cp_bound=radius + rep*1e-12,
                       production_seconds=seconds, scoring_seconds=.1, total_seconds=seconds+.1,
                       source_files_sha256=SOURCES)
            rows.append(row)
            matrices = {f'x{j}': np.array([1., 0, 0, 0]).reshape(1, 4, 1) for j in range(n)}
            matrices['x0'][0, 0, 0] += value
            np.savez(path/'last_accepted.npz', time=.1, bound=row['global_cp_bound'], **matrices)
    for file, obj in [('grid', grid), ('measurements', rows),
                      ('environment', dict(source_files_sha256=SOURCES, settings=dict(variant=variant)))]:
        (run / (file+'.json')).write_text(json.dumps(obj))
    return run


LOOSE_A = ('A', 1e-5, 2e-6, 4e-6, 1., 'confirm')
LOOSE_B = ('B', 1e-5, 1.8e-6, 3e-6, 2., 'confirm')
TIGHT_A = ('A', 1e-7, 4e-8, 5e-8, 6., 'confirm')
TIGHT_B = ('B', 1e-7, 1e-8, 2e-8, 5., 'confirm')


class BracketTests(unittest.TestCase):
    def test_common_reference_across_cohorts_independent_scores_and_archived_sources(self):
        with TemporaryDirectory() as temp:
            t = Path(temp)
            a = cohort(t, 'first', [LOOSE_A, LOOSE_B, TIGHT_A])
            b_run = cohort(t, 'second', [LOOSE_B, TIGHT_B, ('A', 1e-8, 0., 1e-9, .1, 'screen')], 'stream')
            (b_run/'raw').mkdir()
            with ZipFile(b_run/'raw'/'D08.zip', 'w') as z:
                for path in b_run.glob('*/last_accepted.npz'):
                    z.write(path, path.relative_to(b_run))
                    path.unlink()
            rows, omitted = b.analyze([a, b_run], t/'brackets')
            self.assertEqual(len(rows), 9)
            self.assertFalse(omitted)
            for row in rows:
                self.assertEqual((row['reference_cohort'], row['reference_method'],
                                  row['reference_repeat'], row['reference_cert_tol']), ('second', 'B', 1, 1e-7))
                self.assertTrue(row['measured_inside'])
                self.assertAlmostEqual(row['distance'], row['measured_error'] - (1e-8+1e-12), places=13)
                self.assertGreater(row['roundoff_guard'], 0.)
                self.assertGreater(row['lower'], 0.)
                self.assertLess(row['upper'], row['own_bound'])
            self.assertIn('stream / second', (t/'brackets'/'report.md').read_text())
            self.assertIn('reference_config_id', (t/'brackets'/'brackets.csv').read_text())

    def test_missing_or_nonfinal_reference_and_identical_tolerances_are_omitted(self):
        with TemporaryDirectory() as temp:
            t = Path(temp)
            run = cohort(t, 'large', [LOOSE_A, TIGHT_A], n=16)
            file = run/'measurements.json'
            records = json.loads(file.read_text())
            records[-1].update(reached_time=.05, status='timeout', global_cp_bound=None)
            file.write_text(json.dumps(records))
            rows, omitted = b.brackets([run])
            self.assertFalse(rows)
            self.assertEqual([o['cert_tol'] for o in omitted], [1e-5])
            only = cohort(t, 'tight-only', [TIGHT_B], n=16)
            rows, omitted = b.brackets([only])
            self.assertFalse(rows)
            self.assertEqual([o['cert_tol'] for o in omitted], [1e-7])
            rows, _ = b.brackets([run, only])
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(r['measured_inside'] is None for r in rows))

    def test_checks_measurement_checkpoint_and_physical_identity(self):
        with TemporaryDirectory() as temp:
            t = Path(temp)
            bad_score = cohort(t, 'bad_score', [LOOSE_A], family='E', n=16, score_offset=1e-6)
            tight = cohort(t, 'ref', [TIGHT_B], family='E', n=16)
            with self.assertRaisesRegex(ValueError, 'Independent score'):
                b.brackets([bad_score, tight])
            record = json.loads((tight/'measurements.json').read_text())[0]
            with self.assertRaisesRegex(ValueError, 'differs from measurement'):
                b.checkpoint(tight, dict(record, global_cp_bound=3e-8))
            env = tight/'environment.json'
            meta = json.loads(env.read_text())
            meta['source_files_sha256'][b.PHYSICS_SOURCES[0]] = 'other-model'
            env.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'sources differ'):
                b.brackets([bad_score, tight])

    def test_tiny_distances_and_own_bound_cutoff(self):
        from galerkin_sylvester_tdvp.models import tn
        x = tn.vacuum(3)
        result = b.interval(x, x, 2e-10, 5e-10)
        self.assertEqual(result['lower'], 0.)
        self.assertEqual(result['upper'], 2e-10)
        with self.assertRaisesRegex(ValueError, 'Inconsistent witness'):
            b.interval(x, tn.scale(x, 1.1), 1e-5, 1e-7)
