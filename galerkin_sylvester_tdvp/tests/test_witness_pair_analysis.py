"""Repeated comparisons use like-for-like global evidence and source identity."""
import unittest

from galerkin_sylvester_tdvp.witness_pair_analysis import frontiers, summarize_configs


def cfg(method='A', family='D', n=16, phase='confirm', steps=2):
    return dict(family=family, n=n, method=method, phase=phase, total=.1 if family=='D' else 1.,
                steps=steps, chi=4, work=64 if method=='B' else 10,
                levels=[[4,6,10]], passes=4, cert_tol=1e-5, rtol=1e-8,
                maxiter=80, timeout_seconds=30, memory_mib=768,
                config_id=f'{family}{n}-{method}-{steps}',
                requested_repeats=3 if phase=='confirm' else 1)


def observations(c, elapsed=2.):
    return [dict(c,repeat=i,status='completed',reached_time=c['total'],production_seconds=elapsed,
                 total_seconds=elapsed+.09,scoring_seconds=.09,peak_rss_mib=40,
                 global_cp_error=None if c['family']=='D' and c['n']>8 else 5e-6,
                 global_cp_bound=9e-6,source_files_sha256={'impl':'abc'})
            for i in range(1,c['requested_repeats']+1)]


def evaluate(grid,rows,environment=None):
    return frontiers(summarize_configs(grid, rows, environment or {}), targets=(1e-5,))


class WitnessComparisonAnalysisTests(unittest.TestCase):
    def test_matched_uses_bound_above_eight_and_separate_work(self):
        a,b=cfg('A'),cfg('B')
        au,bu=observations(a,2), observations(b,3)
        tuned, same=evaluate([a,b], au+bu)
        self.assertEqual(len(same),1)
        self.assertEqual(same[0]['accuracy_kind'],'global_cp_bound')
        self.assertEqual((same[0]['A_work'],same[0]['B_work']),(10,64))
        self.assertAlmostEqual(tuned[0]['B_over_A_production'],1.5)
        for bad in ({'global_cp_bound':None},{'global_cp_bound':1.1e-5}):
            wrong=bu[:2]+[dict(bu[2], **bad)]
            self.assertIsNone(evaluate([a,b],au+wrong)[0][0]['B_over_A_production'])

    def test_d8_and_parity_require_measured_score(self):
        for family,n in (('D',8),('E',8),('E',64)):
            a=cfg(family=family,n=n)
            ar=observations(a)
            ar[-1]['global_cp_error']=None
            row=evaluate([a],ar)[0][0]
            self.assertEqual(row['accuracy_kind'],'global_cp_error')
            self.assertIsNone(row['A_config_id'])

    def test_missing_repeats_partial_nonfinite_and_provenance_fail(self):
        a,b=cfg('A'),cfg('B')
        ra,rb=observations(a),observations(b)
        for updates in ({'repeat':4},{'repeat':2},{'reached_time':.05},
                        {'global_cp_bound':float('nan')},{'production_seconds':0.},
                        {'total_seconds':float('inf')},{'status':'timeout'},
                        {'source_files_sha256':{}},{'source_files_sha256':{'a':'different'}},
                        {'chi':99},{'cert_tol':1e-3}):
            with self.subTest(updates=updates):
                self.assertFalse(evaluate([a,b],ra+rb[:2]+[dict(rb[2], **updates)])[0][0]['matched'])
        self.assertFalse(evaluate([a,b],ra+rb[:2])[0][0]['matched'])
        self.assertFalse(evaluate([a,b],ra+rb,{'source_files_sha256':{'a':'different'}})[0][0]['matched'])
        self.assertFalse(evaluate([a,b],ra+[dict(x,source_files_sha256={'b':'other'}) for x in rb])[0][0]['matched'])

    def test_screen_cannot_fill_confirmation_and_winner_must_pass_every_repeat(self):
        a=cfg('A')
        b2=cfg('B')
        b4=cfg('B',steps=4)
        bs=cfg('B',phase='screen')
        cheap=observations(b2,1.)
        cheap[1]['global_cp_bound']=2e-5
        raw=observations(a,4.)+cheap+observations(b4,6.)+observations(bs,.1)
        tuned,same=evaluate([a,b2,b4,bs],raw)
        confirm=next(r for r in tuned if r['phase']=='confirm')
        self.assertEqual(confirm['B_steps'],4)
        self.assertEqual(confirm['B_over_A_production'],1.5)
        self.assertFalse(same[0]['matched'])
        alone=evaluate([a,b2,bs],observations(a)+observations(bs))[0]
        self.assertTrue(all(not row['matched'] for row in alone))
