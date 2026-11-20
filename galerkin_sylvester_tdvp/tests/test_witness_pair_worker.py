"""Shared positive-witness endpoint semantics for the A/B comparison."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from galerkin_sylvester_tdvp import witness_pair_worker as pair


class PositiveWitnessPairTests(unittest.TestCase):
    def test_both_predictors_reach_certified_measured_endpoint(self):
        for family in ('D','E'):
            for method in ('A','B'):
                with self.subTest(family=family,method=method), TemporaryDirectory() as folder:
                    out=Path(folder)
                    row=pair.worker(dict(n=3, family=family, method=method, steps=1, total=.1,
                                         cert_tol=1e-5, memory_mib=512),out)
                    self.assertEqual(row['status'],'completed')
                    self.assertEqual(row['reached_time'],.1)
                    self.assertEqual(row['accepted_steps'],1)
                    self.assertLessEqual(row['global_cp_bound'],1e-5)
                    self.assertLessEqual(row['global_cp_error'],row['global_cp_bound']+1e-11)
                    self.assertGreater(row['predictor_seconds'],0.)
                    self.assertGreater(row['witness_seconds'],0.)
                    self.assertGreater(row['production_seconds'],row['predictor_seconds']+row['witness_seconds'])
                    self.assertEqual(row['production_seconds'],json.loads((out/'measurement.json').read_text())['production_seconds'])
                    self.assertEqual((out/'solves.jsonl').exists(),method=='B')
                    event=json.loads((out/'events.jsonl').read_text())
                    self.assertTrue(event['accepted'])
                    self.assertAlmostEqual(event['bound'],row['global_cp_bound'])
                    self.assertGreater(event['correction_updates'],0)
                    with np.load(out/'last_accepted.npz') as checkpoint:
                        self.assertEqual(float(checkpoint['time']),.1)
                        self.assertEqual(float(checkpoint['bound']),row['global_cp_bound'])

    def test_uncertified_candidate_leaves_last_accepted_checkpoint(self):
        real_endpoint=pair._witness_endpoint
        def reject_second(engine,state,pred,t,cfg,row,out):
            proposal, info=real_endpoint(engine,state,pred,t,cfg,row,out)
            if t>.05:
                info=dict(info,accepted=False,bound=2*cfg['cert_tol'])
            return proposal,info
        with TemporaryDirectory() as folder, patch.object(pair,'_witness_endpoint',side_effect=reject_second):
            out=Path(folder)
            row=pair.worker(dict(n=3,family='E',method='B',steps=2,total=.1,
                                 cert_tol=1e-5,memory_mib=512),out)
            self.assertEqual(row['status'],'witness_uncertified')
            self.assertEqual(row['accepted_steps'],1)
            self.assertEqual(row['rejected_steps'],1)
            self.assertEqual(row['reached_time'],.05)
            self.assertIsNone(row['global_cp_error'])
            self.assertIsNone(row['global_cp_bound'])
            with np.load(out/'last_accepted.npz') as checkpoint:
                self.assertEqual(float(checkpoint['time']),.05)
            self.assertTrue((out/'exception.txt').exists())

    def test_nonhermitian_candidate_pays_skew_in_global_bound(self):
        model=pair.make_model('E',3)
        audit=pair.witness.WitnessAudit.parity_chain(3)
        f,x,eta=pair.witness.factor_to_root(pair.witness.parity_power(3,.25),4)
        delta=.002
        trial=pair.tn.scale(x,1+1j*delta)
        distance=pair.tn.distance(trial,pair.mpo.multiply(f,pair.mpo.adjoint(f)))
        check,_=pair._audit(audit,trial,distance,audit.physical,0.,1e-5)
        self.assertAlmostEqual(check['skew_allowance'],delta,places=12)
        self.assertLessEqual(pair.tn.distance(trial,model.root),check['bound']+1e-12)
