"""Balanced independent timing check of the selected tight D64 one-interval setting."""
import argparse
import hashlib
import json
from pathlib import Path
from random import Random
from statistics import median
import subprocess
from unittest.mock import patch

from . import witness_preconditioned_run as study
from .witness_pair_analysis import csv_file


def run(source, out):
    out.mkdir(parents=True, exist_ok=False)
    base = json.loads((source/'environment.json').read_text())
    hashes = study.source_hashes()
    if base['source_files_sha256'] != hashes:
        raise ValueError('Numerical implementation differs from the tuning run')
    script = Path(__file__).resolve()
    hashes[str(script.relative_to(study.bm.ROOT))] = hashlib.sha256(script.read_bytes()).hexdigest()
    sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=study.bm.ROOT,text=True).strip()
    group = 'D64_s1_tol1e-07'
    selected = json.loads((source/'selection.json').read_text())[group]
    grid = json.loads((source/'grid.json').read_text())
    configs = {k:next(dict(c,requested_repeats=6,source_sha=sha,source_files_sha256=hashes)
                      for c in grid if c['phase']=='confirm' and c['config_id']==selected[k])
               for k in ('A','B_pre')}
    orders = [('A','B_pre'), ('B_pre','A')] * 3
    Random(20261119).shuffle(orders)
    study.bm.save(out/'environment.json', dict(base, source_sha=sha, source_files_sha256=hashes,
                  settings=dict(group=group,source=source.name,prior_settings=base['settings'],
                                paired_orders=orders,selected=selected,seed=20261119),
                  schedule='Six fresh adjacent A/B pairs with balanced order, at the frozen screening winner.'))
    study.bm.save(out/'grid.json',list(configs.values()))
    rows, pairs = [], []
    for index, order in enumerate(orders,1):
        by_role = {}
        for role in order:
            with patch.object(study.supplement,'MODULE',study.MODULE):
                r = study.supplement.run_case(out,configs[role],index)
            rows.append(r)
            by_role[role] = r
            study.bm.save(out/'measurements.json',rows)
            study.bm.csv_table(out/'measurements.csv',rows)
        pair = dict(pair=index,order=' then '.join(order),both_eligible=all(map(study.valid,by_role.values())))
        for role,r in by_role.items():
            for field in ('status','global_cp_bound','production_seconds','predictor_seconds','witness_seconds',
                          'correction_updates','cg_iterations','peak_rss_mib'):
                pair[role+'_'+field]=r.get(field)
        if pair['both_eligible']:
            a,b = (by_role[m]['production_seconds'] for m in ('A','B_pre'))
            pair.update(B_over_A=b/a,B_minus_A_seconds=b-a)
        pairs.append(pair)
        csv_file(out/'pairs.csv',pairs)
    good=[p for p in pairs if p['both_eligible']]
    summary=dict(eligible_pairs=len(good),requested_pairs=len(orders))
    if len(good)==len(orders):
        summary.update(median_paired_ratio=median(p['B_over_A'] for p in good),
                       median_paired_difference_seconds=median(p['B_minus_A_seconds'] for p in good),
                       B_faster_pairs=sum(p['B_over_A']<1 for p in good),
                       min_paired_ratio=min(p['B_over_A'] for p in good),
                       max_paired_ratio=max(p['B_over_A'] for p in good))
        for role in configs:
            summary[role+'_median_seconds']=median(p[role+'_production_seconds'] for p in good)
    study.bm.save(out/'summary.json',summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.source.resolve(),args.out.resolve())
