"""Flatten paired archives and summarize only complete measured configurations."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import median
from zipfile import ZipFile


def read_rows(z, name):
    if name not in z.namelist():
        return []
    return [json.loads(line) for line in z.read(name).splitlines()]


def table(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,columns)
        writer.writeheader()
        writer.writerows(rows)


def flatten(directories, output):
    output.mkdir(parents=True,exist_ok=False)
    summaries, endpoints, iterations, digests = [], [], [], []
    groups=defaultdict(list)
    timing_fields=('production_seconds','witness_seconds','predictor_seconds','setup_seconds','scoring_seconds',
                   'checkpoint_seconds','physical_density_seconds','peak_rss_mib','total_seconds','wall_seconds')
    for directory in sorted(directories):
        manifest=json.loads((directory/'manifest.json').read_text())
        repeats=manifest['repeats']
        lines=(directory/'measurements.jsonl').read_text().splitlines()
        if len(lines)!=len(manifest['grid'])*repeats:
            raise ValueError(f'Incomplete enrollment at {directory}')
        observed=set()
        for line in lines:
            r=json.loads(line)
            if r['worker'] in observed:
                raise ValueError(f'Duplicate worker {r["worker"]}')
            observed.add(r['worker'])
            key=dict(cohort=directory.name,family=r['family'],n=r['n'],total=r['total'],steps=r['steps'],
                     cert_tol=r['cert_tol'],label=r['label'],mode=r['mode'],preconditioner=r.get('preconditioner','none'),
                     freeze_step=r.get('freeze_step'),max_elements=r['max_elements'])
            run=dict(key,repeat=r['repeat'],worker=r['worker'])
            summary=dict(run,status=r['status'],reached_time=r.get('reached_time',0.),accepted_steps=r.get('accepted_steps',0),
                         global_cp_bound=r.get('global_cp_bound'),global_cp_error=r.get('global_cp_error'),
                         failed_bound=r.get('failed_bound'),root_bond=r.get('root_bond'),factor_bond=r.get('factor_bond'),
                         source_changed_during_worker=bool(r.get('source_deltas_during_worker')),
                         **{k:r.get(k) for k in timing_fields})
            solves=updates=fit_updates=0
            failures=[]
            archive=directory/(r['worker']+'.zip')
            digest=hashlib.sha256(archive.read_bytes()).hexdigest()
            digests.append(f'{digest}  {directory.name}/{archive.name}')
            with ZipFile(archive) as z:
                if z.testzip() is not None:
                    raise ValueError(f'Corrupt worker ZIP for {r["worker"]}')
                scores={s['step']:s for s in read_rows(z,'scores.jsonl')}
                for event in read_rows(z,'endpoints.jsonl'):
                    step=event['step']
                    details={k:v for k,v in event.items() if k not in ('attempts','timings','last_failure')}
                    details.update(event['timings'])
                    failure=event.get('last_failure')
                    if failure:
                        failures.append(failure['status']+': '+failure['message'])
                    details.update(last_failure=None if failure is None else failure['status'])
                    endpoints.append(dict(run,**dict(details,**scores.get(step,{}))))
                    updates+=event['correction_updates']
                    fit_updates+=event['fit_updates']
                for a in read_rows(z,'attempts.jsonl'):
                    solve=a.get('solve',{})
                    info=dict(run,**{k:a.get(k) for k in ['step','phase','iteration','elapsed_seconds','bound',
                              'square_residual','scale','factor_bond','root_bond','witness','weighted_residual','improved']})
                    info.update({k:solve.get(k) for k in ['solver','power','iterations','matvec_count','dimension',
                                'condition_estimate','projected_residual','full_linear_residual','outside_tangent_residual',
                                'factor_lower_bound','verified_spd','peak_product_bond','factor_power_bond','peak_guarded_elements',
                                'preconditioner_setup_seconds','frame_seconds','operator_setup_seconds','solve_seconds','audit_seconds']})
                    interval=a.get('factor_fourth_residual')
                    if interval: info.update(fourth_residual_lower=interval[0],fourth_residual_upper=interval[1])
                    info['failure']=a.get('failure',{}).get('status')
                    iterations.append(info)
                    solves+=solve.get('iterations',0)
            if r.get('native'):
                updates=r.get('correction_updates')
                fit_updates=None
                solves=None
            summary.update(witness_krylov_iterations=solves,fit_updates=fit_updates,correction_updates=updates,
                           endpoint_failures=' | '.join(dict.fromkeys(failures)))
            summaries.append(summary)
            groups[tuple(key.items())+(('requested_repeats',repeats),)].append(summary)
    grouped=[]
    for keys,rows in groups.items():
        group=dict(keys)
        valid=[r for r in rows if r['status']=='completed' and r['global_cp_bound']<=r['cert_tol']]
        eligible=len(valid)==len(rows)==group['requested_repeats']
        group.update(executed=len(rows),accepted=len(valid),complete_grid=eligible,
                     median_last_accepted_time=median(r['reached_time'] for r in rows))
        if eligible:
            for name in (*timing_fields,'global_cp_bound','global_cp_error','correction_updates','witness_krylov_iterations'):
                values=[r[name] for r in valid if r.get(name) is not None]
                if values:
                    group['median_'+name]=median(values)
                if name=='production_seconds' and values:
                    group.update(min_production_seconds=min(values),max_production_seconds=max(values))
        grouped.append(group)
    table(output/'runs.csv',summaries)
    table(output/'endpoints.csv',endpoints)
    table(output/'iterations.csv',iterations)
    table(output/'configurations.csv',grouped)
    (output/'archives.sha256').write_text('\n'.join(sorted(digests,key=lambda a:a.split('  ',1)[1]))+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohorts',nargs='+',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    flatten(args.cohorts,args.out)
