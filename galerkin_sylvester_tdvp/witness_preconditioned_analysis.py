"""Reduce the target-tuned preparation-preconditioned Galerkin witness experiment."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from math import isclose
from pathlib import Path
from statistics import median
from zipfile import ZipFile

from .witness_pair_analysis import canonical, csv_file, finite, reached, summarize_configs
from .witness_pair_archive import SAFE_NAME


ROLES = {('A', 'ambient'): 'A', ('B', 'preparation'): 'B_pre', ('B', 'none'): 'B_none'}
GROUP_KEYS = ('family', 'n', 'total', 'steps', 'cert_tol')
SHARED = GROUP_KEYS + ('chi', 'levels', 'passes', 'residual_bond', 'memory_mib', 'timeout_seconds')
# Stage totals are summed across all stages in one worker before taking medians over workers.
STAGE_TOTALS = ('iterations', 'matvec_count', 'setup_seconds', 'solve_seconds', 'audit_seconds',
                'preconditioner_setup_seconds', 'preconditioner_apply_seconds', 'preconditioner_calls')
REPORT_FIELDS = ('config_id', 'rtol', 'max_endpoint_score', 'max_global_cp_bound',
                 'max_global_cp_error', 'median_production_seconds', 'min_production_seconds',
                 'max_production_seconds', 'median_total_seconds', 'median_predictor_seconds',
                 'median_witness_seconds', 'median_correction_updates', 'median_peak_rss_mib',
                 'median_iterations', 'median_matvec_count', 'median_setup_seconds_stages',
                 'median_solve_seconds', 'median_audit_seconds', 'median_preconditioner_setup_seconds',
                 'median_preconditioner_apply_seconds', 'median_preconditioner_calls')


def group_key(entry):
    return tuple(entry[k] for k in GROUP_KEYS)


def events(run, row, filename):
    """Read from the working tree or the per-size raw ZIP in a clean checkout."""
    directory = row['directory']
    if not SAFE_NAME.fullmatch(directory):
        raise ValueError('Invalid worker name')
    member = f'{directory}/{filename}'
    if (run / member).is_file():
        raw = (run / member).read_bytes()
    else:
        with ZipFile(run / 'raw' / f"{row['family']}{row['n']:02d}.zip") as archive:
            raw = archive.read(member)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def check_trace(run, row, stage_rows=None):
    """Check the actual endpoint and solve traces for every successful worker."""
    count = row['steps']
    trace = events(run, row, 'events.jsonl')
    times = [row['total'] * i / count for i in range(count + 1)]
    if (row.get('accepted_steps') != count or row.get('parent_complete') is not True
            or len(trace) != count or any(e.get('accepted') is not True
                    or not isclose(e['time'], t, rel_tol=1e-13, abs_tol=1e-15)
                    or not finite(e.get('bound')) or e['bound'] > row['cert_tol']
                    for e, t in zip(trace, times[1:]))
            or trace[-1]['bound'] != row['global_cp_bound']):
        raise ValueError(f"Inconsistent accepted endpoint trace for {row['directory']}")
    if row['method'] == 'A':
        return {}
    solves = events(run, row, 'solves.jsonl')
    expected = [(t, stage) for t in times[:-1] for stage in ('start', 'midpoint')]
    if (len(solves) != len(expected) or any(s['status'] != 'success' or s['stage'] != stage
            or not isclose(s['start'], t, rel_tol=1e-13, abs_tol=1e-15)
            for s, (t, stage) in zip(solves, expected))):
        raise ValueError(f"Incomplete solve history for {row['directory']}")
    if any(s.get('preconditioner') != row['preconditioner'] or s.get('rtol') != row['rtol'] for s in solves):
        raise ValueError(f"Stage tolerance or preconditioner differs from enrollment in {row['directory']}")
    sums = {}
    for field in STAGE_TOTALS:
        vals = [s.get(field) for s in solves]
        if field.startswith('preconditioner_') and row['preconditioner'] == 'none':
            vals = [0 if v is None else v for v in vals]
        if not all(finite(v) for v in vals):
            raise ValueError(f"Invalid {field} in {row['directory']}")
        sums[field] = sum(vals)
    if stage_rows is not None:
        fields = STAGE_TOTALS + ('stage','start','backend','preconditioner',
                                 'galerkin_residual_rel','physical_residual_rel','condition_estimate',
                                 'residual_audit_method','residual_norm_radius')
        for s in solves:
            stage_rows.append(dict(config_id=row['config_id'], repeat=row['repeat'], phase=row['phase'],
                                   family=row['family'], n=row['n'], total=row['total'], steps=count,
                                   cert_tol=row['cert_tol'], rtol=row['rtol'],
                                   requested_preconditioner=row['preconditioner'],
                                   **{f:s.get(f) for f in fields}))
    if sums['iterations'] != row['cg_iterations']:
        raise ValueError(f"Iteration count differs from trace in {row['directory']}")
    return sums


def summarize(run, grid, rows, env):
    entries = summarize_configs(grid, rows, env)
    by_cfg = defaultdict(list)
    for row in rows:
        by_cfg[row['phase'], row['config_id']].append(row)
    traces, stages = [], []
    for entry, cfg in zip(entries, grid):
        key = (cfg['phase'], cfg['config_id'])
        raw = by_cfg[key]
        role = ROLES[cfg['method'], cfg['preconditioner']]
        entry.update(role=role, residual_bond=cfg['residual_bond'])
        if any(r.get('preconditioner') != cfg['preconditioner']
               or r.get('residual_bond') != cfg['residual_bond']
               or r.get('source_files_sha256') != env['source_files_sha256'] for r in raw):
            raise ValueError(f"Settings differ from enrollment for {cfg['config_id']}")
        totals = []
        for r in raw:
            if r['status'] != 'completed':
                continue
            summary = check_trace(run, r, stages)
            if summary:
                totals.append(summary)
                traces.append(dict(zip(GROUP_KEYS, group_key(r)), phase=r['phase'], role=role,
                                   config_id=r['config_id'], repeat=r['repeat'], rtol=r['rtol'], **summary))
        if totals and entry['qualified_endpoints']:
            for k in STAGE_TOTALS:
                field = 'setup_seconds_stages' if k == 'setup_seconds' else k
                entry['median_' + field] = median(s[k] for s in totals)
    return entries, traces, stages


def screened_winner(options):
    valid = [e for e in options if e['phase'] == 'screen' and reached(e, e['cert_tol'])]
    return min(valid, key=lambda e: e['median_production_seconds']) if valid else None


def comparison(groups, selection):
    """Only nominate a preconditioned tolerance from screening, then require all fresh repeats."""
    nominations = {}
    for picks in selection.values():
        for role, key in picks.items():
            if key:
                if key in nominations or role not in ROLES.values():
                    raise ValueError(f'Unexpected selection {role}/{key}')
                nominations[key] = role
    known = {}
    settings = {}
    for e in groups:
        name = e['config_id']
        known[name] = e['role']
        identity = tuple(canonical(e[k]) for k in SHARED + ('role','rtol','atol','maxiter','work'))
        if name in settings and settings[name] != identity:
            raise ValueError('Setting differs between screen and confirmation')
        settings[name] = identity
        if e['phase'] == 'confirm' and name not in nominations:
            raise ValueError('Confirmation has no screening nomination')
    if any(known.get(key) != role for key, role in nominations.items()):
        raise ValueError('Unknown configuration in selection')
    per_group = defaultdict(list)
    for e in groups:
        per_group[group_key(e)].append(e)
    matched, strict = [], []
    for key, options in sorted(per_group.items()):
        row = {k:options[0][k] for k in SHARED}
        row['accuracy_kind'] = options[0]['accuracy_kind']
        # A different work cap is intrinsic to each predictor. Both witness budgets must match.
        if len({tuple(canonical(e[k]) for k in SHARED) for e in options}) != 1:
            raise ValueError('Cannot compare different endpoint/witness budgets')
        best = screened_winner([e for e in options if e['role'] == 'B_pre'])
        selected = [e for e in options if e['phase'] == 'screen' and e['config_id'] in nominations]
        selected_pre = [e['config_id'] for e in selected if e['role'] == 'B_pre']
        if selected_pre != ([best['config_id']] if best else []):
            raise ValueError('B nomination did not use the fastest qualifying screen')
        if any(not reached(e, e['cert_tol']) for e in selected):
            raise ValueError('Failed screen nominated for confirmation')
        if len({e['role'] for e in selected}) != len(selected):
            raise ValueError('Multiple nominations for one solver role')
        if any(e['role'] == 'B_none' and best and
               any(e.get(k) != best.get(k) for k in ('rtol','atol','maxiter','work')) for e in selected):
            raise ValueError('Unpreconditioned control differs from the selected solver budget')
        eligible = {e['role']: e for e in options if e['phase'] == 'confirm'
                    and e['config_id'] in nominations and reached(e, e['cert_tol'])}
        for role in ROLES.values():
            e = eligible.get(role, {})
            row[role + '_preferred'] = next((s['config_id'] for s in selected if s['role'] == role), None)
            row[role + '_confirmed_repeats'] = e.get('completed', 0)
            for f in REPORT_FIELDS:
                row[role + '_' + f] = e.get(f)
        a, b, u = (eligible.get(role) for role in ('A', 'B_pre', 'B_none'))
        row['matched'] = bool(a and b)
        row['B_pre_over_A'] = b['median_production_seconds']/a['median_production_seconds'] if a and b else None
        row['B_pre_over_B_none'] = b['median_production_seconds']/u['median_production_seconds'] if b and u else None
        matched.append(row)
        for role in ('B_pre', 'B_none'):
            for e in options:
                if e['phase'] == 'screen' and e['role'] == role and e['rtol'] == 1e-9:
                    chosen = next((s for s in selected if s['role'] == role), {})
                    valid = reached(e, e['cert_tol'])
                    old, new = e['median_production_seconds'], chosen.get('median_production_seconds')
                    strict.append(dict(zip(GROUP_KEYS, key), phase='screen', role=role, statuses=e['statuses'],
                                       qualified=valid, **{f: e.get(f) for f in REPORT_FIELDS},
                                       selected_config_id=chosen.get('config_id'),selected_rtol=chosen.get('rtol'),
                                       selected_screen_seconds=new,selected_screen_iterations=chosen.get('median_iterations'),
                                       selected_screen_correction_updates=chosen.get('median_correction_updates'),
                                       selected_screen_score=chosen.get('max_endpoint_score'),
                                       selected_screen_over_strict_screen=new/old if new and valid else None))
    return matched, strict


def frontiers(matched):
    """Compare the fastest confirmed settings across step counts at the same target."""
    physical = ('family','n','total','cert_tol')
    groups = defaultdict(list)
    for r in matched:
        groups[tuple(r[k] for k in physical)].append(r)
    result = []
    for key, options in sorted(groups.items()):
        row = dict(zip(physical,key),accuracy_kind=options[0]['accuracy_kind'])
        picks = {}
        for role in ('A','B_pre'):
            candidates = [r for r in options if r[role+'_config_id']]
            picks[role] = min(candidates,key=lambda r:r[role+'_median_production_seconds']) if candidates else {}
        # The unpreconditioned control belongs to B's winning grid and tolerance.
        picks['B_none'] = picks['B_pre'] if picks['B_pre'].get('B_none_config_id') else {}
        for role, picked in picks.items():
            row[role+'_steps'] = picked.get('steps')
            for field in REPORT_FIELDS + ('confirmed_repeats',):
                row[role+'_'+field] = picked.get(role+'_'+field)
        a, b = (row[role+'_median_production_seconds'] for role in ('A','B_pre'))
        row['matched'] = bool(a and b)
        row['B_pre_over_A'] = b/a if a and b else None
        row['B_pre_over_B_none'] = picks['B_pre'].get('B_pre_over_B_none')
        result.append(row)
    return result


def prior_comparisons(prior, matched):
    """Historical fixed-grid times in separate columns, using identical witness levels."""
    read = lambda name: json.loads((prior / name).read_text())
    grid = read('grid.json')
    entries = summarize_configs(grid, read('measurements.json'), read('environment.json'))
    prior_by_key = {(group_key(e), e['method']): (e, cfg) for e, cfg in zip(entries, grid)
                    if e['phase'] == 'confirm' and reached(e, e['cert_tol'])}
    comparisons = []
    for r in matched:
        for historic_role, role in [('A', 'A'), ('B', 'B_pre')]:
            hit = prior_by_key.get((group_key(r), historic_role))
            if hit is None or r[role + '_config_id'] is None:
                continue
            old, cfg = hit
            if any(cfg.get(k) != r.get(k) for k in ('chi','levels','passes','residual_bond')):
                continue
            comparisons.append(dict(zip(GROUP_KEYS, group_key(r)), role=role, prior_run=prior.name,
                prior_config_id=cfg['config_id'], prior_rtol=cfg['rtol'],
                prior_production_seconds=old['median_production_seconds'],
                current_config_id=r[role + '_config_id'], current_rtol=r[role + '_rtol'],
                current_production_seconds=r[role + '_median_production_seconds'],
                current_over_prior=r[role + '_median_production_seconds']/old['median_production_seconds']))
    return comparisons


def tighter_references(run, reference_cohort, matched, raw):
    """Use one independently computed witnessed endpoint as a common comparison root per size."""
    from .witness_pair_brackets import PHYSICS_SOURCES, TARGET, checkpoint, cohort, interval, physical
    _, ref_source, _, references = cohort(reference_cohort)
    env = json.loads((run/'environment.json').read_text())
    if tuple(env['source_files_sha256'][f] for f in PHYSICS_SOURCES) != ref_source:
        raise ValueError('Physical model, witness, or norm source differs between reference cohorts')
    best = {}
    for r in sorted(references,key=lambda x:(x['global_cp_bound'],x['config_id'],x['repeat'])):
        best.setdefault(physical(r),r)
    confirmed = defaultdict(list)
    for r in raw:
        if r['phase'] == 'confirm':
            confirmed[r['config_id']].append(r)
    rows, summary, roots = [], [], {}
    for setting in matched:
        if setting['cert_tol'] != TARGET:
            continue
        key = physical(setting)
        ref = best.get(key)
        for role in ('A','B_pre'):
            name = setting[role+'_config_id']
            if not name:
                continue
            shared = dict(zip(GROUP_KEYS,group_key(setting)), role=role,config_id=name,
                          rtol=setting[role+'_rtol'], reference_cohort=reference_cohort.name)
            if ref is None:
                summary.append(dict(shared,status='no tighter confirmed reference'))
                continue
            if key not in roots:
                roots[key] = checkpoint(reference_cohort,ref)
            shared.update(reference_config_id=ref['config_id'],reference_repeat=ref['repeat'],
                          reference_bound=ref['global_cp_bound'],reference_directory=ref['directory'])
            part = []
            for r in confirmed[name]:
                limits = interval(checkpoint(run,r),roots[key],r['global_cp_bound'],ref['global_cp_bound'])
                score = r.get('global_cp_error')
                if score is not None and not limits['lower'] <= score <= limits['upper']:
                    raise ValueError(f"Independent score lies outside conditional bracket for {r['directory']}")
                part.append(dict(shared,repeat=r['repeat'],directory=r['directory'], own_bound=r['global_cp_bound'],
                                 measured_error=score,measured_inside=True if score is not None else None,**limits))
            if len(part) != setting[role+'_confirmed_repeats']:
                raise ValueError(f'Repeat count differs from confirmed result for {name}')
            rows.extend(part)
            summary.append(dict(shared,status='conditional bracket',repeats=len(part),
                                lower=min(v['lower'] for v in part),upper=max(v['upper'] for v in part),
                                max_measured_error=setting[role+'_max_global_cp_error']))
    return rows, summary


def plot_two_step(matched, path):
    """Fixed-grid medians and min–max ranges for the three repeated methods."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    targets = sorted({r['cert_tol'] for r in matched},reverse=True)
    families = sorted({r['family'] for r in matched})
    sizes = sorted({r['n'] for r in matched})
    fig, axes = plt.subplots(len(families),len(targets),squeeze=False,sharex=True,layout='constrained',
                             figsize=(4.5*len(targets),3.0*len(families)+.35))
    styles = [('A','A','o','#155789',.975),('B_pre','B, preparation PCG','s','#bc5919',1.),
              ('B_none','B, same-tolerance CG','D','#4f6a53',1.025)]
    legend = [Line2D([],[],marker=m,color=c,ls='none',label=l,markersize=6) for _,l,m,c,_ in styles]
    for i,fam in enumerate(families):
        for j,target in enumerate(targets):
            ax = axes[i,j]
            group = [r for r in matched if r['family']==fam and r['cert_tol']==target and r['steps']==2]
            for role,_,marker,color,offset in styles:
                points = sorted((r['n'],r[role+'_median_production_seconds'],r[role+'_min_production_seconds'],
                                 r[role+'_max_production_seconds']) for r in group if r[role+'_config_id'])
                if not points:
                    continue
                xx,yy,lo,hi = zip(*points)
                ax.errorbar([x*offset for x in xx],yy,
                            yerr=([m-l for m,l in zip(yy,lo)],[h-m for h,m in zip(hi,yy)]),
                            marker=marker,linestyle='none',color=color,markersize=5,capsize=2)
            ax.set_xscale('log',base=2)
            ax.set_yscale('log')
            ax.set_xlim(.9*min(sizes),1.12*max(sizes))
            ax.set_xticks(sizes,[str(n) for n in sizes])
            ax.grid(axis='y',alpha=.2)
            name = 'Correlated chain' if fam=='D' else 'Parity chain'
            ax.set_title(f'{name} · target {target:.0e}')
            if i==len(families)-1:
                ax.set_xlabel('Spins')
            if j==0:
                ax.set_ylabel('Production time (s)')
    fig.suptitle('Two midpoint intervals with a positive witness',fontsize=14)
    fig.legend(handles=legend,loc='outside lower center',ncols=3,frameon=False)
    fig.savefig(path,dpi=175)
    plt.close(fig)


def analyze(run, prior=None, reference_cohort=None):
    read = lambda name: json.loads((run / name).read_text())
    raw = read('measurements.json')
    entries, traces, stages = summarize(run, read('grid.json'), raw, read('environment.json'))
    matched, strict = comparison(entries, read('selection.json'))
    out = run / 'analysis'
    out.mkdir(exist_ok=True)
    csv_file(out/'configurations.csv', [{k:v for k,v in e.items() if k!='source_manifest'} for e in entries])
    csv_file(out/'matched_accuracy.csv', matched)
    csv_file(out/'frontier.csv', frontiers(matched))
    csv_file(out/'strict_controls.csv', strict)
    csv_file(out/'worker_solves.csv', traces)
    csv_file(out/'stage_solves.csv', stages)
    failure_fields = ('config_id','phase','repeat','family','n','steps','cert_tol','rtol','method',
                      'preconditioner','status','reached_time','accepted_steps','error','failure_details',
                      'last_accepted_bound','wall_seconds','peak_rss_mib','directory')
    csv_file(out/'failures.csv', [{k:r.get(k) for k in failure_fields} for r in raw
                                 if r['status'] != 'completed'])
    if prior:
        prior_runs = [prior] if isinstance(prior, Path) else prior
        csv_file(out/'historical_comparison.csv', [c for old in prior_runs
                                                 for c in prior_comparisons(old, matched)])
    if reference_cohort:
        details, overview = tighter_references(run, reference_cohort, matched, raw)
        csv_file(out/'bracket_repeats.csv', details)
        csv_file(out/'brackets.csv', overview)
    plot_two_step(matched, out/'two_step_cost.png')
    return matched


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--prior', type=Path, action='append', help='Historical repeated fixed-grid cohort, recorded separately')
    p.add_argument('--reference-cohort',type=Path,
                   help='Use tighter historical confirmed roots to bracket each new 1e-5 configuration')
    args = p.parse_args()
    analyze(args.run, args.prior, args.reference_cohort)
