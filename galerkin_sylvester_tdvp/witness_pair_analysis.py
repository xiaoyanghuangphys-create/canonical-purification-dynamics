"""Compare A+witness and B+witness at a common global-root accuracy target."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
from statistics import median

TARGETS = (1e-3, 1e-5, 1e-7)
METHODS = ('A', 'B')
PHASES = ('screen', 'confirm')
# Fields identifying the physical endpoint and the declared operational budgets.
SETTINGS = ('family', 'n', 'total', 'steps', 'cert_tol', 'chi', 'work', 'levels',
            'passes', 'rtol', 'atol', 'maxiter', 'memory_mib', 'timeout_seconds')
SHARED_ENDPOINT = ('family','n','total','steps','cert_tol','chi','levels','passes')


def finite(value, *, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def canonical(value):
    """Stable representation of a source manifest or a structured configuration."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def csv_file(path, records):
    if not records:
        path.write_text('')
        return
    keys = list(dict.fromkeys(k for record in records for k in record))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        writer.writeheader()
        for row in records:
            writer.writerow({k: canonical(v) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def summarize_configs(grid, rows, environment):
    """Include all enrolled configurations; incomplete or inconsistent groups cannot qualify."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['phase'], row['config_id']].append(row)
    ids = [(c['phase'], c['config_id']) for c in grid]
    if len(set(ids)) != len(ids) or groups.keys() - set(ids):
        raise ValueError('Duplicate grid ID or measurement outside the grid')
    result = []
    for cfg in grid:
        key = cfg['config_id']
        group = groups[cfg['phase'], key]
        family, n, method, phase = (cfg[k] for k in ('family', 'n', 'method', 'phase'))
        if family not in ('D', 'E') or method not in METHODS or phase not in PHASES:
            raise ValueError('Unknown physical family, method, or phase for ' + str(key))
        metric = ('global_cp_error' if family == 'E' or n <= 8 else 'global_cp_bound')
        reps = cfg['requested_repeats']
        status = Counter(r['status'] for r in group)
        enrolled = {r.get('repeat') for r in group}
        complete = [r for r in group if r['status'] == 'completed'
                    and r.get('reached_time') == cfg['total']]
        issues = []
        if not isinstance(reps, int) or reps <= 0:
            raise ValueError('Invalid requested repeat count')
        if len(group) != reps or enrolled != set(range(1, reps + 1)):
            issues.append('incomplete or duplicate repeat IDs')
        keys = SETTINGS + ('method', 'phase', 'requested_repeats')
        if any(r.get(k) != cfg[k] for r in group for k in keys if k in cfg):
            issues.append('worker configuration differs from grid')
        hashes = {canonical(r.get('source_files_sha256')) for r in group}
        expected_hashes = cfg.get('source_files_sha256') or environment.get('source_files_sha256')
        if (len(hashes) != 1 or hashes & {'null', '{}'}
                or (expected_hashes and hashes != {canonical(expected_hashes)})):
            issues.append('unverified or differing source manifests')
        if phase == 'confirm' and reps < 3:
            issues.append('fewer than three confirmation repeats')
        if len(complete) != reps:
            issues.append('requested endpoint incomplete')
        scored = [r for r in complete if finite(r.get(metric))]
        if len(scored) != reps:
            issues.append('required final global score unavailable')
        if any(not finite(r.get('production_seconds'), positive=True) or
               not finite(r.get('total_seconds'), positive=True) or
               not finite(r.get('scoring_seconds')) or
               r['total_seconds'] + 1e-9 < r['production_seconds']
               for r in complete):
            issues.append('invalid timing')
        # A measured error outside the bound is an audit failure. The additive
        # allowance is for ordinary floating point, consistent with D/E scoring.
        if any(finite(r.get('global_cp_error')) and finite(r.get('global_cp_bound'))
               and r['global_cp_error'] > r['global_cp_bound'] + 1e-10 for r in complete):
            issues.append('measured error exceeds the certificate')
        valid = not issues
        entry = {k: cfg[k] for k in SETTINGS if k in cfg}
        entry.update(config_id=key, phase=phase, method=method, requested_repeats=reps,
                     observed=len(group), completed=len(complete), accuracy_kind=metric,
                     qualified_endpoints=valid, statuses=dict(status), problems='; '.join(issues),
                     min_reached_time=min((r['reached_time'] for r in group
                                         if finite(r.get('reached_time'))), default=None),
                     max_reached_time=max((r['reached_time'] for r in group
                                         if finite(r.get('reached_time'))), default=None),
                     max_endpoint_score=max((r[metric] for r in scored), default=None),
                     source_manifest=next(iter(hashes)) if len(hashes) == 1 else None)
        for score in ('global_cp_error', 'global_cp_bound'):
            observed = [r[score] for r in complete if finite(r.get(score))]
            entry['max_' + score] = max(observed, default=None)
        for field in ('production_seconds', 'total_seconds', 'wall_seconds', 'scoring_seconds',
                      'setup_seconds', 'predictor_seconds', 'witness_seconds', 'checkpoint_seconds',
                      'correction_updates', 'correction_trials', 'predictors_selected', 'peak_rss_mib'):
            values = [r[field] for r in complete if finite(r.get(field))]
            entry['median_' + field] = median(values) if valid and len(values) == reps else None
            if field == 'production_seconds':
                entry['min_' + field] = min(values) if valid else None
                entry['max_' + field] = max(values) if valid else None
        result.append(entry)
    return result


def reached(setting, target):
    return (setting['qualified_endpoints'] and setting['max_endpoint_score'] is not None
            and setting['max_endpoint_score'] <= target)


def comparison_row(key, options, target):
    phase, family, n, total = key
    row = dict(phase=phase, family=family, n=n, total=total, target=target,
               accuracy_kind=('global_cp_error' if family == 'E' or n <= 8 else 'global_cp_bound'))
    selected = {}
    for method in METHODS:
        available = [s for s in options if s['method'] == method]
        candidates = [s for s in available if reached(s, target)]
        pick = min(candidates, key=lambda s: (s['median_production_seconds'], s['config_id'])) if candidates else None
        fields = ('config_id', 'steps', 'work', 'cert_tol', 'max_endpoint_score', 'requested_repeats',
                  'median_production_seconds', 'min_production_seconds', 'max_production_seconds',
                  'median_total_seconds', 'median_scoring_seconds', 'median_predictor_seconds',
                  'median_witness_seconds', 'median_correction_updates', 'median_predictors_selected',
                  'median_peak_rss_mib')
        for name in fields:
            row[method + '_' + name] = pick[name] if pick else None
        row[method + '_qualified_configs'] = len(candidates)
        row[method + '_observed_configs'] = sum(s['observed'] > 0 for s in available)
        selected[method] = pick
    a, b = (selected[m] for m in METHODS)
    matched = (a is not None and b is not None
               and a['source_manifest'] == b['source_manifest'])
    row['matched'] = matched
    row['B_over_A_production'] = b['median_production_seconds'] / a['median_production_seconds'] if matched else None
    row['B_over_A_including_scoring'] = b['median_total_seconds'] / a['median_total_seconds'] if matched else None
    return row


def frontiers(configs, targets=TARGETS):
    """Fastest observed setting per method; single screens and repeated confirmations stay separate."""
    groups = defaultdict(list)
    for cfg in configs:
        groups[(cfg['phase'], cfg['family'], cfg['n'], cfg['total'])].append(cfg)
    tuned, shared = [], []
    for phase, family, n, t in list(groups):
        for expected_phase in PHASES:
            groups.setdefault((expected_phase, family, n, t), [])
    for key in sorted(groups):
        opts = groups[key]
        for target in targets:
            tuned.append(comparison_row(key, opts, target))
        common_settings = defaultdict(list)
        for cfg in opts:
            common_settings[tuple(canonical(cfg.get(k)) for k in SHARED_ENDPOINT)].append(cfg)
        for settings in common_settings.values():
            if {s['method'] for s in settings} != set(METHODS):
                continue
            for target in targets:
                row = comparison_row(key, settings, target)
                for k in SHARED_ENDPOINT:
                    row[k] = settings[0].get(k)
                shared.append(row)
    return tuned, shared


def plot(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6.1), layout='constrained', sharex=True)
    colors = {'A': '#13578b', 'B': '#be5912'}
    for i, family in enumerate(('D', 'E')):
        for j, target in enumerate((1e-5, 1e-7)):
            ax = axes[i, j]
            for m in METHODS:
                vals = sorted((r['n'], r[m + '_median_production_seconds']) for r in rows
                              if r['phase'] == 'confirm' and r['family'] == family
                              and r['target'] == target and r[m + '_config_id'] is not None)
                ax.plot([v[0] for v in vals], [v[1] for v in vals],
                        marker='o' if m == 'A' else 's', linestyle='none', ms=7,
                        markerfacecolor='none' if m == 'B' else colors[m], markeredgewidth=1.8,
                        color=colors[m], label=m+' + witness')
            ax.set_yscale('log')
            ax.set_xscale('log', base=2)
            ax.set_xlim(7, 74)
            ax.set_xticks([8, 16, 32, 64], ['8', '16', '32', '64'])
            if i == 1:
                ax.set_xlabel('Spins')
            if j == 0:
                ax.set_ylabel('Median production time (s)')
            name = 'Correlated chain' if family == 'D' else 'Parity'
            ax.set_title(f'{name} · target {target:.0e}')
            ax.grid(alpha=.18, which='major')
            if not ax.has_data() or all(len(line.get_xdata()) == 0 for line in ax.lines):
                ax.text(.5, .5, 'No qualifying configuration', transform=ax.transAxes,
                        ha='center', va='center', color='#555555')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center',
               ncol=2, frameon=False)
    fig.savefig(out / 'matched_cost.png', dpi=160)
    plt.close(fig)


def number(value, fmt='.3g'):
    return format(value, fmt) if value is not None else '—'


def write_report(out, run, configs, tuned):
    raw_link = os.path.relpath(run, out)
    text = ['# Two-site root evolution with a positive endpoint witness', '',
            'The correlated-chain comparison uses the measured global canonical-root error at eight spins '
            'and the conditional global witness bound at 16, 32 and 64 spins. The parity family has an '
            'independent analytic global root at every size. Method A uses the repository’s approximate '
            'ambient MPO predictor. Method B solves the tangent Galerkin equation. Each predictor feeds '
            'the same positive-witness endpoint procedure.', '',
            '## Repeated cost at matched accuracy', '',
            'A setting qualifies when every planned confirmation repeat finishes at the requested time '
            'and passes the target. The selected setting minimizes median production time, including setup, '
            'prediction, witness correction/certification and checkpoint I/O. Independent scoring time, complete worker time, '
            'RSS and selected configuration IDs are in the frontier table. A B/A value above one means B '
            'took longer. A dash means the grid did not establish that entry. '
            '[Accuracy frontiers](matched_accuracy.csv) · [Common step and witness settings](same_settings.csv)', '']
    for target in (1e-5, 1e-7):
        text.extend([f'### Global target {target:.0e}', '',
                     '| Family | Spins | Evidence | A error or bound | A seconds | B error or bound | B seconds | B/A |',
                     '|---|---:|---|---:|---:|---:|---:|---:|'])
        for r in tuned:
            if r['phase'] == 'confirm' and r['target'] == target:
                metric = 'measured' if r['accuracy_kind'] == 'global_cp_error' else 'conditional bound'
                text.append(f"| {r['family']} | {r['n']} | {metric} | {number(r['A_max_endpoint_score'])} | "
                            f"{number(r['A_median_production_seconds'])} | {number(r['B_max_endpoint_score'])} | "
                            f"{number(r['B_median_production_seconds'])} | {number(r['B_over_A_production'])} |")
        text.append('')
    text.extend(['![Cost at global-root accuracy targets](matched_cost.png)', '',
                 'The frontier CSV also covers the looser $10^{-3}$ target. '
                 'The error and bound columns give the largest observed value among all repeats of the selected configuration. The figure shows each method only at sizes with qualifying confirmation repeats. '
                 'An unresolved upper bound leaves the target unconfirmed. It does not measure the true '
                 'root error on the larger correlated chains.', '',
                 '## Coverage and failed endpoints', '',
                 '| Phase | Family | Spins | Method | Completed / planned workers | Status counts |',
                 '|---|---|---:|---|---:|---|'])
    by_method = defaultdict(list)
    for s in configs:
        by_method[(s['phase'], s['family'], s['n'], s['method'])].append(s)
    for key, group in sorted(by_method.items()):
        statuses = Counter()
        for s in group:
            statuses.update(s['statuses'])
        desc = ', '.join(f'{k} ({v})' for k, v in sorted(statuses.items()))
        text.append('| '+' | '.join(map(str, key))+f" | {sum(s['completed'] for s in group)} / "
                    f"{sum(s['requested_repeats'] for s in group)} | {desc or 'No observations'} |")
    text.extend(['', 'The common-setting table pairs the endpoint grid, state cap, witness levels, passes and tolerance. Each predictor’s work cap appears separately. The screening rows in the frontier CSV give single-run observations. '
                 'Confirmation uses its own independently timed workers. Each configuration retains '
                 'its statuses, completed endpoint count and reached times. The raw records hold the '
                 'trial-level diagnostics. [All configurations](configurations.csv) · '
                 f'[Raw worker measurements]({raw_link}/measurements.json) · '
                 f'[Run environment]({raw_link}/environment.json)', ''])
    (out / 'report.md').write_text('\n'.join(text))


def analyze(run, out):
    env = json.loads((run / 'environment.json').read_text())
    grid = json.loads((run / 'grid.json').read_text())
    raw = json.loads((run / 'measurements.json').read_text())
    configs = summarize_configs(grid, raw, env)
    tuned, shared = frontiers(configs)
    out.mkdir(parents=True, exist_ok=True)
    # Source hashes are kept in the raw manifests; the flat table only carries
    # the validation outcome so long digest lists do not obscure measurements.
    config_rows = [{k: v for k, v in s.items() if k != 'source_manifest'} for s in configs]
    csv_file(out / 'configurations.csv', config_rows)
    csv_file(out / 'matched_accuracy.csv', tuned)
    csv_file(out / 'same_settings.csv', shared)
    plot(tuned, out)
    write_report(out, run, configs, tuned)
    return configs, tuned, shared


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    analyze(args.run.resolve(), (args.out or args.run / 'analysis').resolve())
