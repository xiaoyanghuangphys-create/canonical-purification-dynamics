"""Bracket global canonical-root error using a separately executed, witnessed checkpoint."""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from io import BytesIO
import json
from pathlib import Path
from statistics import median
from zipfile import ZipFile

import numpy as np

from .models import tn
from .witness_pair_analysis import csv_file, finite, frontiers, summarize_configs
from .witness_pair_archive import SAFE_NAME

TARGET = 1e-5
TIGHT = 1e-7
PHYSICS_SOURCES = ('galerkin_sylvester_tdvp/models.py',
                   'scalable_cp_witness/code/scalable_cp_witness.py',
                   'galerkin_sylvester_tdvp/witness_pair_worker.py',
                   'scalable_cp_witness/code/tdvp_core.py')


def checkpoint(run, row):
    folder = row['directory']
    if not SAFE_NAME.fullmatch(folder):
        raise ValueError(f'Invalid worker directory {folder!r}')
    member = folder + '/last_accepted.npz'
    loose = run / member
    if loose.is_file():
        data = loose
    else:
        archive = run / 'raw' / f"{row['family']}{row['n']:02d}.zip"
        with ZipFile(archive) as z:
            data = BytesIO(z.read(member))
    n = row['n']
    with np.load(data, allow_pickle=False) as saved:
        if (saved['time'].item() != row['total']
                or saved['bound'].item() != row['global_cp_bound']):
            raise ValueError(f'Final time or bound differs from measurement for {run.name}/{folder}')
        if {k for k in saved.files if k.startswith('x')} != {f'x{i}' for i in range(n)}:
            raise ValueError(f'Incorrect number of root cores for {run.name}/{folder}')
        tensors = [saved[f'x{i}'] for i in range(n)]
    left = 1
    for t in tensors:
        if (t.ndim != 3 or t.shape[:2] != (left, 4) or t.shape[2] < 1
                or not np.isfinite(t).all()):
            raise ValueError(f'Invalid root core for {run.name}/{folder}')
        left = t.shape[2]
    if left != 1:
        raise ValueError(f'Open final bond for {run.name}/{folder}')
    state = tn.MPS(n, max_bond=max(t.shape[2] for t in tensors))
    state.tensors, state.center = tensors, n - 1
    return state


def interval(x, y, own_bound, ref_bound):
    """Intersect the loose certificate with a guarded triangle-inequality interval."""
    d = tn.distance(x, y)
    scale = max(1., tn.norm(x) + tn.norm(y))
    guard = float(64 * np.finfo(float).eps * x.n * (x.max_bond + y.max_bond) * scale)
    if not all(finite(v) for v in (d, guard, own_bound, ref_bound)):
        raise ValueError('Invalid distance, roundoff allowance, or witness radius')
    lower = max(0., d - ref_bound - guard)
    upper = min(own_bound, d + ref_bound + guard)
    if lower > upper:
        raise ValueError(f'Inconsistent witness radii and checkpoint distance {d:g}')
    return dict(distance=d, roundoff_guard=guard, lower=lower, upper=upper)


def cohort(run):
    env = json.loads((run / 'environment.json').read_text())
    grid = json.loads((run / 'grid.json').read_text())
    raw = json.loads((run / 'measurements.json').read_text())
    configs = summarize_configs(grid, raw, env)
    by_config = defaultdict(list)
    for r in raw:
        if r['phase'] == 'confirm' and r.get('parent_complete') is True:
            by_config[r['config_id']].append(r)
    qualified = {s['config_id'] for s in configs if s['phase'] == 'confirm' and s['qualified_endpoints']
                 and len(by_config[s['config_id']]) == s['requested_repeats']
                 and all(finite(r.get('global_cp_bound')) for r in by_config[s['config_id']])}
    tuned, _ = frontiers(configs, targets=(TARGET,))
    winners = []
    for row in tuned:
        if row['phase'] == 'confirm':
            for method in ('A', 'B'):
                key = row[method + '_config_id']
                if key is not None:
                    if key not in qualified:
                        raise ValueError(f'Incomplete parent records or witness radii for {run.name}/{key}')
                    winners.append(sorted(by_config[key], key=lambda r: r['repeat']))
    references = [r for key in sorted(qualified) for r in by_config[key]
                  if r['cert_tol'] < TARGET and r['global_cp_bound'] <= r['cert_tol']]
    fingerprint = tuple(env['source_files_sha256'][f] for f in PHYSICS_SOURCES)
    variant = env.get('settings', {}).get('variant', 'primary')
    return variant, fingerprint, winners, references


def physical(row):
    return row['family'], row['n'], row['total']


def budget(row):
    keys = ('steps', 'chi', 'work', 'levels', 'passes', 'rtol', 'atol', 'maxiter',
            'residual_bond', 'memory_mib', 'timeout_seconds')
    return {k: row[k] for k in keys if k in row}


def brackets(runs):
    """One tight checkpoint per physical endpoint, shared by all loosely evolved roots."""
    runs = [Path(p).resolve() for p in runs]
    if len({run.name for run in runs}) != len(runs):
        raise ValueError('Use distinct cohort directory names')
    states = []
    refs = defaultdict(list)
    fingerprints = set()
    for run in runs:
        variant, fingerprint, winners, candidates = cohort(run)
        fingerprints.add(fingerprint)
        states.extend((run, variant, group) for group in winners)
        for r in candidates:
            refs[physical(r)].append((run, variant, r))
    if len(fingerprints) > 1:
        raise ValueError('Physical model, witness, worker or root-norm sources differ between cohorts')
    def reference_order(candidate):
        path, _, row = candidate
        return (row['cert_tol'] > TIGHT, row['global_cp_bound'], path.name,
                row['config_id'], row['repeat'])
    selected = {key: min(candidates, key=reference_order) for key, candidates in refs.items()}
    roots, detail, omissions = {}, [], []
    for run, variant, group in states:
        cfg = group[0]
        key = physical(cfg)
        base = dict(family=cfg['family'], n=cfg['n'], total=cfg['total'], phase='confirm',
                    cohort=run.name, variant=variant, method=cfg['method'], config_id=cfg['config_id'],
                    cert_tol=cfg['cert_tol'], budget=budget(cfg))
        candidate = selected.get(key)
        if candidate is None or candidate[2]['cert_tol'] >= cfg['cert_tol']:
            omissions.append(dict(base, reason='No common confirmed reference with a stricter tolerance'))
            continue
        ref_run, ref_variant, ref = candidate
        if ref['global_cp_bound'] >= min(r['global_cp_bound'] for r in group):
            omissions.append(dict(base, reason='Reference witness radius does not improve every repeat'))
            continue
        if key not in roots:
            roots[key] = checkpoint(ref_run, ref)
        y = roots[key]
        ref_info = dict(reference_cohort=ref_run.name, reference_variant=ref_variant,
                        reference_method=ref['method'], reference_config_id=ref['config_id'],
                        reference_repeat=ref['repeat'], reference_phase=ref['phase'],
                        reference_directory=ref['directory'], reference_cert_tol=ref['cert_tol'],
                        reference_bound=ref['global_cp_bound'], reference_budget=budget(ref),
                        reference_production_seconds=ref['production_seconds'],
                        reference_measured_error=ref.get('global_cp_error'))
        for r in group:
            result = interval(checkpoint(run, r), y, r['global_cp_bound'], ref['global_cp_bound'])
            score = r.get('global_cp_error')
            inside = None if score is None else result['lower'] <= score <= result['upper']
            if inside is False:
                raise ValueError(f"Independent score {score:g} lies outside bracket for {run.name}/{r['directory']}")
            detail.append(dict(base, repeat=r['repeat'], directory=r['directory'],
                               production_seconds=r['production_seconds'], own_bound=r['global_cp_bound'],
                               **ref_info, **result, measured_error=score, measured_inside=inside))
    return detail, omissions


def outward(value, up):
    """Print three significant figures with the rounding directed out of the interval."""
    if not value:
        return '0'
    x = Decimal.from_float(value)
    place = Decimal(1).scaleb(x.adjusted() - 2)
    return format(x.quantize(place, rounding=ROUND_CEILING if up else ROUND_FLOOR), '.2E')


def write_summary(out, rows, omissions):
    grouped = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in ('family', 'n', 'total', 'phase', 'variant', 'cohort', 'method'))
        grouped[key].append(r)
    lines = ['# Global canonical-root error brackets', '',
             'Each row covers the three confirmation repeats of the fastest qualifying setting '
             'at target 1e-5 in that cohort, chosen by median production time. The table shows '
             'the smallest lower and largest upper endpoint across those repeats. Times cover '
             'the evolution and positive witness. Scoring is offline.', '',
             'For a stored root x and a tighter root y, d is the Frobenius norm of x − y, '
             'evaluated by orthogonalizing their tensor-train difference. The true canonical root '
             'at the same physical time is R. Its distance from y is bounded by the final '
             'witness radius e. The triangle inequality encloses the error of x between '
             'max(0, d − e − g) and min(own witness bound, d + e + g). Both witness bounds '
             'and floating-point accuracy are assumptions. The numerical guard g is '
             '64 × double precision epsilon × spin count × sum of the maximum bond dimensions '
             '× max(1, ‖x‖ + ‖y‖). It is a numerical allowance without interval-arithmetic verification.', '',
             'One common separately executed confirmed reference is selected per family, size and time '
             'across cohorts, prioritizing tolerance 1e-7 or tighter and then the smallest witness radius. '
             'Every displayed loose repeat has a strictly larger radius and a looser configured tolerance. '
             'Measured errors use the independent dense root on eight-spin D chains and the analytic '
             'root on E chains. References, configuration budgets and individual repeats are listed '
             'in the [checkpoint comparisons](brackets.csv).', '',
             '| Family | Spins | T | Phase | Variant / cohort | Method | Repeats | Range across repeats | '
             'Measured range | Production s (median) |', '|---|---:|---:|---|---|---|---:|---|---|---:|']
    for key, group in sorted(grouped.items()):
        f, n, time, phase, variant, run, method = key
        lo, hi = min(r['lower'] for r in group), max(r['upper'] for r in group)
        measured = [r['measured_error'] for r in group if r['measured_error'] is not None]
        score = f'{min(measured):.3g}–{max(measured):.3g}' if measured else '—'
        lines.append(f'| {f} | {n} | {time:g} | {phase} | {variant} / {run} | {method} | {len(group)} '
                     f'| [{outward(lo, False)}, {outward(hi, True)}] | {score} | '
                     f"{median(r['production_seconds'] for r in group):.3g} |")
    if omissions:
        lines.extend(['', '## Settings without a bracket', ''])
        for r in omissions:
            lines.append(f"- {r['cohort']}, {r['config_id']}. {r['reason']}.")
    (out / 'report.md').write_text('\n'.join(lines) + '\n')


def analyze(runs, out):
    rows, omissions = brackets(runs)
    out.mkdir(parents=True, exist_ok=True)
    csv_file(out / 'brackets.csv', rows)
    write_summary(out, rows, omissions)
    return rows, omissions


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    analyze(args.runs, args.out)
