"""Serial matched positive-witness comparison with preparation-preconditioned PCG."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from math import sqrt
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch

from . import witness_pair_scaling as primary, witness_pair_supplement as supplement
from . import witness_pair_worker as paired, witness_stage
from .witness_block_stage import make_block_stage_floor

bm = primary.bm
MODULE = 'galerkin_sylvester_tdvp.witness_preconditioned_run'
STRICT = 1e-9


def candidates(target):
    return sorted({0.1, 0.01, sqrt(target), target, STRICT}, reverse=True)


def worker(cfg, folder):
    if cfg['method'] == 'A':
        return paired.worker(cfg, folder)
    from .witness_precond_solver import solve_velocity
    original = paired.witness.WitnessAudit
    anchor = None

    def capture(engine):
        nonlocal anchor
        anchor = engine.root_preconditioner
        return engine

    class AuditWithAnchor:
        def __new__(cls, *args, **kwargs):
            return capture(original(*args, **kwargs))

        @staticmethod
        def parity_chain(*args, **kwargs):
            return capture(original.parity_chain(*args, **kwargs))

    def stage(*args, **kwargs):
        if anchor is None:
            raise RuntimeError('Missing initial preparation anchor')
        return solve_velocity(*args, anchor=anchor, preconditioner=cfg['preconditioner'], **kwargs)

    with ExitStack() as patches:
        patches.enter_context(patch.object(paired.witness, 'WitnessAudit', AuditWithAnchor))
        patches.enter_context(patch.object(witness_stage, 'make_stage_floor', make_block_stage_floor))
        patches.enter_context(patch.object(paired, 'solve_velocity', stage))
        return paired.worker(cfg, folder)


def source_hashes():
    files = supplement.hashes('streamed-block')
    for name in ('witness_preconditioned_run.py', 'witness_anchor_preconditioner.py',
                 'witness_precond_solver.py'):
        path = Path(__file__).with_name(name)
        files[str(path.relative_to(bm.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def valid(row):
    return (primary.endpoint_score(row) <= row['cert_tol']
            and row.get('global_cp_bound') is not None
            and 0 <= row['global_cp_bound'] <= row['cert_tol']
            and row.get('accepted_steps') == row['steps'])


def run(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    hashes = source_hashes()
    settings = dict(sizes=args.sizes, families=args.families, targets=args.targets,
                    steps=args.steps, one_step_all=args.one_step_all, timeout=args.timeout, memory_mib=args.memory_mib,
                    tolerances={str(t): candidates(t) for t in args.targets})
    envfile = out / 'environment.json'
    if envfile.exists():
        env = json.loads(envfile.read_text())
        if env['settings'] != settings or env['source_files_sha256'] != hashes:
            raise ValueError('Source or settings changed. Use a separate run directory.')
    else:
        revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=bm.ROOT,text=True).strip()
        env = dict(source_sha=revision, source_files_sha256=hashes, settings=settings,
                   base_sha='2edcd4d398c0218e97adbcfb6803d2c50a545909', threads=primary.THREADS,
                   variant='preparation', python=sys.version, platform=platform.platform(),
                   numpy=bm.np.__version__, scipy=bm.scipy.__version__,
                   blas=bm.np.__config__.show(mode='dicts')['Build Dependencies']['blas'],
                   hardware_reference='../validation_20261119/hardware.json',
                   schedule='For each endpoint budget, screen A and cold preparation-PCG. Choose the '
                            'fastest valid B on production time, then run cold CG on this tolerance '
                            'and the strict reference tolerance. Confirm the selected A, B and '
                            'same-tolerance unpreconditioned control in three fresh processes. If '
                            'the two-interval A/B pair fails, repeat the grid at one interval. '
                            'With --one-step-all also execute and confirm the one-interval grid for every pair.')
        bm.save(envfile, env)
    rows, grid, selected = [], {}, {}
    confirm_configs = []

    def configure(family, n, mode, steps, target, rtol=STRICT):
        method = 'A' if mode == 'ambient' else 'B'
        cfg = supplement.configuration(family,n,method,steps,target,args)
        cfg.update(preconditioner=mode, variant='prepared_congruence_study',
                   rtol=rtol, source_sha=env['source_sha'], source_files_sha256=hashes,
                   base_sha=env['base_sha'])
        if method == 'B':
            cfg['config_id'] += f'_{mode}_r{rtol:.8g}'
        return cfg

    def execute(cfg, phase, rep=1):
        c = dict(cfg, phase=phase, requested_repeats=3 if phase=='confirm' else 1)
        key = phase, c['config_id']
        if key not in grid:
            grid[key] = c
            bm.save(out/'grid.json', list(grid.values()))
        if source_hashes() != hashes:
            raise RuntimeError('Source changed during the timed study')
        with patch.object(supplement, 'MODULE', MODULE):
            row = supplement.run_case(out,c,rep)
        rows.append(row)
        bm.save(out/'measurements.json', rows)
        bm.csv_table(out/'measurements.csv', rows)
        return row

    def screen(family, n, steps, target):
        a = configure(family,n,'ambient',steps,target)
        a_ok = valid(execute(a,'screen'))
        accepted = []
        for tol in candidates(target):
            c = configure(family,n,'preparation',steps,target,tol)
            r = execute(c,'screen')
            if valid(r):
                accepted.append((r['production_seconds'], c))
        best = min(accepted,key=lambda item:item[0])[1] if accepted else None
        controls = {}
        for tol in dict.fromkeys([STRICT] + ([best['rtol']] if best else [])):
            c = configure(family,n,'none',steps,target,tol)
            if valid(execute(c,'screen')):
                controls[tol] = c
        chosen = {'A': a if a_ok else None, 'B_pre': best,
                  'B_none': controls.get(best['rtol']) if best else None}
        key = f'{family}{n:02d}_s{steps}_tol{target:g}'
        selected[key] = {k:v['config_id'] for k,v in chosen.items() if v}
        confirm_configs.extend(v for v in chosen.values() if v)
        bm.save(out/'selection.json', selected)
        return a_ok and best is not None

    for target in args.targets:
        for n in args.sizes:
            for family in args.families:
                matched = screen(family,n,args.steps,target)
                if args.steps != 1 and (args.one_step_all or not matched):
                    screen(family,n,1,target)
    for rep in (1, 2, 3):
        ordered = confirm_configs if rep%2 else reversed(confirm_configs)
        for c in ordered:
            execute(c,'confirm',rep)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--sizes',nargs='+',type=int,default=[8,16,32,64])
    p.add_argument('--families',nargs='+',choices=['D','E'],default=['D','E'])
    p.add_argument('--targets',nargs='+',type=float,default=[1e-5,1e-7])
    p.add_argument('--steps',type=int,choices=[1,2],default=2)
    p.add_argument('--one-step-all',action='store_true',help='Also screen and confirm one interval for every pair')
    p.add_argument('--timeout',type=float,default=180)
    p.add_argument('--memory-mib',type=float,default=1536)
    p.add_argument('--resume',action='store_true')
    args = p.parse_args()
    args.extended_witness, args.residual_bond = True, 24
    if args.config:
        worker(json.loads(args.config.read_text()), args.out)
    else:
        if min(args.sizes)<2 or not all(0<t<1 for t in args.targets):
            p.error('Expected at least two spins and a tolerance in (0,1)')
        if any(len(values)!=len(set(values)) for values in (args.sizes,args.families,args.targets)):
            p.error('Duplicate settings would repeat a worker identity')
        run(args)


if __name__ == '__main__':
    main()
