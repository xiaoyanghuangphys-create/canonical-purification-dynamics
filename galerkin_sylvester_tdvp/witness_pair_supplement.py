"""Fresh serial witnessed A/B comparison with the block-congruence stage proof.

Default grid: D64, one and two retracted midpoint steps, two witness tolerances.
Each accepted winner and its qualifying same-grid paired counterpart receive
three fresh confirmation workers. Artifacts use the primary analyzer's schema.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch
from uuid import uuid4

from . import witness_pair_scaling as primary
from . import witness_pair_worker as paired
from . import witness_stage
from .witness_block_stage import make_block_stage_floor

bm=primary.bm
MODULE='galerkin_sylvester_tdvp.witness_pair_supplement'
STREAM_MODULE='witness_streaming'
STANDARD_RESIDUAL_BOND=24


def hashes(variant):
    result=primary.hashes()
    files=['witness_pair_supplement.py','witness_block_stage.py']
    if variant=='streamed-block':
        files.append(STREAM_MODULE+'.py')
    for name in files:
        path=Path(__file__).with_name(name)
        result[str(path.relative_to(bm.ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def worker(cfg, folder):
    # These substitutions live only inside a fresh, single-case process. Keep
    # the predictor controller, witness decisions, timer boundaries and scoring
    # byte-identical to the primary worker.
    with ExitStack() as stack:
        if cfg['residual_bond']!=STANDARD_RESIDUAL_BOND:
            # Only a tight D-chain config can request a different cap. Construct
            # the original engine at the primary worker's timed setup call.
            factory=partial(paired.witness.WitnessAudit,residual_bond=cfg['residual_bond'])
            stack.enter_context(patch.object(paired.witness,'WitnessAudit',factory))
        if cfg['method']=='B':
            stack.enter_context(patch.object(witness_stage,'make_stage_floor',make_block_stage_floor))
            if cfg['variant']=='streamed-block':
                from importlib import import_module
                stream=import_module('.'+STREAM_MODULE, __package__)
                stack.enter_context(patch.object(paired, 'solve_velocity', stream.solve_velocity))
        return paired.worker(cfg,folder)


def configuration(family,n,method,steps,tol,args):
    cfg=primary.configuration(family,n,method,steps,tol,args)
    extended=args.extended_witness and family=='D' and tol<=1e-7
    if extended:
        cfg['levels'] += [[10,14,20],[12,16,24]]
    return dict(cfg,extended_witness=args.extended_witness,extended_witness_active=extended,
                residual_bond=args.residual_bond if extended else STANDARD_RESIDUAL_BOND)


def choose(screens, targets):
    """Screen winners at each target plus qualifying peers on the same grid."""
    selected=set()
    selection={}
    for family,n in sorted({(r['family'],r['n']) for r in screens}):
        for target in targets:
            eligible=[r for r in screens if (r['family'],r['n'])==(family,n)
                      and primary.endpoint_score(r)<=target]
            winners={}
            for m in primary.METHODS:
                options=[r for r in eligible if r['method']==m]
                if options:
                    winners[m]=min(options,key=lambda r:(r['production_seconds'],r['config_id']))
            selection[f'{family}{n}_{target:g}']={m:r['config_id'] for m,r in winners.items()}
            for winner in winners.values():
                for r in eligible:
                    if (r['steps'],r['cert_tol'])==(winner['steps'],winner['cert_tol']):
                        selected.add(r['config_id'])
    return selection,selected


def run_case(out,cfg,repeat):
    key=f"{cfg['phase']}_{cfg['config_id']}_r{repeat}"
    folder=out/key
    if folder.exists():
        previous=folder/'measurement.json'
        if previous.exists():
            row=json.loads(previous.read_text())
            if row.get('parent_complete') and all(row.get(k)==v for k,v in cfg.items()):
                return row
        # Keep partial logs/checkpoints from an interruption across resume.
        old=out/'interrupted'
        old.mkdir(exist_ok=True)
        folder.rename(old/(key+'_'+uuid4().hex))
    folder.mkdir()
    c=dict(cfg,directory=key,repeat=repeat)
    bm.save(folder/'config.json',c)
    cmd=[sys.executable,'-m',MODULE,'--config',str(folder/'config.json'),'--out',str(folder)]
    begin=perf_counter()
    with (folder/'worker.log').open('w') as log:
        try:
            proc=subprocess.run(cmd,cwd=bm.ROOT,env=dict(os.environ,**primary.THREADS),
                                stdout=log,stderr=subprocess.STDOUT,timeout=c['timeout_seconds'])
            rc=proc.returncode
        except subprocess.TimeoutExpired:
            rc=None
    elapsed=perf_counter()-begin
    row=dict(c)
    measured=folder/'measurement.json'
    if rc==0 and measured.exists():
        row.update(json.loads(measured.read_text()))
    else:
        if (folder/'progress.json').exists():
            p=json.loads((folder/'progress.json').read_text())
            row.update(p,phase=c['phase'],worker_phase=p.get('phase'))
        row.update(status='timeout' if rc is None else 'worker_failed',returncode=rc,
                   global_cp_bound=None,global_cp_error=None)
        if (folder/'memory_limit_event.json').exists():
            row.update(json.loads((folder/'memory_limit_event.json').read_text()))
    row.update(wall_seconds=elapsed,parent_complete=True)
    bm.save(measured,row)
    print(key,row['status'],row.get('global_cp_error'),row.get('global_cp_bound'),
          f'{elapsed:.2f}s',flush=True)
    return row


def run(args):
    if (min(args.sizes)<1 or min(args.steps)<1 or min(args.tolerances)<=0
            or min(args.timeout,args.memory_mib)<=0):
        raise ValueError('Sizes, steps, tolerances and resource budgets must be positive')
    if args.residual_bond < 1 or (args.residual_bond!=STANDARD_RESIDUAL_BOND and not args.extended_witness):
        raise ValueError('A custom residual bond must be positive and requires --extended-witness')
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=args.resume)
    fingerprint=hashes(args.variant)
    settings=dict(primary.settings(args),steps=args.steps,variant=args.variant,
                  extended_witness=args.extended_witness,residual_bond=args.residual_bond)
    env=out/'environment.json'
    if env.exists():
        manifest=json.loads(env.read_text())
        if manifest['settings']!=settings or manifest['source_files_sha256']!=fingerprint:
            raise ValueError('Settings or source changed; use a fresh directory')
    else:
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=bm.ROOT,text=True).strip()
        manifest=dict(source_sha=sha,source_files_sha256=fingerprint,base_sha=primary.BRANCH_BASE,
                      python=sys.version,platform=platform.platform(),cpu=platform.machine(),
                      threads=primary.THREADS,numpy=bm.np.__version__,scipy=bm.scipy.__version__,
                      blas=bm.np.__config__.show(mode='dicts')['Build Dependencies']['blas'],
                      settings=settings,wall_cap_seconds=args.timeout,peak_rss_guard_mib=args.memory_mib,
                      hardware_reference='../validation_20261119/hardware.json',
                      schedule='Run every screen serially. For each target and method select the cheapest '
                               'valid screening time. Confirm that setting and any target-qualifying peer '
                               'on its paired grid in three new processes; alternate confirmation order.')
        bm.save(env,manifest)
    configs=[dict(configuration(f,n,m,k,tol,args),variant=args.variant,base_sha=primary.BRANCH_BASE,
                  source_sha=manifest['source_sha'],source_files_sha256=fingerprint)
             for tol in args.tolerances for n in args.sizes for f in args.families
             for k in args.steps for m in primary.METHODS]
    if len({c['config_id'] for c in configs})!=len(configs):
        raise ValueError('Duplicate grid settings')
    screened=[dict(c,phase='screen',requested_repeats=1) for c in configs]
    bm.save(out/'grid.json',screened)
    rows=[]
    def execute(c,rep):
        if hashes(args.variant)!=fingerprint:
            raise RuntimeError('Numerical sources changed during this cohort')
        rows.append(run_case(out,c,rep))
        bm.save(out/'measurements.json',rows)
        bm.csv_table(out/'measurements.csv',rows)
    for c in screened:
        execute(c,1)
    selection,ids=choose(rows,primary.TARGETS)
    bm.save(out/'screen_selection.json',selection)
    confirmed=[dict(c,phase='confirm',requested_repeats=3) for c in configs if c['config_id'] in ids]
    bm.save(out/'grid.json',screened+confirmed)
    for rep in range(1,4):
        for c in (confirmed if rep%2 else reversed(confirmed)):
            execute(c,rep)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--families',nargs='+',choices=['D','E'],default=['D'])
    p.add_argument('--sizes',nargs='+',type=int,default=[64])
    p.add_argument('--steps',nargs='+',type=int,default=[1,2])
    p.add_argument('--tolerances',nargs='+',type=float,default=[1e-5,1e-7])
    p.add_argument('--variant',choices=['block','streamed-block'],default='block')
    p.add_argument('--extended-witness',action='store_true',
                   help='Append witness levels 10/14/20 and 12/16/24 on D chains at tolerance <= 1e-7')
    p.add_argument('--residual-bond',type=int,default=STANDARD_RESIDUAL_BOND,
                   help='Witness residual cap for extended D-chain levels and their earlier attempts (default 24)')
    p.add_argument('--timeout',type=float,default=180)
    p.add_argument('--memory-mib',type=float,default=1536)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.config:
        worker(json.loads(args.config.read_text()),args.out)
    else:
        run(args)

if __name__=='__main__':
    main()
