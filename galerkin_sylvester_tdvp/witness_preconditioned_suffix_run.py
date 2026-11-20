"""Fixed two-interval D64 conditioning follow-up with fourth-moment audit bounds."""
import argparse
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch

from . import witness_preconditioned_run as primary
from . import witness_precond_solver as solver
from . import witness_suffix_audit as suffix
from .witness_preconditioned_run import bm, supplement


MODULE = 'galerkin_sylvester_tdvp.witness_preconditioned_suffix_run'
TARGET = 1e-7
SIZE = 64
STEPS = 2


def source_hashes():
    hashes = primary.source_hashes()
    for name in ('witness_preconditioned_suffix_run.py','witness_suffix_audit.py'):
        source=Path(__file__).with_name(name)
        hashes[str(source.relative_to(bm.ROOT))]=sha256(source.read_bytes()).hexdigest()
    return hashes


def worker(cfg, folder):
    audit = partial(suffix.refined_audit_norms, max_rank=cfg['audit_rank_cap'],
                    diagnostic_relative_width=cfg['diagnostic_relative_width'])
    with patch.object(solver,'_audit_norms',audit):
        return primary.worker(cfg,folder)


def run(args):
    if args.audit_rank_cap<1 or not 1e-12<=args.diagnostic_relative_width<=1:
        raise ValueError('Expected a positive audit rank cap and relative radius in [1e-12,1]')
    out = args.out.resolve()
    out.mkdir(parents=True,exist_ok=args.resume)
    hashes = source_hashes()
    settings=dict(sizes=[SIZE],families=['D'],targets=[TARGET],steps=STEPS,
                  tolerances={str(TARGET): primary.candidates(TARGET)},one_step_all=False,
                  one_step_fallback=False,timeout=args.timeout,memory_mib=args.memory_mib,
                  audit_rank_cap=args.audit_rank_cap,diagnostic_relative_width=args.diagnostic_relative_width,
                  diagnostic_radius_rule='max(1e-12,min(0.01*rtol,diagnostic_relative_width))*generator_norm')
    env_path=out/'environment.json'
    if env_path.exists():
        env=json.loads(env_path.read_text())
        if env['settings']!=settings or env['source_files_sha256']!=hashes:
            raise ValueError('Source or settings changed. Use a separate run directory.')
    else:
        revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=bm.ROOT,text=True).strip()
        env=dict(source_sha=revision,source_files_sha256=hashes,settings=settings,
                 base_sha='2edcd4d398c0218e97adbcfb6803d2c50a545909',threads=primary.primary.THREADS,
                 variant='preparation_suffix_audit',python=sys.version,platform=platform.platform(),
                 numpy=bm.np.__version__,scipy=bm.scipy.__version__,
                 blas=bm.np.__config__.show(mode='dicts')['Build Dependencies']['blas'],
                 hardware_reference='../validation_20261119/hardware.json',
                 schedule='Fixed two-interval D64. Screen A and all cold preparation-PCG tolerances. '
                          'Select fastest qualifying B and screen cold unpreconditioned CG at its '
                          'tolerance and at 1e-9. Run A, winning B and matched-tolerance control '
                          'in three fresh confirmation processes. No one-interval enrollment.')
        bm.save(env_path,env)
    grid, records = {}, []
    def config(variant,rtol=primary.STRICT):
        method='A' if variant=='ambient' else 'B'
        c=supplement.configuration('D',SIZE,method,STEPS,TARGET,args)
        c.update(preconditioner=variant,variant='prepared_congruence_suffix_study',rtol=rtol,
                 source_sha=env['source_sha'],source_files_sha256=hashes,base_sha=env['base_sha'],
                 audit_rank_cap=args.audit_rank_cap,diagnostic_relative_width=args.diagnostic_relative_width)
        if method=='B':
            c['config_id']+=f'_{variant}_r{rtol:.8g}'
        return c
    def execute(c,phase,repeat=1):
        c=dict(c,phase=phase,requested_repeats=3 if phase=='confirm' else 1)
        key=(phase,c['config_id'])
        if key not in grid:
            grid[key]=c
            bm.save(out/'grid.json',list(grid.values()))
        if source_hashes()!=hashes:
            raise RuntimeError('Source changed during this cohort')
        with patch.object(supplement,'MODULE',MODULE):
            row=supplement.run_case(out,c,repeat)
        records.append(row)
        bm.save(out/'measurements.json',records)
        bm.csv_table(out/'measurements.csv',records)
        return row
    a=config('ambient')
    chosen={}
    if primary.valid(execute(a,'screen')):
        chosen['A']=a
    valid=[]
    for tol in primary.candidates(TARGET):
        c=config('preparation',tol)
        row=execute(c,'screen')
        if primary.valid(row):
            valid.append((row['production_seconds'],c))
    tolerances=[primary.STRICT]
    if valid:
        best=min(valid,key=lambda t:t[0])[1]
        chosen['B_pre']=best
        tolerances.append(best['rtol'])
    for tol in dict.fromkeys(tolerances):
        c=config('none',tol)
        if primary.valid(execute(c,'screen')) and 'B_pre' in chosen and tol==chosen['B_pre']['rtol']:
            chosen['B_none']=c
    bm.save(out/'selection.json',{f'D{SIZE:02d}_s{STEPS}_tol{TARGET:g}':
                                 {role:c['config_id'] for role,c in chosen.items()}})
    for rep in (1,2,3):
        order=list(chosen.values())
        for c in (order if rep%2 else reversed(order)):
            execute(c,'confirm',rep)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--config',type=Path,help=argparse.SUPPRESS)
    ap.add_argument('--timeout',type=float,default=180)
    ap.add_argument('--memory-mib',type=float,default=1536)
    ap.add_argument('--audit-rank-cap',type=int,default=384)
    ap.add_argument('--diagnostic-relative-width',type=float,default=1e-6,
                    help='Cap on diagnostic half-width relative to generator norm.')
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    args.extended_witness,args.residual_bond=True,24
    if args.config:
        worker(json.loads(args.config.read_text()),args.out)
    else:
        run(args)


if __name__=='__main__':
    main()
