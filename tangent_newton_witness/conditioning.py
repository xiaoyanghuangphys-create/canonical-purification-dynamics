"""Small independent spectra and balanced linear solves at saved stress checkpoints."""
import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import scipy
from scipy.linalg import eigvalsh

from factor_tdvp.tangent_solver import matrix, matrix_to_mpo, tn
from galerkin_sylvester_tdvp.galerkin_solver import GalerkinError, _evidence
from .operator import HermitianFrame, solve_direction
from .stress import prepare


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(_evidence(data), allow_nan=False, indent=1)+'\n')


def projected_spectrum(w, power, mode):
    """Dense oracle. Form B and evaluate each derivative in W's eigenbasis."""
    f = HermitianFrame(w, mode)
    eig, u = np.linalg.eigh(matrix(w))
    weights = eig[:, None]+eig
    if power == 4:
        weights *= eig[:, None]**2+eig**2
    columns = np.column_stack([(u.conj().T@matrix(f.decode(e))@u).ravel()
                               for e in np.eye(f.dimension)])
    a = (columns.conj().T@(weights.ravel()[:, None]*columns)).real
    spectrum = np.linalg.eigvalsh(a)
    diag = np.sqrt(np.diag(a))
    pre = np.linalg.eigvalsh(a / (diag[:, None]*diag))
    return dict(input_eigenvalues=eig, full_condition=weights.max()/weights.min(),
                dimension=f.dimension, active_ranks=f.ranks, projected_eigenrange=spectrum[[0,-1]],
                projected_condition=spectrum[-1]/spectrum[0],
                jacobi_condition=pre[-1]/pre[0],
                orthogonality_defect=np.linalg.norm(columns.conj().T@columns-np.eye(f.dimension)))


def quadratic():
    rng = np.random.default_rng(482)
    def herm():
        a = rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
        return (a+a.conj().T)/2
    w = np.eye(4)+.15*herm()
    x = w+.025*herm()
    rows=[]
    for power in (2,4):
        trial=x.copy()
        target=matrix_to_mpo(np.linalg.matrix_power(w,power),2)
        errors=[]
        for _ in range(3):
            errors.append(np.linalg.norm(trial-w))
            current=matrix_to_mpo(trial,2)
            k, _=solve_direction(current,target,power,lower_bound=.99*eigvalsh(trial)[0],rtol=1e-11)
            trial+=matrix(k)
        errors.append(np.linalg.norm(trial-w))
        rows.append(dict(power=power,errors=errors,quadratic_ratios=np.array(errors[1:])/np.array(errors[:-1])**2))
    return rows


def read_checkpoint(path, n):
    cohort = path.parent.parent
    member = path.relative_to(cohort).as_posix()
    if path.is_file():
        blob = path.read_bytes()
    else:
        with ZipFile(cohort/'workers.zip') as archive:
            blob = archive.read(member)
    checksum = hashlib.sha256(blob).hexdigest()
    inventory = cohort/'archive_manifest.json'
    if inventory.is_file():
        expected = json.loads(inventory.read_text())['member_sha256'][member]
        if checksum != expected:
            raise OSError(f'Checkpoint disagrees with archived SHA-256: {member}')
    with np.load(BytesIO(blob), allow_pickle=False) as data:
        cores=[data[f'w{j}'] for j in range(n)]
        factor=tn.MPS(n,max_bond=max(x.shape[-1] for x in cores))
        factor.tensors,factor.center=cores,n-1
        return factor,float(data['time']),checksum


def run(args):
    out=args.out
    out.mkdir(parents=True,exist_ok=False)
    configs=[('product',2,None),('product',4,None),('loss',2,None),
             ('product',2,'005_product2_New-only'),('product',4,'013_product4_New-only'),
             ('strong-D',2,'015_strong-D2_New-only'),('loss',2,'018_loss2_New-only')]
    manifest=dict(command=sys.argv,base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  machine=platform.platform(), architecture=platform.machine(), python=sys.version,
                  numpy=np.__version__,scipy=scipy.__version__,seed=482,rtol=1e-8,max_elements=2_000_000,
                  repetitions=4,threads={k:os.environ.get(k) for k in
                  ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
                  hashes={p:sha(Path(__file__).with_name(p)) for p in ('operator.py','conditioning.py','stress.py')})
    save(out/'manifest.json',manifest)
    save(out/'quadratic.json',quadratic())
    with (out/'rows.jsonl').open('x') as f:
        for family,n,where in configs:
            c=dict(n=n,family=family)
            engine,initial,at=prepare(c)
            path=args.checkpoints/where/'last_accepted.npz' if where else None
            w,t,checkpoint_hash=read_checkpoint(path,n) if path else (initial[0],0.,None)
            target=at(t+.5)
            for mode in ('one','two'):
                started=perf_counter()
                spectra=projected_spectrum(w,4,mode)
                vals=spectra['input_eigenvalues']
                true_floor=vals[0]-32*np.finfo(float).eps*len(vals)*max(abs(vals))
                dense_rho,dense_init=matrix(target),matrix(engine.physical)
                ref_ratio=eigvalsh(dense_rho,dense_init)
                solve_args=dict(mode=mode,lower_bound=true_floor,rtol=1e-8)
                scoring=perf_counter()-started
                for rep in range(4):
                    for pre in (('none','jacobi') if rep%2==0 else ('jacobi','none')):
                        row=dict(family=family,n=n,checkpoint=str(path) if path else 'initial',
                                 checkpoint_sha256=checkpoint_hash,
                                 time=t,target_time=t+.5,mode=mode,preconditioner=pre,
                                 repetition=rep+1,spectra=spectra,target_over_anchor_range=ref_ratio[[0,-1]],
                                 dense_setup_scoring_seconds_per_mode=scoring)
                        started=perf_counter()
                        try:
                            k,info=solve_direction(w,target,4,preconditioner=pre,**solve_args)
                            row.update(status='solved',seconds=perf_counter()-started,linear=info)
                            # Independent scoring only. No positive branch is used in the solve.
                            raw=matrix(w)+matrix(k)
                            norm_w=raw/(np.linalg.norm(raw@raw)**.5)
                            row.update(full_trial_after_normalization=np.linalg.norm(np.linalg.matrix_power(norm_w,4)-dense_rho),
                                       raw_trial_residual=np.linalg.norm(np.linalg.matrix_power(raw,4)-dense_rho))
                        except GalerkinError as error:
                            row.update(status=error.status,error=str(error),details=error.details,seconds=perf_counter()-started)
                        f.write(json.dumps(_evidence(row),allow_nan=False)+'\n')
                        f.flush()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,default=Path('tangent_newton_witness/results/stress_late_01'))
    run(p.parse_args())
