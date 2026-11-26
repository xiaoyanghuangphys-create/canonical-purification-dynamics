"""Run the prespecified analytic and independent dense checks, or replot their data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
from factor_tdvp import tangent_solver as ts
from . import oracles as o

HERE = Path(__file__).resolve().parent
DESIGN = dict(dense_n=[2,3,4,6,8], dense_physical_dimension_cap=256,
              dense_gate_element_cap=65536, cmi_chain_sizes=[3,4,6,8,16,32,64,128,256,512],
              horizons=[0.,.1,.3,1.,2.,4.,8.], controls=[[.6,.4],[0.,.4],[.1,.4],[.85,.4],[.6,0.]],
              primary_p=.6, primary_b0=.4, subdivisions=[1,2,5,10,20,50,100],
              historical_grid=[0,.02,.1,.3,.6,1.], tolerance=3e-12,
              dense_width_rule="1, 2, 3, 5, 7, N//2, N-1 and N when 1 <= width <= N",
              cmi_width_rule="A={1}, contiguous B of every length 1 through N-2, nonempty exterior")


def save_rows(out, name, rows):
    with (out/name).open('x', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify(label, value, tol=3.e-12):
    if not np.isfinite(value) or abs(value)>tol:
        raise AssertionError(f"{label} discrepancy {value:.6g} exceeds {tol:.6g}")


def dense_checks():
    p,b0=DESIGN['primary_p'],DESIGN['primary_b0']
    b1=b0*np.exp(-1.)
    result=[]; dense_cmi_rows=[]
    for n in DESIGN['dense_n']:
        rho0=o.dense_density(n,p,b0)
        rho1=o.dense_density(n,p,b1)
        y0,y1=o.dense_power(rho0,.5),o.dense_power(rho1,.5)
        channel_resid=np.linalg.norm(o.dense_dephase(rho0,1.)-rho1)
        verify('physical Kraus step',channel_resid)
        root_resid=np.linalg.norm(ts.matrix(o.root_mpo(n,p,b1))-y1)
        verify('canonical root',root_resid)
        for t,rho in [(0,rho0),(1,rho1)]:
            for k in range(1,n-1):
                theory=o.cmi(p,b0*np.exp(-t))
                measured=o.dense_cmi(rho,[0],list(range(1,k+1)),list(range(k+1,n)))
                verify('dense CMI',measured-theory)
                dense_cmi_rows.append(dict(n=n,t=t,buffer=k,exterior=n-k-1,analytic=theory,
                                           dense=measured,abs_difference=abs(theory-measured)))
        for width in sorted({w for w in (1,2,3,5,7,n//2,n-1,n) if 1<=w<=n}):
            whole=width==n
            floor=0. if whole else o.unitary_floor(p,b0,b1)
            sv=np.linalg.svd(o.interleaved(y1).reshape(4**width,-1),compute_uv=False)
            w=o.root_coefficients(p,b1)[1]**2
            weights=[1.] if whole else [np.sqrt(1-w),np.sqrt(w)]
            schmidt=np.linalg.norm(sv-np.pad(weights,(0,len(sv)-len(weights))))
            verify('Schmidt spectrum',schmidt)
            y_align=o.explicit_optimal_action(y0,y1,width,p,b0,b1)
            align=np.linalg.norm(y_align-o.interleaved(y1).reshape(4**width,-1))
            verify('explicit unitary',align-floor)
            gate_status='size_guard'; polar=None; polar_diff=None
            if 16**width<=DESIGN['dense_gate_element_cap']:
                polar,_=o.dense_optimum(y0,y1,width,max_gate_elements=DESIGN['dense_gate_element_cap'])
                polar_diff=abs(polar-floor); verify('unitary polar',polar_diff)
                gate_status='completed'
            y_hp,_=o.dense_hp_update(y0,rho0,1.,width)
            y_split,r_split=o.dense_hp_update(y0,rho0,.37,width)
            y_split,_=o.dense_hp_update(y_split,r_split,.63,width)
            telescoping=np.linalg.norm(y_split-y_hp); verify('half-Petz composition',telescoping)
            raw=np.linalg.norm(y_hp-y1); length=np.linalg.norm(y_hp)
            skew=np.linalg.norm(y_hp-y_hp.conj().T)
            normalized=np.linalg.norm(y_hp/length-y1)
            ident,par,trans=o.dense_operators(n)
            # QP is real antisymmetric and has unit Hilbert--Schmidt norm after scaling.
            qp=np.vdot(trans@par/np.sqrt(len(ident)),y_hp)
            expect=dict(raw_error=0,norm=1,hermiticity_defect=0,normalized_error=0,
                        qp_coefficient=0) if whole else o.half_petz_metrics(p,b0,b1)
            for label,val in [('raw_error',raw),('norm',length),('hermiticity_defect',skew),
                              ('normalized_error',normalized),('qp_coefficient',qp)]:
                verify('half-Petz '+label,val-expect[label])
            result.append(dict(n=n,width=width,exterior=n-width,t=1,
                dense_physical_channel_residual=channel_resid,dense_root_residual=root_resid,
                dense_schmidt_residual=schmidt,analytic_unitary_floor=floor,
                explicit_unitary_error=align,explicit_unitary_discrepancy=abs(align-floor),
                dense_polar_status=gate_status,dense_polar_error=polar,dense_polar_discrepancy=polar_diff,
                hp_raw_dense=raw,hp_raw_analytic=expect['raw_error'],hp_raw_discrepancy=abs(raw-expect['raw_error']),
                hp_normalized_dense=normalized,hp_norm_dense=length,hp_skew_dense=skew,
                hp_qp_real_dense=qp.real,hp_qp_imag_dense=qp.imag,hp_two_step_change=telescoping))
    return result,dense_cmi_rows


def information_rows():
    p,b0=DESIGN['primary_p'],DESIGN['primary_b0']
    rows=[]
    for subdivisions in DESIGN['subdivisions']+['historical']:
        grid=DESIGN['historical_grid'] if subdivisions=='historical' else np.linspace(0,1,subdivisions+1)
        for step,(left,right) in enumerate(zip(grid[:-1],grid[1:]),1):
            b=b0*np.exp(-left); bp=b0*np.exp(-right)
            before,after=o.cmi(p,b),o.cmi(p,bp); loss=before-after
            bound=np.sqrt(max(0.,loss)); floor=o.unitary_floor(p,b,bp)
            if floor>bound+1.e-14:
                raise AssertionError("Local unitary exceeds square root of CMI loss")
            rows.append(dict(subdivisions=subdivisions,step=step,t0=left,t1=right,
                             active_cmi_before=before,active_cmi_after=after,active_cmi_loss=loss,
                             sqrt_cmi_loss_bound=bound,active_optimal_unitary_error=floor,
                             inactive_site_cmi_loss=0,inactive_identity_gate_error=0))
    return rows


def analytic_rows():
    rows=[]; cmi_rows=[]
    for p,b0 in DESIGN['controls']:
        for t in DESIGN['horizons']:
            b=b0*np.exp(-t)
            r=np.hypot(p,b)
            w=o.root_coefficients(p,b)[1]**2
            hp=o.half_petz_metrics(p,b0,b)
            rows.append(dict(p=p,b0=b0,t=t,b=b,condition_number=(1+r)/(1-r),cmi=o.cmi(p,b),
                             lost_cmi=o.cmi(p,b0)-o.cmi(p,b),initial_w=o.root_coefficients(p,b0)[1]**2,
                             final_w=w,proper_unitary_floor=o.unitary_floor(p,b0,b),whole_unitary_floor=0.,
                             **{'proper_hp_'+k:v for k,v in hp.items()},whole_hp_error=0.))
    for n in DESIGN['cmi_chain_sizes']:
        for t in [0.,1.]:
            v=o.cmi(.6,.4*np.exp(-t))
            for k in range(1,n-1):
                cmi_rows.append(dict(n=n,t=t,buffer=k,exterior=n-k-1,cmi_nats=v))
    return rows,cmi_rows


def locality_counterexample():
    n=8;p=.6;b=.4*np.exp(-1.)
    rho=o.dense_density(n,p,b); modified=o.dense_density(n,-p,b)
    marginal=max(np.linalg.norm(o.marginal(rho-modified,[i for i in range(n) if i!=missing]))
                 for missing in range(n))
    cp=np.linalg.norm(o.dense_power(rho,.5)-o.dense_power(modified,.5))
    predicted=2*abs(o.root_coefficients(p,b)[1])
    verify("global counterexample",cp-predicted);verify("all proper marginals",marginal)
    return dict(n=n,t=1.,true_p=p,alternative_p=-p,b=b,
                max_physical_marginal_difference_over_single_spin_removals=float(marginal),
                global_cp_error=float(cp),analytic_global_cp_error=float(predicted))


def figure(folder,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def read(name):
        with (folder/name).open() as f:
            return list(csv.DictReader(f))
    analytic=read('buffer_cmi.csv');dense=read('dense_cmi.csv')
    fig,ax=plt.subplots(figsize=(6.45,3.38),layout='constrained')
    colors=['#174b70','#b14e15']
    for t,color in enumerate(colors):
        line=[r for r in analytic if int(r['n'])==512 and float(r['t'])==t]
        ax.plot([int(r['buffer']) for r in line],[float(r['cmi_nats']) for r in line],
                color=color,lw=1.9,label=rf'$t={t}$, analytic')
        points=[r for r in dense if int(r['t'])==t]
        ax.scatter([int(r['buffer']) for r in points],[float(r['dense']) for r in points],
                   marker='o' if not t else 's',s=35,facecolor='white',edgecolor=color,
                   zorder=3,label=rf'$t={t}$, dense ($N\leq8$)')
    ax.set_xscale('log',base=2)
    ax.set_xticks([1,2,4,8,16,32,64,128,256,510],['1','2','4','8','16','32','64','128','256','510'])
    ax.set_xlabel('Buffer length $|B|$ (spins)')
    ax.set_ylabel(r'Physical $I(E:S\mid B)$ (nats)')
    ax.set_ylim(.187,.216);ax.set_xlim(.83,620)
    ax.grid(axis='y',alpha=.22);ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,ncol=2,fontsize=8,loc='upper center')
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--out',required=True,type=Path,help='New immutable output directory, or existing input for --plot-only')
    parser.add_argument('--figure',type=Path)
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    if args.plot_only:
        if not args.figure: parser.error('--plot-only requires --figure')
        figure(args.out,args.figure);return
    args.out.mkdir(parents=True,exist_ok=False)
    metadata=dict(design=DESIGN,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],
                  cwd=HERE,text=True).strip(),source_hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (Path(__file__),HERE/'oracles.py',HERE/'tests/test_oracles.py')},
                  python=sys.version,numpy=np.__version__,platform=platform.platform(),
                  blas_env={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS',
                                                            'VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS')})
    (args.out/'design.json').write_text(json.dumps(metadata,indent=2)+'\n')
    started=perf_counter()
    try:
        dense,dcmi=dense_checks()
        save_rows(args.out,'dense_validation.csv',dense);save_rows(args.out,'dense_cmi.csv',dcmi)
        floors,cmi=analytic_rows()
        save_rows(args.out,'analytic_floors.csv',floors);save_rows(args.out,'buffer_cmi.csv',cmi)
        infos=information_rows()
        save_rows(args.out,'information_steps.csv',infos)
        example=locality_counterexample()
        (args.out/'locality_counterexample.json').write_text(json.dumps(example,indent=2)+'\n')
        summary=dict(status='completed',dense_widths=len(dense),physical_cmi_checks=len(dcmi),
                     dense_polar_checks=sum(r['dense_polar_status']=='completed' for r in dense),
                     guarded_polar_checks=sum(r['dense_polar_status']=='size_guard' for r in dense),
                     max_dense_cmi_discrepancy=max(r['abs_difference'] for r in dcmi),
                     max_dense_half_petz_raw_discrepancy=max(r['hp_raw_discrepancy'] for r in dense),
                     max_half_petz_two_step_change=max(r['hp_two_step_change'] for r in dense),
                     max_explicit_unitary_discrepancy=max(r['explicit_unitary_discrepancy'] for r in dense),
                     max_dense_polar_discrepancy=max(r['dense_polar_discrepancy'] or 0 for r in dense),
                     max_dense_root_discrepancy=max(r['dense_root_residual'] for r in dense),
                     max_dense_schmidt_discrepancy=max(r['dense_schmidt_residual'] for r in dense),
                     steps_checked=len(infos),analytic_cases=len(floors),buffer_cases=len(cmi),
                     validation_elapsed_s=perf_counter()-started)
        if args.figure: figure(args.out,args.figure)
    except Exception as exc:
        (args.out/'failure.json').write_text(json.dumps(dict(error=repr(exc),elapsed_seconds=perf_counter()-started),indent=2)+'\n')
        raise
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
