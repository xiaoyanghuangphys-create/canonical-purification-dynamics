"""Rebuild tables and plots using immutable worker archives and independent scoring."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import median
from zipfile import ZipFile
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .benchmark import METHODS, LOCAL, TARGETS, eligible
from .oracles import unitary_floor, half_petz_metrics

PKG = Path(__file__).resolve().parent
COLORS = dict(zip(METHODS, ('#50778e','#87a6bc','#963d2d','#d87c41','#644289','#197356','#1c548d')))


def csv_file(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(',',':')) if isinstance(v,(dict,list)) else v
                             for k,v in row.items()})


def load_data(results):
    workers, checkpoints = [], []
    for path in sorted(results.glob('*/measurements.jsonl')):
        run = path.parent
        manifest = json.loads((run/'manifest.json').read_text())
        if 'configs' not in manifest:
            continue  # Dedicated dense/control harness has its own schema.
        seal = json.loads((run/'archive.json').read_text())
        digest = hashlib.sha256((run/'workers.zip').read_bytes()).hexdigest()
        if digest != seal['sha256']:
            raise ValueError(f'Archive checksum mismatch in {run}')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != seal['workers'] or len(rows)!=len(manifest['configs'])*manifest['repeats']:
            raise ValueError(f'Incomplete grid in {run}')
        expected={(i,j) for i in range(len(manifest['configs'])) for j in range(1,manifest['repeats']+1)}
        if {(r['config_index'],r['repeat']) for r in rows} != expected:
            raise ValueError(f'Duplicate or missing worker in {run}')
        with ZipFile(run/'workers.zip') as archive:
            for row in rows:
                if any(row[k]!=v for k,v in manifest['configs'][row['config_index']].items()):
                    raise ValueError(f'Worker differs from manifest in {run}')
                cfg=json.loads(archive.read(row['worker']+'/config.json'))
                if cfg!=manifest['configs'][row['config_index']]:
                    raise ValueError(f'Archived worker configuration mismatch in {run}')
                row.update(run=run.name, suite=manifest['suite'], steps=len(row['times']),T=row['times'][-1],
                           exterior=row['n']-row['width'] if row['method'] in LOCAL else None)
                member=row['worker']+'/checkpoints.jsonl'
                ends = [json.loads(s) for s in archive.read(member).splitlines()] if member in archive.namelist() else []
                for end in ends:
                    diag=end['diagnostics']
                    independent = dict(end['actual'])
                    if row['method']=='Half-Petz':
                        a=independent
                        a['normalized_cp_error_diagnostic']=np.sqrt(max(0.,a['global_cp_error']**2-a['norm_drift']**2)/a['norm'])
                    checkpoints.append(dict(run=run.name,worker=row['worker'],method=row['method'],n=row['n'],
                        t=end['time'],p=row['p'],b0=row['b0'],W=row['width'],R=row['fit_width'],
                        protocol=row['protocol'],predictor_raw_cp_error=end['predictor_raw_cp_error'],
                        bound=diag.get('bound'),positivity_contribution=diag.get('positivity_contribution'),
                        weighted_residual=diag.get('weighted_residual'),correction_updates=diag.get('correction_updates'),
                        fit_updates=diag.get('fit_updates'),peak_intermediate_bond=diag.get('operations',{}).get('peak_intermediate_bond'),
                        **independent))
                for key in ('nominal_site_updates','applied_site_updates','skipped_site_updates','cache_updates',
                            'gate_seconds','apply_seconds','cache_seconds','audit_seconds'):
                    vals=[e['diagnostics'][key] for e in ends if key in e['diagnostics']]
                    if vals: row[key]=sum(vals)
                if ends:
                    last=ends[-1]
                    for key in ('norm','hermiticity_defect','density_trace_error','density_residual_relative',
                                'site1_physical_marginal_error','root_p_re'):
                        row[key]=last['actual'][key]
                    for key in ('max_inverse_norm','max_local_fit','max_joint_span','truncation_bound',
                                'peak_split_matrix_min_dim_bound','max_inactive_motion'):
                        vals=[e['diagnostics'].get(key) for e in ends if e['diagnostics'].get(key) is not None]
                        if vals: row[key]=sum(vals) if key=='truncation_bound' else max(vals)
                workers.append(row)
    return workers, checkpoints


def candidate_groups(rows, min_repeats=3):
    # Define each candidate by its immutable manifest and complete configuration.
    groups=defaultdict(list)
    for row in rows:
        if row['method'] in LOCAL and (row['fit_width']>=row['n'] or row['completion']!='native' or row['audit_inactive']):
            continue
        groups[row['run'],row['config_index']].append(row)
    for group in groups.values():
        if len(group)!=len({r['repeat'] for r in group}):
            raise ValueError('Duplicate repeat in candidate group')
        if len(group)>=min_repeats:
            yield group


def frontiers(rows, targets=TARGETS, min_repeats=3):
    cases=defaultdict(list)
    for group in candidate_groups(rows,min_repeats):
        r=group[0]
        for tol in targets:
            cases[r['n'],r['p'],r['b0'],r['T'],tol,r['method'],r['protocol']].append(group)
    result=[]
    for (n,p,b,t,tol,m,protocol),groups_ in sorted(cases.items()):
        success=[g for g in groups_ if all(eligible(r,tol) for r in g)]
        base=dict(n=n,p=p,b0=b,T=t,target=tol,method=m,protocol=protocol,settings_tested=len(groups_),
                  settings_successful=len(success))
        if success:
            group=min(success,key=lambda g: median(r['production_seconds'] for r in g))
            r=group[0]
            values=[r['production_seconds'] for r in group]
            base.update(status='attained',median_seconds=median(values),min_seconds=min(values),
                        max_seconds=max(values),worst_cp_error=max(r['global_cp_error'] for r in group),
                        median_scoring_seconds=median(r['scoring_seconds'] for r in group),
                        median_process_overhead_seconds=median(r['process_overhead_seconds'] for r in group),
                        repeats=len(group),run=r['run'],config_index=r['config_index'],steps=r['steps'],
                        cert_tol=r['cert_tol'],chi=r['chi'],W=r['width'] if m in LOCAL else None,
                        R=r['fit_width'] if m in LOCAL else None)
        else:
            base.update(status='no measured hit')
        result.append(base)
    return result


def plot(rows, output):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'legend.frameon':False,'figure.dpi':130,'axes.grid':True,'grid.alpha':.17})
    def figure(width=9,height=3.9):
        return plt.subplots(1,2,figsize=(width,height),layout='constrained')
    def save(fig,name):
        fig.savefig(output/(name+'.png'),dpi=160,bbox_inches='tight')
        plt.close(fig)
    def selected(suite):
        return [r for r in rows if r['suite']==suite and r['status']=='completed']
    def line(ax, data, x, y, name, color, style='-', mark='o'):
        data=sorted(data,key=lambda r:r[x])
        if data: ax.plot([r[x] for r in data],[max(r[y],1e-16) for r in data],
                         style,marker=mark,ms=4,c=color,label=name)
    hp=half_petz_metrics(.6,.4,.4*np.exp(-1))['raw_error']
    opt=unitary_floor(.6,.4,.4*np.exp(-1))
    fig,ax=figure()
    width=selected('width')
    for panel,n in zip(ax,[8,4]):
        subset=[r for r in width if r['n']==n]
        for label,keep,c,mark in [('Half-Petz  R=W',lambda r:r['method']==LOCAL[0],COLORS[LOCAL[0]],'o'),
             ('Local fit  R=W',lambda r:r['method']==LOCAL[1] and r['fit_width']==r['width'],COLORS[LOCAL[1]],'s'),
             ('Local fit  R=W+1' if n==8 else 'Local fit  R=N',lambda r:r['method']==LOCAL[1] and
                r['fit_width']!=r['width'],'#3393af','^')]:
            data=[r for r in subset if keep(r) and (n==4 or r['fit_width']<n)]
            if n==4 and label=='Local fit  R=N':
                data=[r for r in subset if r['method']==LOCAL[1] and r['fit_width']==n]
            line(panel,data,'width','global_cp_error',label,c,mark=mark)
        if n==8:
            line(panel,[r for r in subset if r['method']==LOCAL[1] and r['fit_width']==n],
                 'width','global_cp_error','Whole fitting data  R=N','#883971',mark='D')
        panel.axhline(opt,c='#a15137',ls=':',label='Optimal proper-W unitary')
        panel.axhline(hp,c='#226743',ls='--',label='Proper-W half-Petz')
        panel.set(title=f'{n} spins',xlabel='Active width W (sites)',ylabel='Raw global CP error',yscale='log')
        panel.legend(fontsize=7)
    save(fig,'width')
    fig,ax=figure(10)
    for method in LOCAL:
        for protocol,style in [('active','-'),('uniform','--')]:
            line(ax[0],[r for r in selected('refine') if r['method']==method and r['protocol']==protocol],
                 'steps','global_cp_error',method+' '+protocol,COLORS[method],style)
    ax[0].axhline(opt,color='#a15137',ls=':',label='Optimal fixed window')
    ax[0].set(xscale='log',yscale='log',xlabel='Intervals at T=1',ylabel='Raw global CP error',title='Time refinement, W=2')
    for m in METHODS:
        line(ax[1],[r for r in selected('horizon') if r['method']==m],'T','global_cp_error',m,COLORS[m])
    ts=np.geomspace(.1,8,100)
    ax[1].plot(ts,[unitary_floor(.6,.4,.4*np.exp(-t)) for t in ts],':',color='#333333',label='Optimal fixed window')
    ax[1].set(xscale='log',yscale='log',xlabel='Final time T (five intervals)',ylabel='Raw global CP error',title='Horizon dependence, W=1')
    for a in ax:a.legend(fontsize=7,ncol=2)
    save(fig,'time')
    fig,ax=figure(10)
    for m in METHODS:
        for protocol in (['active','uniform'] if m in LOCAL else ['generator']):
            sub=[r for r in selected('scale') if r['method']==m and r['protocol']==protocol]
            label=m+(' sweep' if protocol=='uniform' else '')
            for panel,y in zip(ax,['global_cp_error','production_seconds']):
                line(panel,sub,'n',y,label,COLORS[m],'--' if protocol=='uniform' else '-')
    ax[0].axvspan(110,540,color='#bbbbbb',alpha=.12)
    ax[0].text(.99,.52,'B stage guard stopped at 128–512',ha='right',va='top',
               transform=ax[0].transAxes,fontsize=7,bbox=dict(facecolor='white',edgecolor='none',alpha=.87))
    for a,title in zip(ax,['Raw global CP error','Production time (seconds)']):
        a.set(xscale='log',yscale='log',xlabel='Number of spins',ylabel=title)
        a.legend(ncol=2,fontsize=7)
    save(fig,'size')
    fig,ax=figure(10)
    settings=[g for g in candidate_groups([r for r in rows if r['suite']=='confirm' and r['n']==8])
              if all(eligible(r,float('inf')) for r in g)]
    for panel,kind in zip(ax,['global','local']):
        for m in METHODS:
            if (kind=='local')!=(m in LOCAL): continue
            for protocol in (['active','uniform'] if m in LOCAL else ['generator']):
                g=[g for g in settings if g[0]['method']==m and g[0]['protocol']==protocol]
                if g:
                    panel.scatter([max(r['global_cp_error'] for r in group) for group in g],
                                  [median(r['production_seconds'] for r in group) for group in g],
                                  color=COLORS[m],marker='x' if protocol=='uniform' else 'o',s=30,
                                  label=m+(' sweep' if protocol=='uniform' else ''))
        panel.set(xscale='log',yscale='log',xlabel='Achieved raw global CP error',ylabel='Median production time (s)',
                  title='Eight spins, '+('global methods' if kind=='global' else 'local lifts'))
        if kind=='local':
            panel.axvspan(1e-7,hp,color=COLORS[LOCAL[0]],alpha=.08,label='Below proper half-Petz plateau')
            panel.axvline(opt,ls=':',c=COLORS[LOCAL[1]],label='Best fixed-window unitary')
        panel.legend(fontsize=7)
    save(fig,'accuracy_cost')
    fig,a=plt.subplots(figsize=(9,4.1),layout='constrained')
    representatives=[g for g in settings if g[0]['steps']==1 and g[0]['cert_tol']==1e-6 and
                     g[0]['protocol']!='uniform']
    representatives.sort(key=lambda g:METHODS.index(g[0]['method']))
    names=[g[0]['method'] for g in representatives]
    left=np.zeros(len(names))
    parts=[]
    for label,key,color in [('Setup','setup_seconds','#697687'),('Predictor','predictor_seconds','#48799c'),
            ('Witness fit','fit_seconds','#9470a8'),('Correction','correction_seconds','#b28fb3'),
            ('Certification','certification_seconds','#795c86'),('Physical target','density_reference_seconds','#cb8966'),('Gate construction','gate_seconds','#519980'),
            ('Gate application','apply_seconds','#83b99e'),('Physical cache updates','cache_seconds','#d3ad6f'),
            ('Other production','other','#bbb7b0'),('Independent scoring','scoring_seconds','#ecdeb8')]:
        def seconds(r):
            stages=r.get('adapter_timings',{})
            mapping={k:stages.get(k,0.) for k in ('predictor_seconds','fit_seconds','correction_seconds',
                                                    'certification_seconds','density_reference_seconds')}
            if key=='other':
                return max(0.,r['production_seconds']-r['setup_seconds']-sum(mapping.values())-
                           sum(r.get(k,0.) for k in ('gate_seconds','apply_seconds','cache_seconds')))
            return mapping.get(key,r.get(key,0.))
        values=np.array([median(seconds(r) for r in group) for group in representatives])
        parts.append((label,color,values))
    totals=sum(x[2] for x in parts)
    for label,color,values in parts:
        share=100*values/totals
        a.barh(names,share,left=left,label=label,color=color)
        left+=share
    a.set(xlabel='Share of summed median stage times (%)',title='Eight spins, one interval, certificate request 10⁻⁶')
    a.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.15),fontsize=8)
    save(fig,'breakdown')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,default=PKG/'results')
    ap.add_argument('--out',type=Path,default=PKG/'analysis')
    ap.add_argument('--figures',type=Path,default=PKG/'figures')
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    args.figures.mkdir(parents=True,exist_ok=True)
    rows,ends=load_data(args.results)
    csv_file(args.out/'workers.csv',rows)
    csv_file(args.out/'endpoints.csv',ends)
    csv_file(args.out/'frontier.csv',frontiers(rows))
    failures=[r for r in rows if r['status']!='completed']
    csv_file(args.out/'failures.csv',failures)
    summary=dict(workers=len(rows),checkpoints=len(ends),outcomes=dict(Counter(r['status'] for r in rows)),
                 by_run={s:dict(Counter(r['status'] for r in rows if r['run']==s)) for s in sorted({r['run'] for r in rows})},
                 certificates=sum(e['bound'] is not None for e in ends),
                 max_actual_minus_bound=max((e['global_cp_error']-e['bound'] for e in ends if e['bound'] is not None),default=None))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot(rows,args.figures)


if __name__=='__main__':
    main()
