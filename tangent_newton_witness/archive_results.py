"""Package completed stress, auxiliary control and contextual factor workers."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def read(path):
    return json.loads(path.read_text())


def expected_workers(cohort):
    if cohort.name.startswith('stress_'):
        return [f'{i:03d}_{c["family"]}{c["n"]}_{c["label"]}'
                for i,c in enumerate(read(cohort/'manifest.json')['configs'])]
    if cohort.name.startswith('controls_'):
        return (['adaptive_old','adaptive_new','pure_new','pure_gn'] +
                [f'{x}_{m}' for x,m in itertools.product(('D','E','Dstress'),('A','B'))])
    if cohort.name in ('context_01','context_02'):
        native = 'factor_svd' if cohort.name=='context_01' else 'factor_explicit'
        return [f'{x}_{m}' for x,m in itertools.product(('D','E'),(native,'factor_lsqr'))]
    raise ValueError('Expected stress, controls or historical context cohort')


def outcome(cohort, worker):
    folder=cohort/worker
    common=dict(worker=worker,records=None,actual_error=None,certificate=None)
    if cohort.name.startswith('stress_') or cohort.name.startswith('context_'):
        name='row.json' if cohort.name.startswith('stress_') else 'measurement.json'
        d=read(folder/name)
        if d['status'] in ('running','started'):
            raise ValueError(f'Unfinished worker {folder}')
        if 'checkpoints' in d:
            count=len(d['checkpoints'])
            if d['status']=='completed' and count!=d['steps']:
                raise ValueError(f'Incomplete trajectory {folder}')
            if len(list(folder.glob('endpoint_*.json')))!=count:
                raise ValueError(f'Endpoint and row counts disagree at {folder}')
        else:
            count=d.get('accepted_steps')
        return dict(common,result=f'{worker}/{name}',status=d['status'],records=count,
                    intended_time=d.get('total'),reached_time=d.get('last_accepted_time',d.get('reached_time')),
                    actual_error=d.get('global_cp_error'),certificate=d.get('global_cp_bound'))
    if worker.startswith('pure_'):
        name='pure.json'
        d=read(folder/name)
        reached=d['reached']
        step=d['checkpoints'][-1]
        passed=step['info']['accepted'] and reached==d['total']
        return dict(common,result=f'{worker}/{name}',status='completed' if passed else 'uncertified',
                    intended_time=d['total'],reached_time=reached,records=len(d['checkpoints']),
                    actual_error=step['error'] if passed else None,certificate=step['info']['bound'])
    if worker.startswith('adaptive_'):
        name='attempts.json'
        d=read(folder/name)
        return dict(common,result=f'{worker}/{name}',status='completed' if d['accepted'] else 'uncertified',
                    intended_time=d['attempts'][0]['end'],reached_time=d['last_accepted_time'],
                    records=len(d['attempts']),actual_error=d.get('actual_error'),
                    certificate=next((a['info'].get('bound') for a in reversed(d['attempts'])
                                      if a['info']['accepted']),None))
    name='raw.json'
    d=read(folder/name)
    steps=d['rows']
    latest=next((s for s in reversed(steps) if 'time' in s),{})
    complete=len(steps)==d['steps'] and steps[-1]['status']=='raw_endpoint'
    return dict(common,result=f'{worker}/{name}',status='raw_completed' if complete else 'raw_failed',
                intended_time=d['total'],reached_time=latest.get('time',0.),records=len(steps),
                actual_error=latest.get('error'),certificate=latest.get('carry_witness_bound'))


def package(cohort):
    workers=sorted(expected_workers(cohort))
    if sorted(p.name for p in cohort.iterdir() if p.is_dir())!=workers:
        raise ValueError(f'Worker directories disagree with grid at {cohort}')
    rows=[outcome(cohort,worker) for worker in workers]
    dest=cohort/'workers.zip'
    index=cohort/'index.tsv'
    audit=cohort/'archive_manifest.json'
    if any(p.exists() for p in (dest,index,audit)):
        raise FileExistsError('A published archive already exists inside '+str(cohort))
    members={}
    with ZipFile(dest,'x',ZIP_DEFLATED,compresslevel=7) as z:
        for name in workers:
            for f in sorted((cohort/name).rglob('*')):
                if f.is_file():
                    key=f.relative_to(cohort).as_posix()
                    members[key]=digest(f.read_bytes())
                    z.write(f,key)
    with ZipFile(dest) as z:
        if z.testzip() is not None or set(z.namelist())!=set(members):
            raise IOError('ZIP inventory/CRC mismatch')
        for k,h in members.items():
            if digest(z.read(k))!=h or digest((cohort/k).read_bytes())!=h:
                raise IOError('Worker changed or ZIP verification failed: '+k)
    with index.open('x') as out:
        w=csv.DictWriter(out,fieldnames=('worker','status','intended_time','reached_time','records',
                                         'actual_error','certificate','result'),delimiter='\t')
        w.writeheader()
        w.writerows(rows)
    audit.write_text(json.dumps(dict(workers=len(rows),files=len(members),zip_sha256=digest(dest.read_bytes()),
        index_sha256=digest(index.read_bytes()),member_sha256=members),indent=1)+'\n')
    print(cohort.name,len(rows),len(members),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cohort',type=Path,nargs='+')
    args=p.parse_args()
    for cohort in args.cohort:
        package(cohort)
