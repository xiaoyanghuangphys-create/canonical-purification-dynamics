import csv
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import pytest

from parity_local_lift_comparison import global_methods as g, oracles
from parity_local_lift_comparison.global_newton import REVISION


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize('method',g.METHODS[:5])
def test_no_future_root_and_physical_reference_access(tmp_path,method):
    solver=g.GlobalEvolution(dict(n=4,method=method),tmp_path)
    channel=g.witness.channel
    physical_calls=[]
    def watched(density,time,*args):
        assert solver.certified
        assert density is solver.engine.physical or density is solver.engine.inverse_density
        physical_calls.append((density,time))
        return channel(density,time,*args)
    with patch.object(g.witness,'parity_power',side_effect=AssertionError('Preparation after t=0')), \
         patch.object(oracles,'root_mpo',side_effect=AssertionError('Scorer inside production')), \
         patch.object(g.witness,'channel',watched):
        x,diag,pred=solver.step(0.,.25)
    if solver.certified:
        assert [d for d,_ in physical_calls] == [solver.engine.physical,solver.engine.inverse_density]
        assert oracles.score(x,.25)['global_cp_error']<=diag['bound']+2e-12
        assert diag['bound']<=1.e-6 and diag['positivity_contribution']==2*diag['witness']
        if method=='Witness-only':
            assert pred is None and diag['fit_trials']==0
            assert 'predictor_seconds' not in solver.timings and 'fit_seconds' not in solver.timings
        else:
            assert 0<=diag['fit_updates']<=diag['fit_trials']<=3
    else:
        assert not physical_calls and solver.factor is None and 'bound' not in diag
    assert solver.timings['setup_seconds']>0 and diag['operations']['peak_intermediate_bond']>=2


@pytest.mark.parametrize('method', ['A+witness','Witness-only'])
def test_native_witness_trajectory(tmp_path,method):
    cfg=dict(n=4,method=method,chi=6,work=8)
    solver=g.GlobalEvolution(cfg,tmp_path/'wrapped')
    times=[.02,.1,.3]
    found=[]
    t0=0.
    for t1 in times:
        _,info,_=solver.step(t0,t1)
        found.append(info['bound'])
        t0=t1
    out=tmp_path/'legacy'
    assert g.witness.run(4,times,[(4,6,8),(6,8,12)],1.e-6,out,8,predict=method.startswith('A'),parity=True)
    checkfile=next(p for p in out.glob('*.csv') if not p.stem.endswith('trials'))
    with checkfile.open() as f:
        native=list(csv.DictReader(f))[1:]
    # Pauli-gauge re-Hermitization can add a few machine epsilons to _audit.
    np.testing.assert_allclose(found,[float(r['bound']) for r in native],rtol=2.e-7,atol=2.e-13)
    with np.load(next(out.glob('*.npz'))) as checkpoint:
        xmax=solver.root
        dense=xmax.__class__(4,max_bond=xmax.bond_dimension())
        dense.tensors=[checkpoint[f'x{j}'] for j in range(4)]
        assert g.tn.distance(xmax,dense)<2.e-12


def test_raw_native_a_and_audited_b_stages(tmp_path):
    a=g.GlobalEvolution(dict(n=4,method='A-raw'),tmp_path/'a')
    reference=g.witness.tdvp_prediction(a.root,0.,.1,a.engine,2,8)
    output,info,_=a.step(0.,.1)
    assert g.tn.distance(output,reference)<5.e-15
    assert len(info['ambient_stages'])==2 and all(e['fixed_passes']==2 for e in info['ambient_stages'])
    b=g.GlobalEvolution(dict(n=4,method='B-raw'),tmp_path/'b')
    solve=g.pcg.solve_velocity
    actual=[]
    def checked(stage,generator,**kwargs):
        actual.append(float(np.linalg.eigvalsh(g.witness.matrix(stage))[0]))
        assert 0<kwargs['lower_bound']<=actual[-1]+1e-13
        assert kwargs['mode']=='two' and kwargs['preconditioner']=='preparation'
        return solve(stage,generator,**kwargs)
    with patch.object(g.pcg,'solve_velocity',checked):
        b.step(0.,.1)
    stages=records(tmp_path/'b'/'stage_bounds.jsonl')
    data=records(tmp_path/'b'/'solves.jsonl')
    assert len(actual)==len(stages)==len(data)==2 and stages[0]['source']=='structural'
    assert all(e['galerkin_residual_abs']<1.e-10 and e['iterations']>=1 for e in data)


def test_uncertified_step_rolls_back_and_can_retry(tmp_path):
    solver=g.GlobalEvolution(dict(n=4,method='Witness-only',levels=[[1,1,1]],passes=1),tmp_path)
    previous=(solver.root,solver.factor,solver.bound,solver.engine.metric_time)
    with pytest.raises(g.EvolutionError,match='witness_uncertified'):
        solver.step(0.,1.)
    assert (solver.root,solver.factor,solver.bound,solver.engine.metric_time)==previous
    assert solver.t==0 and records(tmp_path/'global_failures.jsonl')[0]['last_accepted']==0.
    solver.cfg.update(levels=[[4,6,8],[6,8,12]],passes=8)
    assert solver.step(0.,1.)[1]['accepted']


def test_product_budget_and_stage_floor_failures(tmp_path):
    with pytest.raises(g.EvolutionError,match='memory_budget'):
        g.GlobalEvolution(dict(n=4,method='A-raw',max_elements=8),tmp_path/'cap')
    solver=g.GlobalEvolution(dict(n=4,method='B-raw'),tmp_path/'bad_floor')
    root=solver.root
    solver.floor=lambda _: -1.
    with pytest.raises(g.EvolutionError,match='rank_support_uncertain'):
        solver.step(0.,1.)
    assert solver.t==0 and solver.root is root
    assert not (tmp_path/'bad_floor'/'solves.jsonl').exists()


@pytest.mark.parametrize('method',g.METHODS[:5])
@pytest.mark.parametrize('p,b',[(.6,0.),(0.,.4)])
def test_degenerate_controls(tmp_path,method,p,b):
    solver=g.GlobalEvolution(dict(n=4,method=method,p=p,b0=b),tmp_path)
    state,info,_=solver.step(0.,.125)
    actual=oracles.score(state,.125,p,b)['global_cp_error']
    assert actual < (2.e-12 if b==0. else (1.e-6 if solver.certified else 2.e-4))
    if solver.certified:
        assert actual<=info['bound']+2.e-12


def test_pinned_newton(tmp_path):
    root=Path(__file__).resolve().parents[2]
    if subprocess.run(['git','cat-file','-e',REVISION+'^{commit}'],cwd=root,capture_output=True).returncode:
        pytest.skip('Optional Newton comparison requires the named Git object')
    solver=g.GlobalEvolution(dict(n=4,method='A+Newton',cert_tol=1.e-7),tmp_path)
    state,info,_=solver.step(0.,1.)
    assert oracles.score(state,1.)['global_cp_error']<=info['bound']+2.e-12
    assert info['fit_updates']>0 and info['correction_updates']>0 and info['bound']<=1.e-7
