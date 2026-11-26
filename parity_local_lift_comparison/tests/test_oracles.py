import numpy as np
import pytest
from parity_local_lift_comparison import oracles as o
from factor_tdvp import tangent_solver as ts


@pytest.mark.parametrize("n",[2,3,4,6,8])
def test_clifford_root_schmidt_cmi(n):
    p,b=.6,.4/np.e
    rho=o.dense_density(n,p,b)
    y=o.dense_power(rho,.5)
    state=o.root_mpo(n,p,b)
    np.testing.assert_allclose(ts.matrix(state),y,atol=3e-15,rtol=1e-13)
    assert state.bond_dimension()==2
    np.testing.assert_allclose(y@y,rho,atol=8e-15,rtol=1e-13)
    assert abs(np.linalg.norm(y)-1)<3e-14
    w=o.root_coefficients(p,b)[1]**2
    for width in sorted({1,n//2,n-1}):
        singular=np.linalg.svd(o.interleaved(y).reshape(4**width,-1),compute_uv=False)
        np.testing.assert_allclose(singular[:2],np.sqrt([1-w,w]),atol=4e-14,rtol=5e-14)
        assert np.linalg.norm(singular[2:])<4e-14
    if n>=3:
        one=[0]; buffer=list(range(1,n-1)); exterior=[n-1]
        assert abs(o.dense_cmi(rho,one,buffer,exterior)-o.cmi(p,b))<5e-14
        if n>3:
            assert abs(o.dense_cmi(rho,[2,0],[1],list(range(3,n)))-o.cmi(p,b))<5e-14
    scored=o.score(state,1.)
    assert scored['global_cp_error']<5e-14 and scored['density_residual_relative']<5e-14


def test_full_rank_controls_guard_and_stable_score():
    for p,b in [(0,0),(.6,0),(0,.4),(-.2,-.15)]:
        y=ts.matrix(o.root_mpo(3,p,b))
        np.testing.assert_allclose(y@y,o.dense_density(3,p,b),atol=2e-15)
        if p==0:
            assert o.cmi(p,b)==0 and o.unitary_floor(p,b,b/2)==0
            assert o.half_petz_metrics(p,b,b/2)['raw_error']<3e-15
        if b==0:
            assert o.unitary_floor(p,b,b)==0
            assert o.half_petz_metrics(p,b,b)['raw_error']<3e-15
    with pytest.raises(ValueError,match="requires"):
        o.root_mpo(2,1.,.01)
    with pytest.raises(ValueError,match="dimension"):
        o.dense_density(10,.6,.4)
    with pytest.raises(ValueError,match="exceeds"):
        o.dense_optimum(np.eye(2),np.eye(2),8)
    x=o.root_mpo(12,.6,.4)
    perturbed=o.tn.scale(x,1+1.e-10j)
    s=o.score(perturbed,0,max_product_elements=8)
    assert abs(s['global_cp_error']-1.e-10)<1.e-14
    assert s['density_score_status']=='allocation_guard'


def test_pauli_projections_and_density_yyadj():
    rng=np.random.default_rng(395)
    n=3
    y=rng.normal(size=(8,8))+1j*rng.normal(size=(8,8))
    y*=.8/np.linalg.norm(y)
    m=ts.matrix_to_mpo(y,n)
    rho=o.dense_density(n,.6,.4/np.e)
    score=o.score(m,1)
    i,p,q=o.dense_operators(n)
    for name,op in [('i',i),('p',p),('q',q),('qp',q@p)]:
        actual=score['root_'+name+'_re']+1j*score['root_'+name+'_im']
        assert abs(actual-np.vdot(op/np.sqrt(8),y))<2e-14
    for key,val in [('global_cp_error',np.linalg.norm(y-o.dense_power(rho,.5))),
                    ('hermiticity_defect',np.linalg.norm(y-y.conj().T)),
                    ('density_residual',np.linalg.norm(y@y.conj().T-rho)),
                    ('density_residual_relative',np.linalg.norm(y@y.conj().T-rho)/np.linalg.norm(rho)),
                    ('site1_physical_marginal_error',np.linalg.norm(o.marginal(y@y.conj().T-rho,[0])))]:
        assert abs(val-score[key])<2e-14


@pytest.mark.parametrize("width",[1,2,3])
def test_procrustes_attainment_and_fixed_support(width):
    p,b0,b1=.6,.4,.4/np.e
    n=3
    y0,y1=[o.dense_power(o.dense_density(n,p,b),.5) for b in (b0,b1)]
    limit=o.unitary_floor(p,b0,b1) if width<n else 0.
    measured,u=o.dense_optimum(y0,y1,width)
    assert abs(measured-limit)<1.e-13
    rows=o.interleaved(y0).reshape(4**width,-1)
    np.testing.assert_allclose(u.conj().T@u,np.eye(len(u)),atol=3e-15,rtol=1e-14)
    ext=(u@rows).conj().T@(u@rows)
    np.testing.assert_allclose(ext,rows.conj().T@rows,atol=3e-15,rtol=1e-14)
    explicit=o.explicit_optimal_action(y0,y1,width,p,b0,b1)
    assert abs(np.linalg.norm(explicit-o.interleaved(y1).reshape(4**width,-1))-limit)<1e-13
    if width<n:
        v=o.interleaved(y0).copy()
        for left,right in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
            temp=o.tn.to_matrix(v,n)
            v=o.explicit_optimal_action(temp,y1,width,p,b0*np.exp(-left),b0*np.exp(-right)).ravel()
        assert abs(np.linalg.norm(v-o.interleaved(y1))-limit)<1e-13
    assert limit<=np.sqrt(o.cmi(p,b0)-o.cmi(p,b1))+1.e-14


@pytest.mark.parametrize("n,width",[(2,1),(2,2),(4,1),(4,3),(4,4),(6,2)])
def test_dense_hp_full_chain_kraus_and_telescoping(n,width):
    p,b=.6,.4
    rho=o.dense_density(n,p,b)
    initial=o.dense_power(rho,.5)
    y,rho1=o.dense_hp_update(initial,rho,1.,width)
    target=o.dense_power(rho1,.5)
    current,r=initial,rho
    for h in (.04,.08,.24,.64):
        current,r=o.dense_hp_update(current,r,h,width)
    np.testing.assert_allclose(current,y,atol=3.e-14,rtol=1e-13)
    if width==n:
        np.testing.assert_allclose(y,target,atol=2e-14,rtol=1e-13)
    else:
        c=o.half_petz_metrics(p,b,b/np.e)
        assert abs(np.linalg.norm(y-target)-c['raw_error'])<2e-14
        assert abs(np.linalg.norm(y)-c['norm'])<2e-14
        assert abs(np.linalg.norm(y-y.conj().T)-c['hermiticity_defect'])<2e-14
        i,par,q=o.dense_operators(n)
        # The fourth branch is an anti-Hermitian Pauli product.
        assert abs(np.vdot(q@par/np.sqrt(2**n),y)-c['qp_coefficient'])<2e-14
        assert abs(c['qp_coefficient'])>8.e-4


def test_all_proper_physical_marginals_can_miss_global_root_error():
    n,p,b=4,.6,.4/np.e
    r_plus,r_minus=o.dense_density(n,p,b),o.dense_density(n,-p,b)
    for missing in range(n):
        keep=[j for j in range(n) if j!=missing]
        np.testing.assert_allclose(o.marginal(r_plus,keep),o.marginal(r_minus,keep),atol=0.)
    # At site j>1 the FULL physical update is identity on this path.
    for site in range(1,n):
        np.testing.assert_allclose(o.dense_dephase(r_plus,.5,site),r_plus,atol=2e-17)
    s=o.score(o.root_mpo(n,-p,b),1.)
    assert s['site1_physical_marginal_error']<1.e-13
    assert abs(s['global_cp_error']-2*o.root_coefficients(p,b)[1])<1.e-13


def test_no_dense_scoring_at_512(monkeypatch):
    x=o.root_mpo(512,.6,.4/np.e)
    monkeypatch.setattr(type(x),'dense',lambda self:pytest.fail("dense chain allocation"))
    s=o.score(x,1.)
    assert s['global_cp_error']<2.e-13
    assert s['density_score_status']=='completed' and s['density_residual_relative']<3e-13


def test_half_petz_telescopes_on_arbitrary_complex_amplitude():
    rng=np.random.default_rng(488)
    n=4
    rho=o.dense_density(n,.6,.4)
    y=rng.normal(size=rho.shape)+1j*rng.normal(size=rho.shape)
    y/=np.linalg.norm(y)
    for width in (1,3,4):
        direct,_=o.dense_hp_update(y,rho,1,width)
        first,r=o.dense_hp_update(y,rho,.23,width)
        composed,_=o.dense_hp_update(first,r,.77,width)
        np.testing.assert_allclose(direct,composed,atol=2e-14,rtol=1e-13)
