import numpy as np
import pytest
from tangent_newton_witness.support import finite_start, auto, matrix_to_mpo, matrix, sw
from galerkin_sylvester_tdvp.galerkin_solver import GalerkinError


@pytest.mark.parametrize("theta,jump,expected", [
    (0, np.diag([1.,-1.]), "smooth_within_budget"),
    (0, sw.mpo.X, "first_order_opening"),
    (1, sw.mpo.SM, "delayed_opening"),
])
def test_support_gate(theta,jump,expected):
    p=matrix_to_mpo(np.diag([1.,0.]),1)
    state=(p,p,0.)
    child,h,info=finite_start(state,auto.Model(theta*sw.mpo.X,[jump]),.02,1.e-6,
                              root_bond=1,factor_bond=1)
    assert info['mode']==expected
    if h:
        assert info['post']['rank']==2
        assert np.linalg.eigvalsh(matrix(child[0]))[0] > 0
        assert abs(np.trace(np.linalg.matrix_power(matrix(child[0]),4))-1) < 1.e-13
    else:
        assert child is state


def test_newborn_and_stiff_scales_and_unsupported_size():
    p=matrix_to_mpo(np.diag([1.,0.]),1)
    model=auto.Model(0*sw.mpo.X,[sw.mpo.X])
    for h in (.001,.0001):
        (w,x,_),used,_=finite_start((p,p,0.),model,h,.1,root_bond=1,factor_bond=1)
        np.testing.assert_allclose(np.linalg.eigvalsh(matrix(w))[0],used**.25, rtol=.001)
        np.testing.assert_allclose(np.linalg.eigvalsh(matrix(x))[0],np.sqrt(used),rtol=.001)
    tiny=matrix_to_mpo(np.diag(np.sqrt([1.-1.e-8,1.e-8])),1)
    _,_,audit=finite_start((p,tiny,0.),model,.001,.01,root_bond=1,factor_bond=1)
    assert audit['mode']=='near_boundary'
    big=sw.prepared_power(8,.5)
    with pytest.raises(GalerkinError,match="audited"):
        finite_start((big,big,0.),model,.01,.01,root_bond=2,factor_bond=2)
