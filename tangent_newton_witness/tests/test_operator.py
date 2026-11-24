"""Independent dense checks; production routines never call these oracles."""
import numpy as np
import pytest

from cp_tdvp import tdvp_core as legacy
from factor_tdvp.tangent_solver import matrix, matrix_to_mpo, mpo, tn
from galerkin_sylvester_tdvp.galerkin_solver import GalerkinError
from tangent_newton_witness.operator import HermitianFrame, NewtonOperator, factor_power, solve_direction


def herm(rng, d):
    a = rng.normal(size=(d, d))+1j*rng.normal(size=(d, d))
    return (a+a.conj().T)/2


def applied(w, k, power):
    powers = [np.linalg.matrix_power(w, i) for i in range(power)]
    return sum(powers[i] @ k @ powers[-i-1] for i in range(power))


@pytest.mark.parametrize('m', [2, 4])
def test_noncommuting_derivative_inverse_and_local_newton(m):
    rng = np.random.default_rng(482)
    w = np.eye(4)+.15*herm(rng, 4)
    h = herm(rng, 4)
    assert np.linalg.norm(w@h-h@w) > .01
    deriv = applied(w, h, m)
    eps = 1e-5
    diff = (np.linalg.matrix_power(w+eps*h, m)-np.linalg.matrix_power(w-eps*h, m))/(2*eps)
    np.testing.assert_allclose(deriv, diff, rtol=1e-8, atol=1e-9)
    if m == 4:
        sw = w@h+h@w
        np.testing.assert_allclose(deriv, w@w@sw+sw@w@w, atol=1e-13)
    eig, u = np.linalg.eigh(w)
    spec = eig[:, None]+eig
    if m == 4:
        spec *= eig[:, None]**2+eig**2
    r = herm(rng, 4)
    k = u@((u.conj().T@r@u)/spec)@u.conj().T
    matrix4 = np.column_stack([applied(w, e.reshape(4, 4), m).ravel() for e in np.eye(16)])
    np.testing.assert_allclose(np.linalg.solve(matrix4, r.ravel()).reshape(4, 4), k, atol=1e-13)
    assert np.isclose(np.linalg.cond(matrix4), (eig[-1]/eig[0])**(m-1))
    target = np.linalg.matrix_power(w, m)
    current = w+.025*h
    errors = []
    for _ in range(3):
        errors.append(np.linalg.norm(current-w))
        # Dense test only: eigvalsh supplies a valid current-W floor.
        delta, _ = solve_direction(matrix_to_mpo(current, 2), matrix_to_mpo(target, 2),
                                   m, lower_bound=np.linalg.eigvalsh(current)[0]*.99, rtol=1e-11)
        current += matrix(delta)
    assert errors[1] < 4*errors[0]**2
    assert errors[2] < 4*errors[1]**2


@pytest.mark.parametrize('mode', ['one', 'two'])
@pytest.mark.parametrize('m', [2, 4])
def test_independent_tangent_metric_and_original_projector(mode, m):
    rng = np.random.default_rng(403)
    # Nonuniform physical Schmidt ranks 1,2,2,1,1 and complex noncommuting directions.
    z = np.diag([.5, -.5])
    y = np.array([[0, -1j], [1j, 0]])
    w = np.eye(16)+.2*np.kron(np.kron(np.kron(y, z), y), np.eye(2))
    wt = matrix_to_mpo(w, 4)
    op = NewtonOperator(wt, m, mode=mode)
    f = op.frame
    ranks = (1, 2, 2, 1, 1)
    assert f.ranks == ranks
    d1 = sum(4*a*b for a,b in zip(ranks, ranks[1:]))-sum(r*r for r in ranks[1:-1])
    d2 = sum(16*a*c for a,c in zip(ranks,ranks[2:]))-sum(4*a*b for a,b in zip(ranks[1:-2],ranks[2:-1]))
    assert f.dimension == (d1 if mode == 'one' else d2)
    eye = np.eye(f.dimension)
    B = np.column_stack([matrix(f.decode(e)).ravel() for e in eye])
    np.testing.assert_allclose(B.conj().T@B, eye, atol=3e-13)
    np.testing.assert_allclose(B, B.reshape(16, 16, -1).swapaxes(0, 1).conj().reshape(256,-1), atol=1e-14)
    h = herm(rng,16)
    ht = matrix_to_mpo(h,4)
    np.testing.assert_allclose(f.encode(ht), (B.conj().T@h.ravel()).real, atol=1e-13)
    original = legacy.tangent_project if mode == 'one' else legacy.tangent_project_two
    np.testing.assert_allclose(matrix(f.decode(f.encode(ht))), matrix(original(wt, ht)), atol=2e-13)
    np.testing.assert_allclose(matrix(f.decode(f.encode(wt))), w, atol=1e-13)
    DB = np.column_stack([applied(w, col.reshape(16,16),m).ravel() for col in B.T])
    A = (B.conj().T@DB).real
    np.testing.assert_allclose(A, A.T, atol=2e-13)
    np.testing.assert_allclose(op.diagonal(),np.diag(A),atol=2e-13)
    assert np.linalg.eigvalsh(A)[0] > .9
    v = rng.normal(size=f.dimension)
    np.testing.assert_allclose(op.matvec(v), A@v, atol=2e-12)
    target = matrix_to_mpo(h+np.linalg.matrix_power(w,m),4)
    k, info = solve_direction(wt, target, m, mode=mode, preconditioner='jacobi', rtol=1e-11)
    np.testing.assert_allclose(matrix(k).ravel(), B@np.linalg.solve(A,f.encode(ht)), atol=2e-10)
    physical = applied(w,matrix(k),m)-h
    np.testing.assert_allclose(info['full_linear_residual'],np.linalg.norm(physical), atol=1e-12)
    np.testing.assert_allclose(info['outside_tangent_residual'],
                               np.linalg.norm(physical.ravel()-B@(B.conj().T@physical.ravel())),atol=1e-12)
    assert info['solver'] == 'cg' and info['projected_residual'] < 1e-9
    with pytest.raises(ValueError):
        f.decode(v+0j)


def test_bond_opening_rank_trim_and_factor_normalization():
    vac = tn.vacuum(3)
    padded = tn.add(vac, vac, alpha=.3, beta=.7)
    target = np.zeros((8,8))
    target[6,6] = 1
    t = matrix_to_mpo(target,3)
    one,two = (HermitianFrame(padded, mode) for mode in ('one','two'))
    assert one.ranks == (1,1,1,1)
    assert max(one.metadata['input_ranks']) > max(one.ranks)
    assert one.metadata['rank_discard_bound'] < 1e-13
    assert np.linalg.norm(one.encode(t)) < 1e-13
    np.testing.assert_allclose(matrix(two.decode(two.encode(t))),target,atol=1e-13)
    grown = HermitianFrame(tn.add(vac,t,beta=.125),'one')
    assert grown.ranks == (1,2,1,1)
    h = np.diag([2., .5])
    t = matrix_to_mpo(h,1)
    quart = factor_power(t,4)
    normalized = tn.scale(t,mpo.trace(quart).real**(-.25))
    assert abs(mpo.trace(factor_power(normalized,4))-1.) < 1e-13
    np.testing.assert_allclose(matrix(HermitianFrame(t).decode(HermitianFrame(t).encode(t))),h,atol=1e-13)


@pytest.mark.parametrize('m', [2,4])
def test_indefinite_singular_tiny_and_fixed_point(m):
    h = np.diag([-1.,.5])
    factor = matrix_to_mpo(h,1)
    op = NewtonOperator(factor,m)
    A = np.column_stack([op.matvec(x) for x in np.eye(4)])
    assert min(np.linalg.eigvalsh(A)) < 0 < max(np.linalg.eigvalsh(A))
    r = np.diag([-.2,.1])
    t = matrix_to_mpo(np.linalg.matrix_power(h,m)+r,1)
    direction, info = solve_direction(factor,t,m)
    assert info['solver'] == 'minres'
    np.testing.assert_allclose(applied(h,matrix(direction),m),r,atol=1e-13)
    singular = np.diag([1.,0.])
    with pytest.raises(GalerkinError,match='null forcing'):
        solve_direction(matrix_to_mpo(singular,1),matrix_to_mpo(np.eye(2),1),m)
    # Small positive diagonal, well above absolute floating-point range.
    w = np.diag([1.,1e-4])
    t = matrix_to_mpo(np.linalg.matrix_power(w,m)+np.diag([0.,1e-6]),1)
    k, info = solve_direction(matrix_to_mpo(w,1),t,m,lower_bound=0.99e-4,rtol=1e-3)
    assert info['solver'] == 'cg'
    np.testing.assert_allclose(applied(w,matrix(k),m),np.diag([0.,1e-6]),atol=1.1e-9)
    for h in [np.eye(2),-np.eye(2)]:
        k, info = solve_direction(matrix_to_mpo(h,1),matrix_to_mpo(np.eye(2),1),m)
        assert info['iterations'] == 0 and tn.norm(k) == 0.


def test_original_unrounded_equation_and_no_intermediate_projection():
    w = np.diag([1.1,.9])
    full = np.kron(np.kron(w,w), w)
    wt = matrix_to_mpo(full,3)
    op=NewtonOperator(wt,4,'one')
    f=op.frame
    B=np.column_stack([matrix(f.decode(e)).ravel() for e in np.eye(f.dimension)])
    sw=np.column_stack([(full@u+u@full).ravel() for u in B.T.reshape(-1,8,8)])
    x=full@full
    sx=np.column_stack([(x@u+u@x).ravel() for u in B.T.reshape(-1,8,8)])
    twice_projected=(B.conj().T@sx)@(B.conj().T@sw)
    exact=np.column_stack([op.matvec(e) for e in np.eye(f.dimension)])
    assert np.linalg.norm(exact-twice_projected) > 1e-4


def test_guard_limits_and_no_dense_16(monkeypatch):
    w = mpo.product_root(16,polarization=.01)
    target = tn.scale(factor_power(w,4),1.01)
    for module, name in [(tn.MPS,'dense'),(tn,'to_matrix'),(tn,'from_dense')]:
        monkeypatch.setattr(module,name,lambda *a: pytest.fail('production called dense'))
    for m, t in [(2,tn.scale(factor_power(w,2),1.01)),(4,target)]:
        k,info=solve_direction(w,t,m,rtol=1e-8)
        assert info['full_linear_residual'] < 1e-8*info['rhs_norm']
        assert tn.norm(k) > 0
    with pytest.raises(GalerkinError,match='element budget'):
        solve_direction(w,target,4,max_elements=4)
    with pytest.raises(GalerkinError,match='Iteration limit'):
        solve_direction(w,target,4,maxiter=1,rtol=1e-14)
    with pytest.raises(GalerkinError,match='non-Hermitian'):
        NewtonOperator(tn.scale(w,1j),2)
