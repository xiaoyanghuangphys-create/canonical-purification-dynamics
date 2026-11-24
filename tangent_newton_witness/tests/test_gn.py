import numpy as np
import pytest
from factor_tdvp.tangent_solver import matrix, matrix_to_mpo
from galerkin_sylvester_tdvp.galerkin_solver import GalerkinError
from tangent_newton_witness.gn import jacobian, jacobian_adjoint, solve_direction, pack


@pytest.mark.parametrize("power", [2, 4])
def test_real_adjoint_finite_difference_and_physical_gauge(power):
    rng = np.random.default_rng(921)
    w, k, r, omega = (rng.normal(size=(4,4)) + 1j*rng.normal(size=(4,4)) for _ in range(4))
    omega -= omega.conj().T.copy()

    def value(a):
        x = a @ a.conj().T
        return x if power == 2 else x @ x

    deriv = jacobian(w, k, power)
    fd = (value(w + 1.e-6*k) - value(w - 1.e-6*k)) / 2.e-6
    np.testing.assert_allclose(deriv, fd, atol=2.e-7, rtol=2.e-8)
    np.testing.assert_allclose(np.vdot(r, deriv).real, np.vdot(jacobian_adjoint(w, r, power), k).real)
    assert np.linalg.norm(jacobian(w, w@omega, power)) < 1.e-11


@pytest.mark.parametrize("power", [2, 4])
def test_damped_lsqr_matches_explicit_real_design(power):
    rng = np.random.default_rng(602)
    w = np.diag([1., .8, .5, .35]).astype(complex)
    w[0,1] = .14j
    x = w @ w.conj().T
    r = rng.normal(size=(4,4)) + 1j*rng.normal(size=(4,4))
    r = (r + r.conj().T) / 50
    target = r + (x if power == 2 else x@x)
    direction, info = solve_direction(matrix_to_mpo(w,2), matrix_to_mpo(target,2),
                                      power, damping=.03, rtol=1.e-9)
    unit = np.eye(16).reshape(16,4,4)
    design = np.stack([pack(jacobian(w,k,power)) for k in [*unit, *(1j*unit)]],axis=1)
    ref = np.linalg.lstsq(np.vstack([design, .03*np.eye(32)]), np.r_[pack(r),np.zeros(32)],rcond=None)[0]
    np.testing.assert_allclose(pack(matrix(direction)), ref, atol=1.e-9)
    assert info["normal_residual"] <= info["normal_threshold"]


def test_unresolved_kernel_forcing_rejected():
    w = matrix_to_mpo(np.diag([1.,0.]),1)
    t = matrix_to_mpo(np.diag([1.,.01]),1)
    with pytest.raises(GalerkinError, match="Unresolved forcing"):
        solve_direction(w,t,4)


@pytest.mark.parametrize('weight', [3e-15, 3e-14])
def test_qr_adjoint_retains_forcing_above_and_below_a_relative_svd_cutoff(weight):
    from tangent_newton_witness.gn import _untruncated_mpo
    from galerkin_sylvester_tdvp.tangent_frame import build_frame

    # Orthogonal rank-one Schmidt sectors. The second stays resolvable in
    # double precision because its physical matrix entry is separate.
    large = np.diag([1.,0.,0.,0.]).astype(complex)
    small = np.diag([0.,0.,0.,weight]).astype(complex)
    w = np.eye(4)
    frame = build_frame(matrix_to_mpo(w,2), 'two')
    columns = np.column_stack([matrix(frame.decode(z)).ravel() for z in np.eye(frame.dimension)])

    def adjoint(r):
        dense = jacobian_adjoint(w,r,4)
        return frame.encode(_untruncated_mpo(dense,2))

    together = adjoint(large+small)
    dense_reference = columns.conj().T@jacobian_adjoint(w,large+small,4).ravel()
    np.testing.assert_allclose(together, dense_reference, rtol=1e-14, atol=1e-30)
    np.testing.assert_allclose(together-adjoint(large), adjoint(small), rtol=1e-14, atol=1e-30)
    assert np.linalg.norm(adjoint(small)) > 3*weight
