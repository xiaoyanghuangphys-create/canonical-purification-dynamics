"""Independent dense physics and explicit-coordinate audits for Method B."""
import json
import unittest
from unittest.mock import patch

import numpy as np

from factor_tdvp.tangent_solver import matrix, matrix_to_mpo, mpo, tn
from galerkin_sylvester_tdvp import galerkin_solver as gs
from galerkin_sylvester_tdvp.tangent_frame import build_frame


def problem(n=3, seed=173, product=True):
    rng = np.random.default_rng(seed)
    def positive(d):
        a = rng.normal(size=(d,d)) + 1j*rng.normal(size=(d,d))
        x = np.eye(d) + a@a.conj().T/d
        return x/np.linalg.norm(x)
    if product:
        x = np.ones((1,1))
        for _ in range(n):
            x = np.kron(x, positive(2))
    else:
        x = positive(2**n)
    root = matrix_to_mpo(x, n)
    h = mpo.hamiltonian(n)
    rates = np.linspace(.16,.35,n)
    def generator(rho):
        coherent = tn.add(mpo.multiply(h,rho),mpo.multiply(rho,h),-1j,1j)
        return tn.add(coherent, mpo.onsite_dissipation(rho, rates))
    physical_h = matrix(h)
    rho = x@x
    g = -1j*(physical_h@rho-rho@physical_h)
    for j, rate in enumerate(rates):
        jump = np.kron(np.eye(2**j),np.kron(mpo.SM,np.eye(2**(n-j-1))))
        number = jump.conj().T@jump
        g += rate*(jump@rho@jump.conj().T-.5*(number@rho+rho@number))
    return root, generator, g, np.linalg.eigvalsh(x)[0]


class SolverTests(unittest.TestCase):
    def test_independent_dense_metric_and_rhs_full_and_restricted(self):
        for n, product in ((2,False),(3,True)):
            root, generator, g, lower = problem(n, product=product)
            for mode in ('one','two'):
                with self.subTest(n=n,mode=mode):
                    f = build_frame(root,mode)
                    oracle = gs.dense_oracle(root,g,mode=mode)
                    rng = np.random.default_rng(623)
                    v = rng.normal(size=f.dimension)+1j*rng.normal(size=f.dimension)
                    for cap in (None,4**(n//2)):
                        op = gs.GalerkinOperator(root,f,work_bond=cap)
                        np.testing.assert_allclose(op.matvec(v),oracle['M']@v,atol=2e-13)
                        rhs = f.encode(generator(op.product(root,root)))
                        np.testing.assert_allclose(rhs,oracle['b'],atol=4e-14)
                        audit = op.audit()
                        self.assertLess(audit['linearity_defect'], 5e-14)
                        self.assertLess(audit['self_adjoint_defect'], 5e-14)
                    spectrum = oracle['metric_spectrum']
                    self.assertGreaterEqual(spectrum[0],2*lower-1e-13)
                    self.assertLessEqual(spectrum[-1],2*oracle['root_spectrum'][-1]+1e-13)
                    for constrained in (True,False):
                        o = oracle if constrained else gs.dense_oracle(root,g,mode=mode,constrained=False)
                        velocity, info = gs.solve_velocity(root,generator,lower_bound=lower,mode=mode,
                            constrained=constrained, rtol=1e-11)
                        np.testing.assert_allclose(matrix(velocity),o['B_velocity'],atol=3e-11)
                        self.assertLess(info['galerkin_residual_rel'],1.1e-11)
                        r = matrix(root)@matrix(velocity)+matrix(velocity)@matrix(root)-g
                        self.assertAlmostEqual(info['physical_residual_abs'],np.linalg.norm(r),places=13)
                        proj = o['B'].conj().T@r.ravel()
                        self.assertAlmostEqual(info['projected_residual_abs'],np.linalg.norm(proj),places=13)
                        self.assertAlmostEqual(info['orthogonal_residual_abs'],
                            np.linalg.norm(r.ravel()-o['B']@proj),delta=2e-13)
                        self.assertLess(info['hermiticity_defect'],2e-12)
                        if constrained:
                            self.assertLess(info['normalization_velocity_defect'],2e-13)

    def test_distinct_weighted_objectives_and_constrained_update(self):
        root, onsite, g, lower = problem()
        x = matrix(root)
        pair = np.kron(np.kron(mpo.Z,mpo.Z),mpo.I)
        jump = matrix_to_mpo(pair,3)
        g += pair@(x@x)@pair - x@x
        def generator(rho):
            kicked = mpo.multiply(mpo.multiply(jump,rho),jump)
            return tn.add(onsite(rho),tn.add(kicked,rho,beta=-1.))
        o = gs.dense_oracle(root,g,constrained=False)
        a,b,c = o['A'],o['B_velocity'],o['C']
        for u,w in ((a,b),(b,c),(a,c)):
            self.assertGreater(np.linalg.norm(u-w),1e-4)
        x,f = matrix(root),o['exact_force']
        def losses(y):
            e = y-f
            s = x@e+e@x
            return (np.linalg.norm(e)**2,np.vdot(e,s).real,np.linalg.norm(s)**2)
        scores = np.array([losses(y) for y in (a,b,c)])
        np.testing.assert_array_equal(np.argmin(scores,axis=0),np.arange(3))
        free, fd = gs.solve_velocity(root,generator,constrained=False,lower_bound=lower)
        constrained, kd = gs.solve_velocity(root,generator,lower_bound=lower)
        self.assertGreater(fd['normalization_velocity_defect'],1e-5)
        retracted = matrix(free)-x*np.vdot(x,matrix(free)).real
        self.assertGreater(np.linalg.norm(retracted-matrix(constrained)),1e-5)
        self.assertLess(kd['normalization_velocity_defect'],1e-14)
        self.assertGreater(kd['physical_residual_rel'],.1)
        self.assertLess(kd['galerkin_residual_rel'],1.1e-9)
        self.assertGreater(kd['projected_residual_abs'],kd['galerkin_residual_abs']*100)

    def test_full_tangent_coincidence_and_coherent_split(self):
        root, generator, g, _ = problem(2,product=True)
        o = gs.dense_oracle(root,g,mode='two')
        for answer in (o['A'],o['B_velocity'],o['C']):
            np.testing.assert_allclose(answer,o['exact_force'],atol=2e-14)
        h,x = matrix(mpo.hamiltonian(2)),matrix(root)
        fcoh = -1j*(h@x-x@h)
        gcoh = -1j*(h@x@x-x@x@h)
        np.testing.assert_allclose(x@fcoh+fcoh@x,gcoh,atol=1e-15)
        split = gs.dense_oracle(root,g-gcoh,mode='two')['B_velocity'] + fcoh
        np.testing.assert_allclose(split,o['B_velocity'],atol=2e-14)
        # Restricted tangent splits must project the *source* of the omitted
        # coherent part or add a separately solved coherent Galerkin velocity.
        restricted = gs.dense_oracle(root,g,mode='one')['B_velocity']
        split_r = sum(gs.dense_oracle(root,source,mode='one')['B_velocity'] for source in (gcoh,g-gcoh))
        np.testing.assert_allclose(split_r,restricted,atol=2e-14)

    def test_product_preconditioner_warm_start_and_dense_export_guard(self):
        root, generator, _, lower = problem()
        with patch.object(tn.MPS,'dense',side_effect=AssertionError('global dense vector in production')):
            cold, old = gs.solve_velocity(root,generator,lower_bound=lower,preconditioner='none')
            pre, new = gs.solve_velocity(root,generator,lower_bound=lower,preconditioner='product')
            warm, reused = gs.solve_velocity(root,generator,lower_bound=lower,initial=cold)
        self.assertLess(tn.distance(cold,pre),2e-9)
        self.assertLess(tn.distance(warm,pre),2e-9)
        self.assertGreater(old['iterations'],0)
        self.assertGreater(new['preconditioner_setup_seconds'],0.)
        self.assertEqual(reused['iterations'],0)
        self.assertGreaterEqual(new['condition_estimate'],1.)

    def test_fixed_compression_is_linear_and_measured(self):
        root, _, _, _ = problem(2, product=False)
        x = matrix(root)
        base = np.kron(mpo.I+.2*mpo.X,mpo.I+.1*mpo.Z)
        x = base + 1e-6*x
        root = matrix_to_mpo(x,2)
        frame = build_frame(root,'two')
        op = gs.GalerkinOperator(root,frame,work_bond=1)
        exact = gs.GalerkinOperator(root,frame)
        rng=np.random.default_rng(121)
        v = rng.normal(size=frame.dimension)+1j*rng.normal(size=frame.dimension)
        v /= np.linalg.norm(v)
        delta = np.linalg.norm(op.matvec(v)-exact.matvec(v))
        self.assertGreater(op.discard,0.)
        self.assertLessEqual(delta,2*op.discard+1e-14)
        self.assertLess(op.audit()['linearity_defect'],1e-14)

    def test_resource_accuracy_rank_and_curvature_failures(self):
        root, generator, _, floor = problem(2, product=False)
        cases = [({'work_bond':1},'compression_overflow'),({'maxiter':0},'nonconvergence'),
                 ({'max_elements':1},'memory_budget'),({'lower_bound':None},'rank_support_uncertain')]
        for options, status in cases:
            with self.subTest(options=options), self.assertRaises(gs.GalerkinError) as caught:
                gs.solve_velocity(root,generator,**dict({'lower_bound':floor},**options))
            self.assertEqual(caught.exception.status,status)
        wrong = tn.scale(root,-1.)
        with self.assertRaises(gs.GalerkinError) as caught:
            gs.solve_velocity(wrong,generator,lower_bound=floor)
        self.assertEqual(caught.exception.status,'rank_support_uncertain')
        with patch.object(gs.GalerkinOperator,'matvec',side_effect=lambda v:-v):
            with self.assertRaises(gs.GalerkinError) as caught:
                gs.solve_velocity(root,generator,lower_bound=floor)
        self.assertEqual(caught.exception.status,'curvature_failure')
        root,generator,_,floor = problem(3)
        with self.assertRaises(gs.GalerkinError) as caught:
            gs.solve_velocity(root,generator,lower_bound=floor,physical_rtol=1e-6)
        self.assertEqual(caught.exception.status,'physical_residual_failure')
        self.assertLess(caught.exception.details['galerkin_residual_rel'],1e-8)
        self.assertGreater(caught.exception.details['physical_residual_rel'],.1)

    def test_strict_json_and_current_coefficient_symmetry(self):
        root, generator, _, lower = problem(2, product=False)
        x = matrix(root)
        skew = 1e-12j*np.diag([.3,.7,-.1,-.4])
        perturbed = matrix_to_mpo(x + skew,2)
        f = build_frame(perturbed,'two')
        op = gs.GalerkinOperator(perturbed,f)
        np.testing.assert_allclose(matrix(op.coefficient), x, atol=2e-15)
        self.assertLess(op.audit()['self_adjoint_defect'],1e-13)
        self.assertIsNone(gs._ratio(1.,0.))
        with self.assertRaises(gs.GalerkinError) as error:
            gs.solve_velocity(root,generator,lower_bound=np.nan)
        json.dumps(error.exception.details, allow_nan=False)
        def zero(state):
            return tn.scale(state,0), {'test_complex':np.array(3j)}
        _, info = gs.solve_velocity(root,zero,lower_bound=lower,initial=root)
        self.assertEqual(info['iterations'],0)
        self.assertIn('rank_discard_bound',info['frame_metadata'])
        json.dumps(info,allow_nan=False)

    def test_zero_rhs(self):
        root = mpo.product_root(2)
        v, info = gs.solve_velocity(root,lambda rho:tn.scale(rho,0),lower_bound=.1)
        self.assertEqual(info['iterations'],0)
        self.assertEqual(tn.norm(v),0.)


if __name__ == '__main__':
    unittest.main()
