# Conditioning at the current witness

The quartic derivative became easier to invert at late times on the product dephasing trajectory and harder on the loss trajectory. The two trends occurred while the initial density became a poor spectral reference in both families. All 112 linear solves below passed their original projected-residual checks. These measurements isolate a direction at a fixed checkpoint. Endpoint success and time to a common global error are measured separately in the [main report](report.md).

## Measured spectra and cost

Each sampled factor came from the shared initialization or the last accepted checkpoint of a correction-only run. Targets were independently evolved for another 0.5 time units. A separate dense calculation diagonalized the actual stored Hermitian factor, constructed its Hermitian tangent basis and obtained every eigenvalue of the projected derivative. The production solve received a conservative bound from the small-system factor eigenvalues. All solves used relative tolerance $10^{-8}$. Every table entry for time is the median of four repetitions at the same checkpoint, alternating the order of the two preconditioner settings. The elapsed time includes frame, operator, preconditioner, reconstruction and linear-residual checks. The dense spectral calculations and subsequent nonlinear scoring were timed outside each solve.

| Family | $N$ | Start time | Mode | Full $\kappa$ | Projected $\kappa$ | Jacobi $\kappa$ | CG iterations, plain / Jacobi | Milliseconds, plain / Jacobi |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Product | 4 | 0 | One | 2,345 | 15.7 | 17.4 | 2 / 3 | 5.58 / 7.48 |
| Product | 4 | 0 | Two | 2,345 | 81.9 | 86.8 | 5 / 8 | 5.73 / 8.12 |
| Product | 4 | 4 | One | 1.10 | 1.10 | 1.10 | 5 / 5 | 7.72 / 10.46 |
| Product | 4 | 4 | Two | 1.10 | 1.10 | 1.09 | 5 / 5 | 7.30 / 9.05 |
| Strong D | 2 | 4 | Two | 2.46 | 2.46 | 2.46 | 6 / 7 | 2.14 / 2.61 |
| Loss | 2 | 0 | One | 2.99 | 2.99 | 2.97 | 6 / 10 | 3.07 / 4.70 |
| Loss | 2 | 4 | One | 1,951 | 1,951 | 864 | 8 / 16 | 3.10 / 5.23 |
| Loss | 2 | 4 | Two | 1,951 | 1,951 | 1,951 | 8 / 11 | 2.25 / 2.87 |

Jacobi constructs every projected diagonal entry by contracting the fixed current-factor superoperators against the canonical environments and local complement columns. For positive $W$, the diagonal entries are positive. The code checks them before inverting the diagonal. Krylov convergence depends on the eigenvalues populated by the particular right-hand side. This matters in late loss where even a reduced worst-case condition number accompanied more iterations. Jacobi had the larger median inclusive time in all 14 sampled mode/checkpoint pairs. These millisecond measurements include interpreter overhead and do not establish performance at large bond.

In the four-spin product family, the generalized eigenvalue range of the target relative to the initial density changed from $[0.448,136]$ at $t=0.5$ to $[0.0868,2505]$ at $t=4.5$. Meanwhile, the spectrum of the actual witness at $t=4$ was nearly flat. At late loss, the target/initial range at $t=4.5$ was $[1.17\times10^{-4},2.62]$, and small current populations stiffened the actual Newton equation.

The two-spin product example also illustrates the need for nonlinear safeguarding. Starting from $t=0$ with target time 0.5, the unrestricted two-site direction left an ambient linear residual of $1.44\times10^{-14}$. A full normalized step nevertheless raised the physical fourth-power residual from 0.33 to 0.47. The one-site solve left an ambient linear residual of 0.039 and attained 0.17 after normalization. This trial used the full direction with no line search or bond truncation. A production endpoint independently evaluates the attained certificate after retraction.

## Current-factor assumptions and operator checks

With $W>0$, the eigenvalues of the square derivative are $w_a+w_b$. For the fourth power they are $(w_a+w_b)(w_a^2+w_b^2)$. Compression of the derivative to an orthonormal subspace preserves positivity, and its Rayleigh quotients lie between the full spectral extrema. In particular $\kappa(\mathcal D_{4,W})=(w_{\max}/w_{\min})^3$. Galerkin minimizes the derivative-induced energy norm of the direction error when the derivative is positive definite. A projected least-squares solve minimizes a different quantity, the ambient linearized residual with optional damping.

The implementation realifies the full Pauli coefficient train of the current factor, then builds the independent one-site or overlapping two-site coordinates at the measured active ranks. Real coordinate columns decode to Hermitian matrices even when the original virtual gauge is complex. Rank removal affects the variation space. The coefficient powers and residual use the original factor. There is no intermediate tangent projection. For a Hermitian test column and direction, the coefficients of $LKR$ and its adjoint are complex conjugates. The matvec can therefore contract one representative of each pair and take twice its real part. Setup reuses the fixed powers and superoperator cores for every application. Environments involving the decoded trial direction are swept for each matvec.

A Pauli-string triangle bound of the represented factor charges Frobenius rank discard and Hermitianization. An automatic bound at cancellation scale is treated as inconclusive. External bounds must come from the actual numerical factor. A finite nonpositive lower bound selects MINRES. A contradictory trace, nonfinite lower bound or measured nonpositive CG curvature rejects the solve. Positive trace alone gives no positivity guarantee. Tests include a positive-trace mixed-sign witness, a deliberately false caller guarantee that triggers negative curvature, non-Hermitian input/RHS, and forcing in a physical kernel. An actual external positivity lower bound is a caller obligation. A false guarantee can evade sampled curvature tests. The original projected residual, the ambient residual and its component outside the variation space are recomputed using MPO products. These are floating-point audits.

At full tangent rank in a complex noncommuting two-qubit example, the observed errors $\|W_k-W_*\|_F$ were as follows. Here the true solution is used to prepare a controlled local initial condition and to score it. The direction takes the power target only. These iterations used neither retraction nor normalization.

| Power | Initial | First iteration | Second iteration | Third iteration |
|---:|---:|---:|---:|---:|
| 2 | 0.1052 | 0.004537 | $1.246\times10^{-5}$ | $1.056\times10^{-10}$ |
| 4 | 0.1052 | 0.01348 | $3.245\times10^{-4}$ | $2.226\times10^{-7}$ |

The successive ratios $e_{k+1}/e_k^2$ were $0.410,0.605,0.680$ and $1.22,1.78,2.11$. The local full-space behavior does not establish quadratic convergence under tangent restriction, line search or retraction.

## Reproduction

Run from the repository root with an unused output directory. The default input checkpoints are preserved under `tangent_newton_witness/results/stress_late_01`. The spectral cohort records their SHA-256 hashes.

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m tangent_newton_witness.conditioning --out /tmp/witness-conditioning-replay
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m pytest -q tangent_newton_witness/tests/test_operator.py
```

The operator suite passed all 12 tests. The dense oracles compare derivatives to finite differences, eigenspace inverses to direct solves, and the real tangent projection to both original projectors. Coverage includes a nonuniform four-site rank profile, N=1 boundaries, Hermiticity, orthonormality, coordinate adjointness, radial directions, positive semidefinite/singular and mixed-sign guards, direct comparison of the projected action/diagonal and original ambient residual, and a no-dense 16-site MPO solve. Direct dense comparisons use relative tolerance $10^{-12}$ (finite differences use $10^{-8}$) and explicitly stated absolute tolerances from $10^{-14}$ to $2\times10^{-10}$ depending on the quantity, with a looser residual test for intentionally ill-conditioned eigenvalues. Full settings, timings and tolerances are in [the tests](tests/test_operator.py) and the [conditioning results](results/conditioning_01/rows.jsonl).
