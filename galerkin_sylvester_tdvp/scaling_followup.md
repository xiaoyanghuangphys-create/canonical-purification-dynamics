# Correlated-chain support audit and parity time refinement

The follow-up keeps the production Galerkin solver, root representation and midpoint integrator fixed. For the circuit-prepared mixed chain (D), a new callback can certify the positive branch of each numerical stage at finite size. The physical reference enters only at endpoint scoring. For the parity family (E), the original structural positivity callback is reused on a finer uniform time grid. The study uses one worker per observation on one machine. [Implementation](scaling_followup.py) · [Prespecified grid](results/scaling_followup_20261119/grid.json) · [Machine, limits and source hashes](results/scaling_followup_20261119/environment.json)

## Stage audit

Let $A$ be the positive initial circuit root, $\ell_0=(0.44)^{N/2}$ its analytical spectral floor, $X$ the numerical stage and $H=(X+X^\dagger)/2$. If $\Delta=X-A$, then $\lambda_{\min}(H)\geq\ell_0-\lVert\Delta\rVert_{\mathrm{op}}$. Compress $\Delta$ once and add its Frobenius discard allowance to the upper bound on the rounded operator norm. First try its Frobenius norm. If that cannot certify the stage, group up to four consecutive MPO cores. For each group $g$ with open virtual pair $(a,b)$, form its physical $2^w\times 2^w$ operator and set $K_g(a,b)$ equal to that block's largest singular value. Summing the norms of all virtual paths gives $\lVert\Delta_{\mathrm{rounded}}\rVert_{\mathrm{op}}\leq(K_1\cdots K_m)_{11}$ by submultiplicativity and the triangle inequality. The physical block size is at most 16.

If the direct bound fails, the exact initial circuit supplies $W=A^{-1/2}=\rho_0^{-1/4}$. Apply it on both sides of the perturbation. A bound $r\geq\lVert W\Delta W\rVert_{\mathrm{op}}$ with $r<1$ gives $H\succeq(1-r)A$ and hence a positive floor $(1-r)\ell_0$. The three compression discard allowances for $\Delta$, $W\Delta$ and $W\Delta W$ enter the upper bound with coefficients $1/\ell_0$, $1/\sqrt{\ell_0}$ and 1. A value $r\geq1$ provides no certificate in this test. All stage bounds are conditional on the stated initial spectrum and ordinary floating-point contractions.

## Independent endpoint bound

Uniform dephasing is a unital positive channel, so $\rho(t)\succeq\ell_0^2 I$. Given a certified numerical floor $H\succeq\ell_H I$, let $S=\sqrt{\rho(t)}$. The identity $H(H-S)+(H-S)S=H^2-\rho(t)$ and the spectrum of this positive Sylvester map give

$$
\lVert X-S\rVert_F \leq \lVert X-H\rVert_F
  + \frac{\lVert H^2-\rho(t)\rVert_F}{\ell_H+\ell_0}.
$$

The physical density on the right is propagated independently. D8 also has a measured error against a dense global diagonalization. The identity and the two anticommuting Pauli strings give an exact E reference root at every size. The D bounds and parity errors retain separate columns. Three additional tests compare dense stage spectra and endpoint distances against the reported inequalities, including skewed candidates, deliberately excessive compression and an unresolved branch. [Tests](tests/test_scaling_followup.py)

## Observed endpoints and failures

D8 and D16 reached $t=0.1$ on two uniform steps with root cap 4 and work cap 64. E8–64 reached $t=1$ on eight steps with root cap 2 and work cap 16. Wall time includes starting the process, production and scoring. RSS is each worker's measured peak. Empty scores indicate that the final endpoint or a separate reference was unavailable. [All observations](results/scaling_followup_20261119/measurements.csv)

| Family | Spins | Accepted | Outcome | Measured CP error | Conditional global bound | Wall seconds | Peak MiB |
|---|---:|---:|---|---:|---:|---:|---:|
| D | 8 | 2 | Completed | 1.124e-5 | 2.204e-5 | 2.24 | 71.8 |
| D | 16 | 2 | Completed | — | 8.714e-5 | 10.24 | 247.8 |
| D | 32 | 0 | Endpoint positivity unresolved | — | — | 13.40 | 436.0 |
| D | 64 | 0 | Midpoint positivity unresolved | — | — | 15.19 | 460.2 |
| E | 8 | 8 | Completed | 3.135e-4 | — | 0.89 | 61.9 |
| E | 16 | 8 | Completed | 3.135e-4 | — | 1.87 | 57.3 |
| E | 32 | 8 | Completed | 3.135e-4 | — | 3.71 | 68.9 |
| E | 64 | 8 | Completed | 3.135e-4 | — | 8.34 | 94.4 |

The D8 run recovered after increasing the work cap from 16 to 64. Its 1.12e-5 measured error still exceeds the 1e-5 accuracy target. For D16 the grouped stage bound proved sufficient throughout. The endpoint density discrepancy was 1.419e-7 in Frobenius norm. Paying for positivity with distance to the initial root alone gives the older conditional bound 0.02463, versus 8.714e-5 using a positive numerical floor. The latter certifies the 1e-3 target and leaves 1e-5 unresolved. [D16 stage audits](results/scaling_followup_20261119/D16/floor_audit.jsonl) · [D16 scores](results/scaling_followup_20261119/D16/measurement.json)

D32's first internal midpoint passed with relative congruence bound 0.879. At the proposed endpoint the same upper bound was 1.699, beyond the positivity threshold of 1. D64's first midpoint had relative upper bound 31.10 including a compression allowance of 2.61. These outcomes establish that the available inequalities could not certify those numerical stages. The sign of the actual smallest eigenvalue remains unknown in both cases. [D32 stages](results/scaling_followup_20261119/D32/floor_audit.jsonl) · [D64 stages](results/scaling_followup_20261119/D64/floor_audit.jsonl)

For E the two-step benchmark measured 6.918e-3 across these sizes. Eight steps reduced the global error by a factor of 22.1 and reached the 1e-3 target at all four sizes. At 64 spins the eight-step density error has fallen to 1.31e-13 in Frobenius norm while the CP error is 3.135e-4, demonstrating why the two metrics must be recorded separately. Each E run required 32 CG iterations in total. This size independence is evidence for the special parity trajectory and chosen grid. [Two-step study](results/scaling_20261119/measurements.csv) · [Eight-step E64](results/scaling_followup_20261119/E64/measurement.json)

## Reproduction

Run from the repository root with a fresh output directory. Each worker has a 35-second wall deadline, 768 MiB peak-RSS watchdog (40 ms sampling), and 80 CG iterations per force. The worker saves bound and stage events as it progresses, and the parent retains checkpoint metadata on timeout. All eight exploratory size observations have one repeat.

```sh
python -m pytest -q galerkin_sylvester_tdvp/tests/test_scaling_followup.py
python -m galerkin_sylvester_tdvp.scaling_followup --out run_outputs/new_galerkin_scaling_followup
```
