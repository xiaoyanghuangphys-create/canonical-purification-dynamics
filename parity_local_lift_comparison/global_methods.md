# Global TDVP and positive-witness comparison

## Production inputs and numerical objects

Each raw predictor starts with the numerical, Hermitian-rounded preparation root. A uses the original two-pass ambient MPO approximation to the Sylvester force, the normalized Euclidean two-site tangent projection and retracted explicit midpoint. The adapter measures the final ambient residual against the same work-bond right-hand side used by the native force. It records this contraction separately as `force_audit_seconds` inside predictor and production time. A exposes the intermediate first-pass residual as well. B uses the preparation-preconditioned, normalization-constrained Galerkin solve at **each actual numerical stage**. Its archive contains the original-equation residuals, iterations, support evidence and both retractions. A uses root/work caps 2/8 and B uses 2/16 unless the configuration overrides them. Running a full onsite generator action at the initial and midpoint root applies the local generator twice per site per interval. The parity oracle never enters either force. [Adapter](global_methods.py) · [Paired native predictor](../galerkin_sylvester_tdvp/witness_pair_worker.py) · [Prepared-PCG solver](../galerkin_sylvester_tdvp/witness_precond_solver.py)

B takes the initial physical root eigenvalue floor and subtracts the last certified global root error when such a certificate exists. Alternatively, it fits the **numerical stage** to the normalized identity, parity and first-site $`X`$ operators. The structural bound subtracts the full Frobenius norm of the component outside this span from the fitted minimum eigenvalue. The second stage may also use the certified first-stage floor less the measured midpoint displacement. A nonpositive bound halts the step before its solve. At large $`N`$ an $`O(2^{-N/2})`$ physical floor becomes hard to resolve using an absolute Frobenius remainder in double precision. A rejection at this test records a limit of the applied support proof. [Current-state floor](../galerkin_sylvester_tdvp/models.py) · [Stage regression](tests/test_global_methods.py)

The corrected pipelines start with a positive quarter-power factor $`W`$ and the normalized Hermitian compression $`X`$ of $`WW^\dagger`$. They apply the physical channel independently to the initial density and, for certification, to its inverse. Each of those MPO propagations acts on all $`N`$ sites once per new endpoint. A and B use whichever gives the lower bound between the carried tuple and their factor-fitted predictor. The fixed-anchor correction uses the stored-root density residual and accepts trial factors only on improvement of the independently computed bound. Witness-only omits the force and factor fit altogether. All three use the same level sequence of factor/root/work caps (4,6,8) then (6,8,12), up to eight passes per level by default. Rejection preserves the prior physical time, accepted factor/root and cached metric. The benchmark archives every accepted endpoint’s global score and bound. [Native witness code](../scalable_cp_witness/code/scalable_cp_witness.py) · [Endpoint and rollback regressions](tests/test_global_methods.py)

The audit reports $`2\eta`$ as its positivity contribution, with $`\eta`$ equal to the attained factor distance including the numerical symmetry allowance of the native audit. The remaining contribution uses the physical density residual and pays for residual rounding. The global root score is an independent Frobenius subtraction, with separate timing. Per-step work records count MPO compressions, the largest recorded discard, and the largest actual intermediate bond and core size in the observed additions, products and compression inputs. Native B/Newton records supplement these generic counters with their internal allocation audits. Solver clocks within the predictor or the witness are nested. [Worker and scoring](benchmark.py) · [Analytic reference and global diagnostics](oracles.py)

## Independent replay at 512 spins

The original native CLI completed all five positive times at tolerance $`10^{-6}`$, using the original settings and five-step grid. This replay uses the original predictor and adaptive root cap from the published recipe. The comparison adapter's common-cap grid is a separate experiment. The final stored factor and root both have bond two. [Raw checkpoints](results/global_native_512/n512_t1_s5_parity0.6_x0.4_jensen_tdvp.csv) · [Native settings](results/global_native_512/n512_t1_s5_parity0.6_x0.4_jensen_tdvp_settings.json)

| $`t`$ | Measured global CP error | Conditional bound |
|---:|---:|---:|
| 0.02 | $`1.05510\times10^{-7}`$ | $`2.80543\times10^{-7}`$ |
| 0.10 | $`9.74161\times10^{-8}`$ | $`2.58843\times10^{-7}`$ |
| 0.30 | $`2.26401\times10^{-7}`$ | $`5.78531\times10^{-7}`$ |
| 0.60 | $`1.30789\times10^{-7}`$ | $`3.25546\times10^{-7}`$ |
| 1.00 | $`2.31662\times10^{-7}`$ | $`5.80532\times10^{-7}`$ |

The independent Kraus/QR audit of the saved tensors produced $`5.80532521\times10^{-7}`$, differing from the original endpoint certificate by $`2.57\times10^{-13}`$. The preceding published run gave actual error $`2.31662226\times10^{-7}`$ at $`t=1`$. The fresh replay used 23.970 s of process wall time including startup, independent checkpoint scoring and file writes. The separate final QR process used 0.514 s. Those historical-driver costs belong outside the production-only frontier. [Verification](results/global_native_512/n512_t1_s5_parity0.6_x0.4_jensen_tdvp_verification.csv) · [Original trajectory](../scalable_cp_witness/n512_t1_s5_parity0.6_x0.4_jensen_tdvp.csv) · [Process provenance](results/global_native_512/provenance.json)

## Tangent-space Newton cost as the number of spins grows

The optional new fit solves the projected derivative of $`W^2`$ against the raw predictor. The physical correction solves the projected derivative of $`W^4`$ against the propagated density. For fixed local and intermediate bond dimensions, a fixed number of Krylov applications and a fixed number of accepted/backtracked nonlinear trials, the tensor sweeps have work proportional to $`N`$. Storing the train, the environments and the tangent vector likewise takes space proportional to $`N`$ under those conditions. For a uniform interior bond $`\chi`$ and local operator dimension four, the two-site tangent dimension is $`O(N\chi^2)`$. This conditional operation count cannot predict wall time from the stored root bond alone. [Verified source loader](global_newton.py) · [Original Newton implementation](https://github.com/xiaoyanghuangphys-create/canonical-purification-dynamics/blob/ba0d08c6b6cce07733204a7e2a9334dabb046134/tangent_newton_witness/operator.py)

The parity preparation's extremal density eigenvalue ratio is $`(1+r)/(1-r)`$, with $`r=\sqrt{p^2+b^2}`$. It is independent of $`N`$ at fixed $`p,b`$. At the exact positive fourth root, the unprojected derivative condition number is that ratio to power $`3/4`$. This removes one possible reason for the iteration count to change with $`N`$. The ambient residual audit, however, explicitly constructs all four terms $`W^{3-j}KW^j`$ and adds their tensor trains. Products use QR reductions at full geometric rank. A product can have bond up to $`\chi_W^4`$ for the fourth power and $`\chi_W^3\chi_K`$ for a derivative term. Direct sums in the audited residual can require further enlarged bonds. Very short chains cap such ranks at the small Hilbert dimensions near the boundaries. As the chain grows, more sites support the wide interior contractions. The measured wall times therefore need not behave linearly over the accessible sizes.

The fixed one-step experiment used $`p=0.6`$, $`b_0=0.4`$, $`T=1`$, endpoint tolerance $`10^{-7}`$, two-site Hermitian Newton with no inner preconditioner, a 75 s process limit and 768 MiB sampled RSS limit per worker. The paired initial predictors still use native A or prepared-PCG B. Source files from the earlier Newton commit are loaded into a private module namespace only after SHA-256 verification. The comparison requires those Git objects, takes no merges and uses no files from a sibling checkout. [Newton grid](newton_grid.json) · [Raw measurements](results/global_newton_01/measurements.jsonl)

### Measured Newton scaling

Each entry gives the median and [minimum–maximum] over completed workers out of three independent attempts per cell, in seconds. Correction includes construction and solution of every fourth-power Newton direction and its ambient residual audit. The witness column also includes the predictor fit, construction of the physical target, reconstruction, certification and logging. Production includes all of those costs plus preparation, predictor, guards and checkpointing. Independent scoring and process startup are separate. The runs used one BLAS thread per worker. Concurrent benchmark jobs can affect timings. [All workers and archived logs](results/global_newton_01/workers.zip) · [Build and environment](results/global_newton_01/manifest.json)

| Spins | Predictor + Newton | Completed | Fourth-power correction | Whole witness endpoint | Total production |
|---:|---|---:|---:|---:|---:|
| 4 | A | 3/3 | 0.020 [0.019–0.021] | 0.053 [0.050–0.055] | 0.101 [0.101–0.107] |
| 4 | B | 3/3 | 0.021 [0.019–0.022] | 0.053 [0.050–0.057] | 0.102 [0.095–0.106] |
| 8 | A | 3/3 | 0.704 [0.669–0.707] | 0.839 [0.796–0.841] | 0.906 [0.860–0.909] |
| 8 | B | 3/3 | 0.688 [0.659–0.709] | 0.826 [0.786–0.843] | 0.908 [0.856–0.917] |
| 16 | A | 3/3 | 14.0 [12.4–15.0] | 14.5 [12.9–15.6] | 14.6 [12.9–15.7] |
| 16 | B | 3/3 | 14.9 [12.8–15.7] | 15.3 [13.2–16.2] | 15.5 [13.3–16.4] |
| 32 | A | 3/3 | 47.0 [40.6–47.7] | 48.1 [41.7–48.7] | 48.4 [41.9–48.9] |
| 32 | B | 2/3 | 43.9 [42.4–45.4] | 45.1 [43.5–46.8] | 45.4 [43.7–47.1] |

At 64 spins all six workers exceeded their 768 MiB RSS guard before accepting $`t=1`$. Their recorded high-water marks at guard activation were 774–926 MiB. One B+Newton worker at 32 spins also reached its RSS guard (772.9 MiB) before accepting its endpoint. Its 32-spin timing cell summarizes the two completions. The 64-spin workers produced no accepted endpoint. At the four smaller sizes the completed workers’ final raw errors were about $`7.15\times10^{-12}`$ with A and $`2.53\times10^{-12}`$ with B, always within the recorded certificate. Both stored bonds were two.

Every completed worker took three fourth-power corrections with three Krylov iterations each. The tangent dimensions at 4, 8, 16 and 32 spins were 96, 288, 672 and 1440. Every physical solve recorded direction bond 10, fourth-power bond 16 and peak product bond 80. Including direct-sum residuals, the A pipeline reached intermediate bonds 32, 320, 349 and 349 respectively. Across the six 16-spin runs, median time summed over the three physical directions was 14.37 s in their ambient audits and 0.016 s inside their Krylov solves. Each larger chain cost more in these comparisons. The increase chiefly reflects the width and number of exact audit contractions, even with unchanged iteration counts. These four finite sizes do not establish an asymptotic exponent.

## Reproduction

Use an unused output directory for each independent invocation. For a clone missing the Newton revision, fetch the fork's named branch. The source loader verifies the pinned commit and both Python file hashes, regardless of later branch activity.

```sh
python3 -m parity_local_lift_comparison.global_replay --out /tmp/parity-native-replay
git fetch origin research/tangent-newton-witness
python3 -m parity_local_lift_comparison.benchmark --grid parity_local_lift_comparison/newton_grid.json --out /tmp/parity-newton --repeats 3
python3 -m pytest -q parity_local_lift_comparison/tests/test_global_methods.py
```
