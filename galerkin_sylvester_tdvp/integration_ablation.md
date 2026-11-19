# Integration, normalization and rank-opening controls

The full-rank control exposes an accuracy failure at a converged Galerkin solution. At three spins, one-site bond two and $t=0.08$, the largest constrained Galerkin residual over eight force solves was $6.92\times10^{-11}$. The largest full Sylvester residual was 0.455, and the measured global canonical-root error was 0.0673. The input representation error was $6.33\times10^{-16}$. [Measurements](results/evolution_confirm_20261119/measurements.csv) · [Stage and endpoint records](results/evolution_confirm_20261119/00/integration.json)

The initial root is the Frobenius-normalized sum of two positive products. The first has local factors $(I+0.78P_j)/2$ with axes $(X,Y,Z)$. The second has prefactor 0.46 and local factors $(I+0.63P_j)/2$ with axes $(Z,X,Y)$. The physical dynamics use the repository’s Ising fields and site-dependent amplitude-damping rates. Independent endpoint scoring exponentiates the dense Lindblad generator on the initial density and diagonalizes the resulting Hermitian density. Production force calls use the numerical stage's MPO. The other control starts one qubit in its excited state under unit-rate loss. [Model and runner](evolution_ablation.py)

## Constraint and preconditioner

All five rows use four steps, state bond two, working bond 16, relative tolerance $10^{-10}$ and an iteration cap of 80 per force. Iterations and setup seconds sum over the trajectory. The condition column is the maximum observed Ritz estimate. Each configuration was timed once, and timing differences in this table are exploratory. [Raw measurements](results/evolution_confirm_20261119/measurements.csv)

| Normalization | Preconditioner | Start | CG iterations | Condition estimate | Preconditioner setup (s) | Integration (s) | Global root error |
|---|---|---|---:|---:|---:|---:|---:|
| Constrained | Identity | Cold | 163 | 6.87 | <0.00001 | 0.178 | 0.0673094 |
| Endpoint normalization only | Identity | Cold | 171 | 11.00 | <0.00001 | 0.187 | 0.0673011 |
| Constrained | Identity | Reprojected | 151 | 6.87 | <0.00001 | 0.183 | 0.0673094 |
| Constrained | Product | Cold | 136 | 7.36 | 0.00510 | 0.218 | 0.0673094 |
| Constrained | Product | Reprojected | 125 | 7.36 | 0.00507 | 0.209 | 0.0673094 |

With no tangent constraint, the largest $|\mathrm{Tr}(XV)|$ reached 0.00957 before midpoint/endpoint retraction. The constrained solver held it to $6.99\times10^{-18}$ in the cold identity run. Reprojection and the product preconditioner reduced iteration counts. Applying the latter incurred extra contraction cost. A working bond of one rejected the fixed root coefficient with discard 0.305. Bond two retained the root but rejected the source with discard 0.554. Bonds four and eight reproduced the work-16 result. [Coefficient failure](results/evolution_confirm_20261119/05/integration.json) · [Source failure](results/evolution_confirm_20261119/06/integration.json)

## State-bond and step refinement

Each entry is an independently measured global root error for the same two-site evolution and final time. Bond one includes initial projection onto the capped space. At the full three-spin bond of four, each step halving reduced error by about four. Smaller state bonds reached an error plateau. With bond four and a $10^{-5}$ step-doubling criterion, three rejected trials and four accepted two-half-step updates reached $1.66\times10^{-5}$ global error. The local tolerance applies to the divided-by-three full/half-step discrepancy. [Prespecified grid](results/evolution_confirm_20261119/grid.json) · [Adaptive history](results/evolution_confirm_20261119/18/integration.json)

| Root bond cap | 2 steps | 4 steps | 8 steps |
|---|---:|---:|---:|
| 1 | 0.21657 | 0.21656 | 0.21655 |
| 2 | 0.056792 | 0.056796 | 0.056802 |
| 4 | 0.00027049 | 0.000066911 | 0.000016636 |

## Positive seed

For excited-state loss the initially empty sector has unit kernel-kernel forcing. The exact root is diagonal with entries $\sqrt{1-e^{-t}}$ and $e^{-t/2}$. At $t=10^{-18}$, the finite Kraus factor retained the new $10^{-9}$ root eigenvalue. The stored root differed from the analytic solution by $2.72\times10^{-17}$. At $t=0.04$, four, eight and sixteen equal intervals in $u=\sqrt t$ gave endpoint errors $7.11\times10^{-6}$, $9.24\times10^{-7}$ and $1.07\times10^{-7}$. The first interval in every run is the positive seed. [Tiny-population test](results/evolution_confirm_20261119/22/integration.json) · [Rank-opening trajectories](results/evolution_confirm_20261119/measurements.csv)

The confirmation ran 23 prespecified cases in separate processes with 20-second and 768-MiB caps. All configurations have per-worker hashes for the numerical sources and benchmark driver. Both failures stopped before a first accepted step. Scoring times are recorded separately. An earlier exploratory run is retained in its own directory. Run the command below from the repository root with a fresh output path. [Environment and caps](results/evolution_confirm_20261119/environment.json) · [Exploratory cohort](results/evolution_ablations_20261119/measurements.csv)

```sh
python3 -m galerkin_sylvester_tdvp.evolution_ablation --out run_outputs/galerkin-integration-ablations
```
