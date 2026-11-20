# Four-spin interacting loss with a shared retraction

On the full-rank four-spin Ising chain with onsite loss, direct Galerkin evolution took 0.288 s to reach measured global root error below $10^{-5}$. Euclidean projection of the dense Sylvester force took 0.154 s with the same two-site midpoint stages, state bond eight and four steps. Both reached global errors of $7.39\times10^{-6}$ at $t=0.03$. Times are median worker wall times across three serial repetitions, including setup, stage checks, evolution and independent scoring. [Repeated configurations](results/restricted_20261119/configurations.csv) · [Accuracy frontiers](results/restricted_20261119/matched_accuracy.csv)

The initial density is $0.8|+\rangle\langle+|^{\otimes 4}+0.2I/16$. The root adapters use the same initial compression, normalized Hermitian Pauli-coordinate TT-SVD, dense positivity audit, physical clock, current-state MPO generator and independent dense Lindblad reference. At each stage the A adapter solves the ambient Sylvester equation on the numerical root, applies the original MPS tangent projector, and removes the radial component. B uses its constrained Galerkin solver. The separate original Method A control retains its native dense polar retraction. Factor LSQR retains the factor objective and its own support handling. [Physical model](models.py) · [Comparison driver](restricted_comparison.py)

## Matched numerical budgets

Each cell is the maximum measured global root error over all three repeats at the indicated cap. All rows use working bond 32. Root and factor caps describe different stored tensors. LSQR failed its midpoint-force stationarity check on all six factor trials, with an 80-iteration cap and tolerance $10^{-9}$. The worker archives preserve the exceptions. [Raw measurements](results/restricted_20261119/measurements.csv) · [Prespecified grid](results/restricted_20261119/grid.json)

| Method | Bond 2, two steps | Bond 4, four steps | Bond 8, four steps |
|---|---:|---:|---:|
| A one-site, shared retraction | 0.0401401 | 0.00371771 | 0.00448487 |
| B one-site | 0.0401468 | 0.00405690 | 0.00486208 |
| A two-site, shared retraction | 0.0401400 | 0.000236914 | 0.00000739115 |
| B two-site | 0.0401400 | 0.000236914 | 0.00000739114 |
| A two-site, original polar | 0.0401400 | 0.000236914 | 0.00000739115 |
| Factor LSQR | Stationarity failure | Stationarity failure | Outside grid |

At bond two, the two-site complex dimension stayed 96 in an ambient space of dimension 256. At caps four and eight, the two-site dimension grew from 96 to 256 by the first midpoint. The subsequent unrestricted stages explain why very similar two-site endpoint errors can coexist with different restricted-tangent objectives. The one-site B dimension stayed 36 at bond two, reached 112 at bond four and reached 192 at bond eight. The one-site restricted dynamics missed both target accuracies throughout this grid. [Raw worker index](results/restricted_20261119/measurements.csv)

## Matched measured accuracy

Every selected configuration met its target in all three repeats. Each table entry gives the fastest observed median worker time for that method. Parent process startup time and worker-internal setup plus evolution time are both available in the configuration table. [Repeated frontiers](results/restricted_20261119/matched_accuracy.csv)

| Method | Target $10^{-3}$ | Target $10^{-5}$ |
|---|---:|---:|
| A two-site, shared retraction | 0.1527 s at cap 4 | 0.1539 s at cap 8 |
| B two-site | 0.2878 s at cap 8 | 0.2878 s at cap 8 |
| A two-site, original polar | 0.1188 s at cap 8 | 0.1188 s at cap 8 |

Relative to the shared-retraction A control, B cost 1.88 times as much at the looser target and 1.87 times as much at the tighter target. At equal cap four the worker medians were 0.1527 s for A and 0.2980 s for B. Physical reference scoring is itemized separately. For cap eight, median setup plus evolution times were 0.1509 s for A and 0.2835 s for B. Neither two-site method reached $10^{-7}$. These small-system dense-A measurements cover one physical trajectory and three numerical budgets. [Per-configuration medians](results/restricted_20261119/configurations.csv)

The confirmatory cohort contains 51 single-thread-requested serial processes with a 35-second and sampled 768-MiB RSS limit per worker. Of those, 45 reached the physical endpoint and six stopped at the factor solver. Each method/configuration repeated three times with identical numerical source hashes throughout the cohort. The [four-case smoke test](results/restricted_pilot_20261119/measurements.csv) ran separately. Use a fresh output directory for replay. [Environment](results/restricted_20261119/environment.json)

```sh
python3 -m galerkin_sylvester_tdvp.restricted_comparison --out run_outputs/galerkin-restricted --timeout 35 --budget 900 --memory-mib 768
```
