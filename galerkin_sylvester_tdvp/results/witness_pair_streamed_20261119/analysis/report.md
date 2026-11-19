# Two-site root evolution with a positive endpoint witness

The correlated-chain comparison uses the measured global canonical-root error at eight spins and the conditional global witness bound at 16, 32 and 64 spins. The parity family has an independent analytic global root at every size. Method A uses the repository’s approximate ambient MPO predictor. Method B solves the tangent Galerkin equation. Each predictor feeds the same positive-witness endpoint procedure.

## Repeated cost at matched accuracy

A setting qualifies when every planned confirmation repeat finishes at the requested time and passes the target. The selected setting minimizes median production time, including setup, prediction, witness correction/certification and checkpoint I/O. Independent scoring time, complete worker time, RSS and selected configuration IDs are in the frontier table. A B/A value above one means B took longer. A dash means the grid did not establish that entry. [Accuracy frontiers](matched_accuracy.csv) · [Common step and witness settings](same_settings.csv)

### Global target 1e-05

| Family | Spins | Evidence | A error or bound | A seconds | B error or bound | B seconds | B/A |
|---|---:|---|---:|---:|---:|---:|---:|
| D | 8 | measured | 8.48e-07 | 0.14 | 8.37e-07 | 0.261 | 1.86 |
| D | 16 | conditional bound | 2.88e-06 | 0.344 | 2.82e-06 | 0.73 | 2.12 |
| D | 32 | conditional bound | 4.75e-06 | 0.745 | 4.56e-06 | 1.65 | 2.21 |
| D | 64 | conditional bound | 8.73e-06 | 1.55 | 8.03e-06 | 3.49 | 2.25 |
| E | 8 | measured | 2.94e-06 | 0.0992 | 2.66e-06 | 0.118 | 1.19 |
| E | 16 | measured | 2.94e-06 | 0.19 | 2.66e-06 | 0.236 | 1.24 |
| E | 32 | measured | 2.94e-06 | 0.373 | 2.66e-06 | 0.469 | 1.26 |
| E | 64 | measured | 2.94e-06 | 0.723 | 2.66e-06 | 0.937 | 1.3 |

### Global target 1e-07

| Family | Spins | Evidence | A error or bound | A seconds | B error or bound | B seconds | B/A |
|---|---:|---|---:|---:|---:|---:|---:|
| D | 8 | measured | 1.85e-08 | 0.497 | 1.85e-08 | 0.617 | 1.24 |
| D | 16 | conditional bound | 6.14e-08 | 1.5 | 6.14e-08 | 1.86 | 1.24 |
| D | 32 | conditional bound | 9.27e-08 | 3.45 | 9.27e-08 | 4.37 | 1.27 |
| D | 64 | conditional bound | 1.29e-08 | 20.3 | 1.29e-08 | 22.3 | 1.09 |
| E | 8 | measured | 2.46e-08 | 0.122 | 2.23e-08 | 0.139 | 1.14 |
| E | 16 | measured | 2.46e-08 | 0.236 | 2.23e-08 | 0.28 | 1.19 |
| E | 32 | measured | 2.46e-08 | 0.475 | 2.23e-08 | 0.559 | 1.18 |
| E | 64 | measured | 2.46e-08 | 0.926 | 2.23e-08 | 1.12 | 1.21 |

![Cost at global-root accuracy targets](matched_cost.png)

The frontier CSV also covers the looser $10^{-3}$ target. The error and bound columns give the largest observed value among all repeats of the selected configuration. The figure shows each method only at sizes with qualifying confirmation repeats. An unresolved upper bound leaves the target unconfirmed. It does not measure the true root error on the larger correlated chains.

## Coverage and failed endpoints

| Phase | Family | Spins | Method | Completed / planned workers | Status counts |
|---|---|---:|---|---:|---|
| confirm | D | 8 | A | 6 / 6 | completed (6) |
| confirm | D | 8 | B | 6 / 6 | completed (6) |
| confirm | D | 16 | A | 6 / 6 | completed (6) |
| confirm | D | 16 | B | 6 / 6 | completed (6) |
| confirm | D | 32 | A | 6 / 6 | completed (6) |
| confirm | D | 32 | B | 6 / 6 | completed (6) |
| confirm | D | 64 | A | 6 / 6 | completed (6) |
| confirm | D | 64 | B | 6 / 6 | completed (6) |
| confirm | E | 8 | A | 6 / 6 | completed (6) |
| confirm | E | 8 | B | 6 / 6 | completed (6) |
| confirm | E | 16 | A | 6 / 6 | completed (6) |
| confirm | E | 16 | B | 6 / 6 | completed (6) |
| confirm | E | 32 | A | 6 / 6 | completed (6) |
| confirm | E | 32 | B | 6 / 6 | completed (6) |
| confirm | E | 64 | A | 6 / 6 | completed (6) |
| confirm | E | 64 | B | 6 / 6 | completed (6) |
| screen | D | 8 | A | 4 / 4 | completed (4) |
| screen | D | 8 | B | 4 / 4 | completed (4) |
| screen | D | 16 | A | 4 / 4 | completed (4) |
| screen | D | 16 | B | 4 / 4 | completed (4) |
| screen | D | 32 | A | 4 / 4 | completed (4) |
| screen | D | 32 | B | 3 / 4 | completed (3), memory_budget (1) |
| screen | D | 64 | A | 4 / 4 | completed (4) |
| screen | D | 64 | B | 3 / 4 | completed (3), memory_budget (1) |
| screen | E | 8 | A | 4 / 4 | completed (4) |
| screen | E | 8 | B | 4 / 4 | completed (4) |
| screen | E | 16 | A | 4 / 4 | completed (4) |
| screen | E | 16 | B | 4 / 4 | completed (4) |
| screen | E | 32 | A | 4 / 4 | completed (4) |
| screen | E | 32 | B | 4 / 4 | completed (4) |
| screen | E | 64 | A | 4 / 4 | completed (4) |
| screen | E | 64 | B | 4 / 4 | completed (4) |

The common-setting table pairs the endpoint grid, state cap, witness levels, passes and tolerance. Each predictor’s work cap appears separately. The screening rows in the frontier CSV give single-run observations. Confirmation uses its own independently timed workers. Each configuration retains its statuses, completed endpoint count and reached times. The raw records hold the trial-level diagnostics. [All configurations](configurations.csv) · [Raw worker measurements](../measurements.json) · [Run environment](../environment.json)
