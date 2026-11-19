# Two-site root evolution with a positive endpoint witness

The correlated-chain comparison uses the measured global canonical-root error at eight spins and the conditional global witness bound at 16, 32 and 64 spins. The parity family has an independent analytic global root at every size. Method A uses the repository’s approximate ambient MPO predictor. Method B solves the tangent Galerkin equation. Each predictor feeds the same positive-witness endpoint procedure.

## Repeated cost at matched accuracy

A setting qualifies when every planned confirmation repeat finishes at the requested time and passes the target. The selected setting minimizes median production time, including setup, prediction, witness correction/certification and checkpoint I/O. Independent scoring time, complete worker time, RSS and selected configuration IDs are in the frontier table. A B/A value above one means B took longer. A dash means the grid did not establish that entry. [Accuracy frontiers](matched_accuracy.csv) · [Common step and witness settings](same_settings.csv)

### Global target 1e-05

| Family | Spins | Evidence | A error or bound | A seconds | B error or bound | B seconds | B/A |
|---|---:|---|---:|---:|---:|---:|---:|
| D | 8 | measured | 5.56e-07 | 0.297 | 5.54e-07 | 2.68 | 9.02 |
| D | 16 | conditional bound | 1.89e-06 | 0.727 | 1.89e-06 | 18.3 | 25.2 |
| D | 32 | conditional bound | 2.87e-06 | 1.59 | 2.84e-06 | 50.6 | 31.9 |
| D | 64 | conditional bound | 4.41e-06 | 3.32 | — | — | — |
| E | 8 | measured | 1.21e-06 | 0.172 | 2.36e-06 | 0.252 | 1.47 |
| E | 16 | measured | 1.21e-06 | 0.333 | 2.36e-06 | 0.581 | 1.74 |
| E | 32 | measured | 1.21e-06 | 0.66 | 2.36e-06 | 1.23 | 1.87 |
| E | 64 | measured | 1.21e-06 | 1.3 | 2.36e-06 | 2.54 | 1.96 |

### Global target 1e-07

| Family | Spins | Evidence | A error or bound | A seconds | B error or bound | B seconds | B/A |
|---|---:|---|---:|---:|---:|---:|---:|
| D | 8 | measured | 1.85e-08 | 0.791 | 1.85e-08 | 4.52 | 5.72 |
| D | 16 | conditional bound | 6.14e-08 | 2.36 | — | — | — |
| D | 32 | conditional bound | 9.28e-08 | 8.28 | — | — | — |
| D | 64 | conditional bound | — | — | — | — | — |
| E | 8 | measured | 3.35e-08 | 0.211 | 1.98e-08 | 0.297 | 1.41 |
| E | 16 | measured | 3.35e-08 | 0.408 | 1.98e-08 | 0.663 | 1.63 |
| E | 32 | measured | 3.35e-08 | 0.798 | 1.98e-08 | 1.39 | 1.75 |
| E | 64 | measured | 3.35e-08 | 1.57 | 1.98e-08 | 2.87 | 1.83 |

![Cost at global-root accuracy targets](matched_cost.png)

The frontier CSV also covers the looser $10^{-3}$ target. The error and bound columns give the largest observed value among all repeats of the selected configuration. The figure shows each method only at sizes with qualifying confirmation repeats. An unresolved upper bound leaves the target unconfirmed. It does not measure the true root error on the larger correlated chains.

## Coverage and failed endpoints

| Phase | Family | Spins | Method | Completed / planned workers | Status counts |
|---|---|---:|---|---:|---|
| confirm | D | 8 | A | 6 / 6 | completed (6) |
| confirm | D | 8 | B | 6 / 6 | completed (6) |
| confirm | D | 16 | A | 6 / 6 | completed (6) |
| confirm | D | 16 | B | 3 / 3 | completed (3) |
| confirm | D | 32 | A | 6 / 6 | completed (6) |
| confirm | D | 32 | B | 3 / 3 | completed (3) |
| confirm | D | 64 | A | 3 / 3 | completed (3) |
| confirm | E | 8 | A | 6 / 6 | completed (6) |
| confirm | E | 8 | B | 6 / 6 | completed (6) |
| confirm | E | 16 | A | 6 / 6 | completed (6) |
| confirm | E | 16 | B | 6 / 6 | completed (6) |
| confirm | E | 32 | A | 6 / 6 | completed (6) |
| confirm | E | 32 | B | 6 / 6 | completed (6) |
| confirm | E | 64 | A | 6 / 6 | completed (6) |
| confirm | E | 64 | B | 6 / 6 | completed (6) |
| screen | D | 8 | A | 2 / 2 | completed (2) |
| screen | D | 8 | B | 2 / 2 | completed (2) |
| screen | D | 16 | A | 2 / 2 | completed (2) |
| screen | D | 16 | B | 1 / 3 | completed (1), memory_budget (1), memory_limit (1) |
| screen | D | 32 | A | 2 / 2 | completed (2) |
| screen | D | 32 | B | 1 / 3 | completed (1), memory_budget (1), memory_limit (1) |
| screen | D | 64 | A | 1 / 3 | completed (1), witness_uncertified (2) |
| screen | D | 64 | B | 0 / 5 | memory_budget (1), memory_limit (1), rank_support_uncertain (2), timeout (1) |
| screen | E | 8 | A | 2 / 2 | completed (2) |
| screen | E | 8 | B | 2 / 2 | completed (2) |
| screen | E | 16 | A | 2 / 2 | completed (2) |
| screen | E | 16 | B | 2 / 2 | completed (2) |
| screen | E | 32 | A | 2 / 2 | completed (2) |
| screen | E | 32 | B | 2 / 2 | completed (2) |
| screen | E | 64 | A | 2 / 2 | completed (2) |
| screen | E | 64 | B | 2 / 2 | completed (2) |

The common-setting table pairs the endpoint grid, state cap, witness levels, passes and tolerance. Each predictor’s work cap appears separately. The screening rows in the frontier CSV give single-run observations. Confirmation uses its own independently timed workers. Each configuration retains its statuses, completed endpoint count and reached times. The raw records hold the trial-level diagnostics. [All configurations](configurations.csv) · [Raw worker measurements](../measurements.json) · [Run environment](../environment.json)
