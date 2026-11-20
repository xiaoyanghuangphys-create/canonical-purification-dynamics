# Global canonical-root error brackets

Each row covers the three confirmation repeats of the fastest qualifying setting at target 1e-5 in that cohort, chosen by median production time. The table shows the smallest lower and largest upper endpoint across those repeats. Times cover the evolution and positive witness. Scoring is offline.

For a stored root x and a tighter root y, d is the Frobenius norm of x − y, evaluated by orthogonalizing their tensor-train difference. The true canonical root at the same physical time is R. Its distance from y is bounded by the final witness radius e. The triangle inequality encloses the error of x between max(0, d − e − g) and min(own witness bound, d + e + g). Both witness bounds and floating-point accuracy are assumptions. The numerical guard g is 64 × double precision epsilon × spin count × sum of the maximum bond dimensions × max(1, ‖x‖ + ‖y‖). It is a numerical allowance without interval-arithmetic verification.

One common separately executed confirmed reference is selected per family, size and time across cohorts, prioritizing tolerance 1e-7 or tighter and then the smallest witness radius. Every displayed loose repeat has a strictly larger radius and a looser configured tolerance. Measured errors use the independent dense root on eight-spin D chains and the analytic root on E chains. References, configuration budgets and individual repeats are listed in the [checkpoint comparisons](brackets.csv).

| Family | Spins | T | Phase | Variant / cohort | Method | Repeats | Range across repeats | Measured range | Production s (median) |
|---|---:|---:|---|---|---|---:|---|---|---:|
| D | 8 | 0.1 | confirm | primary / witness_pair_20261119 | A | 3 | [5.18E-7, 5.94E-7] | 5.56e-07–5.56e-07 | 0.297 |
| D | 8 | 0.1 | confirm | primary / witness_pair_20261119 | B | 3 | [5.16E-7, 5.93E-7] | 5.54e-07–5.54e-07 | 2.68 |
| D | 8 | 0.1 | confirm | streamed-block / witness_pair_streamed_20261119 | A | 3 | [8.10E-7, 8.86E-7] | 8.48e-07–8.48e-07 | 0.14 |
| D | 8 | 0.1 | confirm | streamed-block / witness_pair_streamed_20261119 | B | 3 | [7.99E-7, 8.75E-7] | 8.37e-07–8.37e-07 | 0.261 |
| D | 8 | 0.1 | confirm | streamed-block / witness_pair_two_step_20261119 | A | 3 | [5.18E-7, 5.94E-7] | 5.56e-07–5.56e-07 | 0.286 |
| D | 8 | 0.1 | confirm | streamed-block / witness_pair_two_step_20261119 | B | 3 | [5.16E-7, 5.93E-7] | 5.54e-07–5.54e-07 | 0.629 |
| D | 16 | 0.1 | confirm | primary / witness_pair_20261119 | A | 3 | [8.37E-7, 9.60E-7] | — | 0.727 |
| D | 16 | 0.1 | confirm | primary / witness_pair_20261119 | B | 3 | [8.32E-7, 9.56E-7] | — | 18.3 |
| D | 16 | 0.1 | confirm | streamed-block / witness_pair_streamed_20261119 | A | 3 | [1.31E-6, 1.45E-6] | — | 0.344 |
| D | 16 | 0.1 | confirm | streamed-block / witness_pair_streamed_20261119 | B | 3 | [1.28E-6, 1.42E-6] | — | 0.73 |
| D | 16 | 0.1 | confirm | streamed-block / witness_pair_two_step_20261119 | A | 3 | [8.37E-7, 9.60E-7] | — | 0.719 |
| D | 16 | 0.1 | confirm | streamed-block / witness_pair_two_step_20261119 | B | 3 | [8.32E-7, 9.56E-7] | — | 2.05 |
| D | 32 | 0.1 | confirm | primary / witness_pair_20261119 | A | 3 | [1.26E-6, 1.46E-6] | — | 1.59 |
| D | 32 | 0.1 | confirm | primary / witness_pair_20261119 | B | 3 | [1.24E-6, 1.44E-6] | — | 50.6 |
| D | 32 | 0.1 | confirm | streamed-block / witness_pair_streamed_20261119 | A | 3 | [2.17E-6, 2.37E-6] | — | 0.745 |
| D | 32 | 0.1 | confirm | streamed-block / witness_pair_streamed_20261119 | B | 3 | [2.08E-6, 2.28E-6] | — | 1.65 |
| D | 32 | 0.1 | confirm | streamed-block / witness_pair_two_step_20261119 | A | 3 | [1.26E-6, 1.46E-6] | — | 1.58 |
| D | 32 | 0.1 | confirm | streamed-block / witness_pair_two_step_20261119 | B | 3 | [1.24E-6, 1.44E-6] | — | 4.87 |
| D | 64 | 0.1 | confirm | primary / witness_pair_20261119 | A | 3 | [2.05E-6, 2.08E-6] | — | 3.32 |
| D | 64 | 0.1 | confirm | streamed-block / witness_pair_streamed_20261119 | A | 3 | [4.14E-6, 4.17E-6] | — | 1.55 |
| D | 64 | 0.1 | confirm | streamed-block / witness_pair_streamed_20261119 | B | 3 | [3.80E-6, 3.83E-6] | — | 3.49 |
| D | 64 | 0.1 | confirm | streamed-block / witness_pair_two_step_20261119 | A | 3 | [2.05E-6, 2.08E-6] | — | 3.3 |
| D | 64 | 0.1 | confirm | streamed-block / witness_pair_two_step_20261119 | B | 3 | [1.94E-6, 1.98E-6] | — | 10.6 |
| E | 8 | 1 | confirm | primary / witness_pair_20261119 | A | 3 | [1.17E-6, 1.28E-6] | 1.21e-06–1.21e-06 | 0.172 |
| E | 8 | 1 | confirm | primary / witness_pair_20261119 | B | 3 | [2.29E-6, 2.39E-6] | 2.36e-06–2.36e-06 | 0.252 |
| E | 8 | 1 | confirm | streamed-block / witness_pair_streamed_20261119 | A | 3 | [2.86E-6, 2.97E-6] | 2.94e-06–2.94e-06 | 0.0992 |
| E | 8 | 1 | confirm | streamed-block / witness_pair_streamed_20261119 | B | 3 | [2.58E-6, 2.69E-6] | 2.66e-06–2.66e-06 | 0.118 |
| E | 8 | 1 | confirm | streamed-block / witness_pair_two_step_20261119 | A | 3 | [1.17E-6, 1.28E-6] | 1.21e-06–1.21e-06 | 0.171 |
| E | 8 | 1 | confirm | streamed-block / witness_pair_two_step_20261119 | B | 3 | [2.29E-6, 2.39E-6] | 2.36e-06–2.36e-06 | 0.199 |
| E | 16 | 1 | confirm | primary / witness_pair_20261119 | A | 3 | [1.17E-6, 1.28E-6] | 1.21e-06–1.21e-06 | 0.333 |
| E | 16 | 1 | confirm | primary / witness_pair_20261119 | B | 3 | [2.29E-6, 2.39E-6] | 2.36e-06–2.36e-06 | 0.581 |
| E | 16 | 1 | confirm | streamed-block / witness_pair_streamed_20261119 | A | 3 | [2.86E-6, 2.97E-6] | 2.94e-06–2.94e-06 | 0.19 |
| E | 16 | 1 | confirm | streamed-block / witness_pair_streamed_20261119 | B | 3 | [2.58E-6, 2.69E-6] | 2.66e-06–2.66e-06 | 0.236 |
| E | 16 | 1 | confirm | streamed-block / witness_pair_two_step_20261119 | A | 3 | [1.17E-6, 1.28E-6] | 1.21e-06–1.21e-06 | 0.331 |
| E | 16 | 1 | confirm | streamed-block / witness_pair_two_step_20261119 | B | 3 | [2.29E-6, 2.39E-6] | 2.36e-06–2.36e-06 | 0.408 |
| E | 32 | 1 | confirm | primary / witness_pair_20261119 | A | 3 | [1.17E-6, 1.28E-6] | 1.21e-06–1.21e-06 | 0.66 |
| E | 32 | 1 | confirm | primary / witness_pair_20261119 | B | 3 | [2.29E-6, 2.39E-6] | 2.36e-06–2.36e-06 | 1.23 |
| E | 32 | 1 | confirm | streamed-block / witness_pair_streamed_20261119 | A | 3 | [2.86E-6, 2.97E-6] | 2.94e-06–2.94e-06 | 0.373 |
| E | 32 | 1 | confirm | streamed-block / witness_pair_streamed_20261119 | B | 3 | [2.58E-6, 2.69E-6] | 2.66e-06–2.66e-06 | 0.469 |
| E | 32 | 1 | confirm | streamed-block / witness_pair_two_step_20261119 | A | 3 | [1.17E-6, 1.28E-6] | 1.21e-06–1.21e-06 | 0.653 |
| E | 32 | 1 | confirm | streamed-block / witness_pair_two_step_20261119 | B | 3 | [2.29E-6, 2.39E-6] | 2.36e-06–2.36e-06 | 0.823 |
| E | 64 | 1 | confirm | primary / witness_pair_20261119 | A | 3 | [1.17E-6, 1.28E-6] | 1.21e-06–1.21e-06 | 1.3 |
| E | 64 | 1 | confirm | primary / witness_pair_20261119 | B | 3 | [2.29E-6, 2.39E-6] | 2.36e-06–2.36e-06 | 2.54 |
| E | 64 | 1 | confirm | streamed-block / witness_pair_streamed_20261119 | A | 3 | [2.86E-6, 2.97E-6] | 2.94e-06–2.94e-06 | 0.723 |
| E | 64 | 1 | confirm | streamed-block / witness_pair_streamed_20261119 | B | 3 | [2.58E-6, 2.69E-6] | 2.66e-06–2.66e-06 | 0.937 |
| E | 64 | 1 | confirm | streamed-block / witness_pair_two_step_20261119 | A | 3 | [1.17E-6, 1.28E-6] | 1.21e-06–1.21e-06 | 1.32 |
| E | 64 | 1 | confirm | streamed-block / witness_pair_two_step_20261119 | B | 3 | [2.29E-6, 2.39E-6] | 2.36e-06–2.36e-06 | 1.67 |
