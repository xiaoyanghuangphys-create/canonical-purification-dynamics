# Measurement index

Run directories are immutable. Each main driver refuses an existing destination. A terminal global error is present only when the requested endpoint is reached and independently scored.

| Data | Scope |
|---|---|
| [witness_preconditioned_20261119](witness_preconditioned_20261119) | Positive-witness A versus preparation-preconditioned B at matched endpoint accuracy, with same-tolerance unpreconditioned controls. [Results and reproduction](../witness_preconditioned_report.md) |
| [witness_preconditioned_d64_20261119](witness_preconditioned_d64_20261119) | Fixed two-interval correlated 64-spin comparison at $10^{-7}$ with explicitly bounded high-bond force diagnostics. |
| [witness_preconditioned_pairs_20261119](witness_preconditioned_pairs_20261119) | Six order-balanced, adjacent 64-spin pairs at the frozen one-interval setting from the main study. |
| [witness_pair_20261119](witness_pair_20261119) | Two-site A and B with a shared positive endpoint witness, correlated and parity chains at 8, 16, 32 and 64 spins. [Study and reproduction commands](../witness_pair_report.md) |
| [witness_pair_streamed_20261119](witness_pair_streamed_20261119) | Fresh paired runs with one and two midpoint intervals, streamed Galerkin contractions, a stronger stage-positivity check and additional witness levels at the tight D target. |
| [witness_pair_two_step_20261119](witness_pair_two_step_20261119) | Repeated fixed two-interval comparison using the streamed B solver and identical endpoint grids for A and B at $10^{-5}$. |
| [witness_pair_brackets_20261119](witness_pair_brackets_20261119/report.md) | Conditional two-sided estimates of achieved global root errors using tighter completed checkpoints. |
| [comparison_20261119](comparison_20261119) | Definitive D/E measured-error comparison, 180 workers on one frozen implementation. |
| [restricted_20261119](restricted_20261119) | Four-spin interacting loss with aligned Euclidean and Galerkin integrator geometry. 51 workers, including six factor failures. |
| [physics_20261119](physics_20261119) | 60 interacting and exactly soluble controls. |
| [refine_20261119](refine_20261119) | 12 step- and bond-refinement trajectories. |
| [scaling_20261119](scaling_20261119) | Initial eight D/E size-extension attempts. |
| [scaling_followup_20261119](scaling_followup_20261119) | Eight independent D/E size-extension attempts, grouped positivity audit/work refinement for D and smaller time steps for E. |
| [oracle_20261119b](oracle_20261119b) | Final 38 dense-objective, conditioning and solver ablation records. |
| [evolution_confirm_20261119](evolution_confirm_20261119) | Final 23 integrator controls including two expected guard failures. |
| [validation_20261119](validation_20261119) | New and predecessor regression logs, machine information and validation records. |
| [pilot_20261119](pilot_20261119), [matched_20261119](matched_20261119), [oracle_20261119a](oracle_20261119a), [evolution_ablations_20261119](evolution_ablations_20261119), [restricted_pilot_20261119](restricted_pilot_20261119) | Development cohorts retained for provenance. They are excluded from the definitive comparison. |

An archive named `workers.zip` contains the original per-process files; its adjacent SHA-256 verifies the archive bytes. The flat measurements CSV is a convenience index. The corresponding JSON files carry numeric types, complete timing data and available iteration traces. In the native root/witness driver the common cap/work columns describe the shared harness; actual native levels and measured bonds have separate fields. It performs internal reference scoring and I/O in addition to final independent scoring. Development-run environment hashes describe the source at launch; solver development continued during those preliminary cohorts. The definitive repeated cohort carries per-worker hashes.
