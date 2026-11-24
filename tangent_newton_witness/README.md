# Tangent Newton witnesses for canonical-purification dynamics

The witness direction is computed from the current MPO and its residual. The solver fits a predicted root with a square equation and corrects its physical density with a fourth-power equation. Both operations can be switched independently. Existing predictors and the independent global certificate remain available unchanged.

[Research report](report.md) · [Paired benchmark protocol](paired_methods.md) · [Conditioning and local convergence](conditioning.md) · [Frozen noncommuting checkpoints](frozen_report.md) · [Research handoff](HANDOFF.md)

## Run the validated example

From the repository root with Python 3.9 and the recorded NumPy/SciPy stack. The original machine used macOS on an Apple M2, NumPy 2.0.2, SciPy 1.13.1, Matplotlib 3.9.4 and pytest 8.4.2.

```sh
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/witness-newton-mpl"
python3 -m pytest -q tangent_newton_witness/tests galerkin_sylvester_tdvp/tests \
    factor_tdvp factor_tdvp_comparison factor_tdvp_accelerated .github/scripts

python3 -m tangent_newton_witness.paired --out run_outputs/newton-smoke \
    --families E --sizes 3 --steps 2 --total 1 --tolerances 1e-5 \
    --labels A-old A-new B-old B-new Old-only New-only
```

The run directory must be new. Its manifest fixes the inputs, thread settings and source-file hashes. Each isolated process writes a measurement and an archive containing step histories, independent scores and accepted tensors. Inspect `status`, `reached_time`, `global_cp_bound` and `global_cp_error` together. A partial run can preserve valuable checkpoints.

## Use the corrector

```python
from galerkin_sylvester_tdvp.models import witness
from tangent_newton_witness.endpoint import run_endpoint

n, time = 3, 0.5
engine = witness.WitnessAudit.parity_chain(n)
previous = witness.factor_to_root(witness.parity_power(n, .25), 2)
target = witness.channel(engine.physical, time)
state, info = run_endpoint(
    engine, previous, None, target, time,
    levels=[(4, 6, 8), (6, 8, 12)], tolerance=1e-7,
    fit="none", correction="new", mode="two", passes=8,
)
assert info["accepted"], info["status"]
factor, stored_root, witness_distance = state
```

* `operator.solve_direction(factor, target, power)` accepts 2 or 4 and returns a Hermitian tangent MPO plus numerical diagnostics. It holds one real frame fixed per solve. The optional `lower_bound` must hold for this actual factor. Positive bounds admit CG, and inconclusive bounds use residual-checked MINRES. A current-factor Jacobi option is available.
* `endpoint.run_endpoint` takes the previous accepted tuple, an optional predictor, an independently obtained target and a compatible certificate engine. Its `fit` and `correction` switches accept `old`, `new`, `gn` or `none`. New-only and Old-only supply `prediction=None`. GN uses an arbitrary complex factor and a capped dense ambient residual space. Direction failure, unsuccessful trials and rejections carry evidence in `info["attempts"]`.
* Each level sets factor, root and work caps. The fixed operator products have a separate `max_elements` ceiling per guarded allocation. Work caps control approximation in the endpoint diagnostics and legacy path. LAPACK scratch and process-wide RSS have separate costs.
* `adaptive.advance` accepts an interval callback and a halving budget. Its returned state and time are the last accepted checkpoint even when a later subinterval fails. `support.finite_start` classifies small-chain kernel forcing and uses the repository's finite positive bridge. The caller must certify a seed against its actual physical endpoint. The controls also exercise the existing whole-interval Kraus startup.

## Reproduce the measured comparisons

Each command writes a new cohort. Replace the output directories before rerunning. The published manifests store the original full argument vectors.

```sh
python3 -m tangent_newton_witness.paired --out run_outputs/newton-native \
    --native --sizes 8 --steps 1 2 --labels A-old B-old
python3 -m tangent_newton_witness.paired --out run_outputs/newton-n8 \
    --sizes 8 --steps 1 --tolerances 1e-5 --repeats 4 --phase confirm
python3 -m tangent_newton_witness.paired --out run_outputs/newton-tight-d8 \
    --families D --sizes 8 --steps 1 --tolerances 1e-7 --labels A-old A-new \
    --max-elements 8000000 --memory-mib 768 --timeout 45 --repeats 4 --phase confirm
python3 -m tangent_newton_witness.paired --out run_outputs/newton-track \
    --families Dstress E --sizes 3 --total 4 --steps 8 --tolerances 1e-7 --repeats 4 --phase confirm
python3 -m tangent_newton_witness.paired --out run_outputs/newton-scale \
    --families D E --sizes 16 32 64 --steps 1 --tolerances 1e-5 --timeout 12 --memory-mib 768
python3 -m tangent_newton_witness.stress --suite late --out run_outputs/newton-drift --timeout 35
python3 -m tangent_newton_witness.stress --suite frozen --out run_outputs/newton-frozen --timeout 35
python3 -m tangent_newton_witness.stress --suite short --out run_outputs/newton-ising --timeout 35
python3 -m tangent_newton_witness.controls --out run_outputs/newton-controls
python3 -m tangent_newton_witness.context --out run_outputs/newton-factor-context
python3 -m tangent_newton_witness.conditioning --out run_outputs/newton-conditioning
python3 -m tangent_newton_witness.figures --out run_outputs/newton-figures
```

Frozen A/B predictor swaps, the immutable late-checkpoint replay, loss-anchor controls, dense spectral checks and the raw figure tables have dedicated commands. [Paired replay](paired_methods.md) · [Loss and strong-D controls](frozen_report.md) · [Conditioning](conditioning.md)

## Data and clocks

The primary 24 paired cohorts have 392 worker archives, including pilots, resource exits and failed checkpoint generation. The final release check adds two four-repeat cohorts from the committed numerical source. Twelve further workers tested standard D at longer physical times. Raw outcomes remain distinct across cohorts. The paired worker clock includes initialization, independent propagation, predictor, fitting, physical correction, reconstruction, checks and event/checkpoint I/O. Production time excludes independent scoring and frozen-prefix setup. Process wall time also includes the Python launch. In the endpoint, the six top-level timers are exclusive. Frame, setup, Krylov and ambient-audit timers are nested in fitting or physical correction. The shared certificate engine updates its metric inside certification. Diagnostic contractions and the number of attempted trials affect measured witness cost. Stress controls and native baselines have separately documented clocks.

| Data | Contents |
|---|---|
| [Paired configurations](results/paired_tables/configurations.csv) | Outcome count, repeat eligibility, timings and scores for each pre-release setting |
| [Trajectory table](results/paired_tables/runs.csv) | Completed trajectories and resource/accuracy failures |
| [Checkpoints](results/paired_tables/endpoints.csv) · [trial and solve records](results/paired_tables/iterations.csv) | Per-step merit, error, actual bonds, inner iterations and timing evidence |
| [Tight repeated D8](results/paired_tuned_confirm/measurements.jsonl) | Eight fresh higher-budget workers |
| [Sequential late repeats](results/paired_late_serial/measurements.jsonl) | Common strong-D/E late targets, four fresh runs each |
| [Release D/E](results/release_n8/measurements.jsonl) · [release tracking](results/release_late/measurements.jsonl) | Fresh repeats at the committed solver revision |
| [Stress](results/stress_late_final/index.tsv) · [frozen](results/stress_frozen_final/index.tsv) · [support and raw controls](results/controls_final/index.tsv) | Worker indices, digests and complete ZIP archives |
| [Validation](results/validation_01/) | Actual test output, dense certificate and support checks |

The historical base paper's files and results remain in their original directories. The tested larger-chain grids stop at 64 spins. The tight larger-chain follow-up compares correction-only at one interval. The full A/B two-interval tight grid at those sizes, a 128-spin study and scalable generic Ising support propagation remain open.
