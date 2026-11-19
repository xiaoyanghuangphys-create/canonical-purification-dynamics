# Tangent-space Galerkin evolution of a canonical root

[Research report](report.md) · [Dense oracle and conditioning](oracle_report.md) · [Integrator ablations](integration_ablation.md) · [Restricted interacting comparison](restricted_comparison.md) · [Measured configurations](figures/configurations.csv) · [Accuracy-matched costs](figures/matched_accuracy.csv)

Method B advances a matrix-product operator for the positive root of a density matrix. Its force solves the Sylvester equation **inside** a one-site or two-site tangent space. The small-system dense oracle and the production MPO operator have separate entry points.

## Reproduce from the repository root

The measured environment used Python 3.9.6 on Apple silicon, NumPy 2.0.2, SciPy 1.13.1, Matplotlib 3.9.4 and pytest 8.4.2. With a compatible Python in a virtual environment, the following pins reproduce the numerical library versions. Set the thread limits before starting Python.

```sh
python3 -m pip install numpy==2.0.2 scipy==1.13.1 matplotlib==3.9.4 pytest==8.4.2
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/galerkin-mpl"
python3 -m pytest -q galerkin_sylvester_tdvp/tests
python3 -m pytest -q factor_tdvp factor_tdvp_comparison factor_tdvp_accelerated .github/scripts
```

Smoke test and independently assembled dense comparison, with **new** output directories:

```sh
python3 -m galerkin_sylvester_tdvp.benchmarks --suite smoke --out run_outputs/galerkin-smoke
python3 -m galerkin_sylvester_tdvp.oracle_ablations --out run_outputs/galerkin-oracle
python3 -m galerkin_sylvester_tdvp.evolution_ablation --out run_outputs/galerkin-evolution
```

Reproduce the physical grids, then generate the tables and cost plot. Each configuration gets a separate worker, a wall-clock deadline and sampled peak-RSS monitoring. The generator and Krylov routines impose additional tensor-size, truncation and iteration limits. The memory watchdog can overshoot between samples. A budget can leave unstarted repetitions, which are excluded from the repeated frontier.

```sh
python3 -m galerkin_sylvester_tdvp.benchmarks --suite physics --out run_outputs/galerkin-physics --timeout 35 --memory-mib 768 --budget 840
python3 -m galerkin_sylvester_tdvp.benchmarks --suite scaling --out run_outputs/galerkin-scaling --timeout 35 --memory-mib 768 --budget 300
python3 -m galerkin_sylvester_tdvp.benchmarks --suite refine --out run_outputs/galerkin-refine --timeout 35 --memory-mib 768 --budget 420
python3 -m galerkin_sylvester_tdvp.benchmarks --suite comparison --out run_outputs/galerkin-comparison --timeout 45 --memory-mib 768 --budget 1200
python3 -m galerkin_sylvester_tdvp.analyze_results run_outputs/galerkin-comparison --out run_outputs/galerkin-figures
```

Each run records its grid, environment and measurements. Published comparison/physics archives contain the original worker JSON, endpoint histories, failure details and stdout. Inspect a single worker or unpack into a fresh directory.

```sh
unzip -l galerkin_sylvester_tdvp/results/comparison_20261119/workers.zip
unzip -p galerkin_sylvester_tdvp/results/comparison_20261119/workers.zip '003_D3_B_one_s2_r1/measurement.json'
```

## Interfaces

- `tangent_frame.build_frame(root, mode)` provides complex coordinates, active and stored ranks, complement sizes, rank-trimming allowance, `encode`, `decode` and `project`. Local indices are ordered as ket bit followed by bra bit.
- `galerkin_solver.solve_velocity(root, generator, lower_bound=..., mode=...)` calls the supplied physical generator on the represented current density. It returns a velocity MPO and a diagnostic dictionary. An independent positive lower bound on the **current numerical** root or its Hermitian part is required. An unverified model floor at a later stage is insufficient.
- `galerkin_solver.dense_oracle(root, g_matrix, ...)` returns the explicit metric, basis, dense Sylvester solution and A/B/C minimizers under an allocation cap.
- `evolution.integrate(initial, generator, times, chi, ...)` returns the last accepted state and diagnostic history, including on numerical rejection. Inspect `status` and `final_time`. Time coordinates in the public grid are physical endpoints. Optional finite Kraus startup, square-root clock, projected warm starts, step doubling, dense polar retraction and D/E positive-witness endpoints are independently selectable.
- `models.make_model(family, n, work)` provides the initial root, current-density generator and a separate scoring callback. `D` and `E` retain the scaling study's family names. The remaining model names describe their physical setups.
- `benchmarks.py` and `analyze_results.py` contain the grids and evidence selection. `oracle_ablations.py` and `evolution_ablation.py` isolate force and integration effects.

The [base handoff](../HANDOFF.md) documents predecessor branches and datasets. All new source and measurements are contained in this directory.

The sharper large-chain positivity audit and its isolated eight-worker grid are reproduced separately:

```sh
python3 -m galerkin_sylvester_tdvp.scaling_followup --out run_outputs/galerkin-grouped-floor
```

[Grouped-stage bound and scaling results](scaling_followup.md)

The four-spin interacting comparison uses the same retraction and stage checks for A and B. Its tuning grid, repeats and worker limits are written before the runs begin.

```sh
python3 -m galerkin_sylvester_tdvp.restricted_comparison --out run_outputs/galerkin-restricted
```
