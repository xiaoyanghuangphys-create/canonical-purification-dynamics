# Reproduce the parity comparison

[Read the report](report.md) · [Derivations and dense checks](math.md) · [Global solvers](global_methods.md) · [Local solvers](local_methods.md) · [Matched timings](analysis/frontier.csv)

This study evolves a full-rank, globally parity-correlated state of spin-½ sites under uniform onsite dephasing. It asks which algorithms reproduce the **stored positive canonical root** at a specified endpoint error when the physical conditional mutual information persists across arbitrarily wide buffers.

## Checkout and environment

Use the repository root as the working directory. The comparison branch is based on commit `23fbd4c89a630688e65d441d9edb6a0056808d21` of the fork's `research/galerkin-sylvester-tdvp` branch. The production adapter snapshot is `8ad2e03`. The optional Newton loader reads two Git blobs at `ba0d08c6b6cce07733204a7e2a9334dabb046134` and verifies their SHA-256 digests before import. No additional checkout is required. [Base revision](https://github.com/xiaoyanghuangphys-create/canonical-purification-dynamics/commit/23fbd4c89a630688e65d441d9edb6a0056808d21) · [Newton revision](https://github.com/xiaoyanghuangphys-create/canonical-purification-dynamics/commit/ba0d08c6b6cce07733204a7e2a9334dabb046134)

The recorded environment was CPython 3.9.6, NumPy 2.0.2, SciPy 1.13.1, Matplotlib 3.9.4 and pytest 8.4.2 on arm64 macOS 26.5.1 with Accelerate. Reproduction with a different numerical library, CPU or interpreter can change rounding, stage acceptance and timings. Use the following environment in a fresh clone. Set the numerical thread limits before Python starts.

```sh
git clone --branch research/parity-local-lift-comparison https://github.com/xiaoyanghuangphys-create/canonical-purification-dynamics.git
cd canonical-purification-dynamics
git fetch origin research/tangent-newton-witness
python3 -m venv .venv
. .venv/bin/activate
python -m pip install numpy==2.0.2 scipy==1.13.1 matplotlib==3.9.4 pytest==8.4.2
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/parity-local-mpl
```

## Run

Each raw result directory must be new. These commands write new measurements beneath a temporary directory and leave the historical datasets available for comparison.

```sh
study=parity_local_lift_comparison
out=$(mktemp -d)
python -m parity_local_lift_comparison.benchmark --suite smoke --out "$out/smoke"
python -m parity_local_lift_comparison.oracle_validation --out "$out/oracle" --figure "$out/cmi.png"
python -m parity_local_lift_comparison.local_controls --output "$out/local_controls"
python -m parity_local_lift_comparison.global_replay --out "$out/native_512"
for suite in width refine horizon controls scale historical-grid screen; do
    python -m parity_local_lift_comparison.benchmark --suite "$suite" --out "$out/$suite"
done
for grid in bond_and_skip raw_refinement; do
    python -m parity_local_lift_comparison.benchmark --grid "$study/grids/$grid.json" --out "$out/$grid"
done
python -m parity_local_lift_comparison.benchmark --suite confirm --grid "$study/grids/confirm8.json" --repeats 4 --out "$out/confirm8"
python -m parity_local_lift_comparison.benchmark --grid "$study/grids/confirm_large.json" --repeats 4 --out "$out/confirm_large"
python -m parity_local_lift_comparison.benchmark --grid "$study/newton_grid.json" --repeats 3 --out "$out/newton"
python -m parity_local_lift_comparison.analyze --results "$out" --out "$out/tables" --figures "$out/figures"
```

To rerun a specific archived cohort, pass its manifest to `--grid`, its recorded `--suite` label and its `--repeats` value, with an unused output path. The CLI also accepts `--methods` and `--sizes` filters. Suitable quick entry points are `--suite width` and `--suite scale --sizes 8 16`. The optional Newton cohort is computationally costly, and its 64-spin workers can exceed the recorded memory budget. Build comparisons from the committed raw measurements without rerunning the solvers as follows.

```sh
python -m parity_local_lift_comparison.analyze --out "$out/rebuilt-tables" --figures "$out/rebuilt-figures"
python -m parity_local_lift_comparison.oracle_validation --plot-only --out "$study/results/oracle_01" --figure "$out/rebuilt-figures/cmi.png"
```

## Validation

```sh
python -m pytest -q parity_local_lift_comparison/tests galerkin_sylvester_tdvp/tests factor_tdvp factor_tdvp_comparison factor_tdvp_accelerated .github/scripts
python half_petz_evolution/validate_half_petz_evolution.py
python half_petz_evolution/test_half_petz_evolution.py --n 6 --out "$out/predecessor-hp"
python local_cp_large/local_cp_large.py --part sanity --outdir "$out/predecessor-fit"
node "$study/tests/check_report.mjs" "$out/katex.json" "$study/"*.md
python "$study/tests/check_gfm.py" "$out/gfm.json" "$study/"*.md
```

The two document checks use KaTeX (an installed package or VS Code's bundled renderer) and cmark-gfm (an executable or library). They check the equations, table geometry and local links. After publication, pass the immutable public commit SHA and a fresh output path to the read-only GitHub HTML check with the same document list.

```sh
python "$study/tests/check_published.py" "$(git rev-parse HEAD)" "$out/published.json" "$study/"*.md
```

The dense oracle builds physical matrices through 256 dimensions (eight spins), and caps dense unitary allocations at 65,536 elements. The matrix-free attaining action covers larger local-gate dimensions in the dense validation cohort. The main worker checks every accepted witness endpoint against its independent global CP error with an absolute roundoff allowance of 2e-10. An orthogonal QR contraction checks the native 512-spin checkpoint separately. Tests exercise native regression, positivity and allocation refusals, source isolation, raw scoring, inactive sites, tensor gauge and uncertainty in fitted directions. [Validation logs](results/validation) · [Predecessor numerical controls](results/local_predecessors_01)

## Data and accounting

| Evidence | Contents |
|---|---|
| [Oracle measurements](results/oracle_01) | Dense Kraus and polar controls, CMI and its per-step loss, the attained optimum, half-Petz diagnostics, equal-marginal counterexample |
| [Local validation](results/local_controls_01) | Native and alternative completions, local ranks, inactive-site checks and the finite-range cluster control |
| [Widths](results/width_01) · [time refinement](results/refine_01) · [bond and skip policies](results/bond_and_skip_01) | Window sizes, whole-chain controls, time subdivisions through 100, two to eight allowed bond states |
| [Parameter controls](results/controls_01) · [time horizons](results/horizon_01) · [raw A/B refinement](results/raw_refinement_01) | Stationary, product, weak and strong parity parameters, five-interval trajectories to 8 and raw predictor refinement |
| [Size grid](results/scale_01) · [historical endpoints](results/historical_01) · [native replay](results/global_native_512) | One-interval runs to 512 spins, tight historical replay and independent checkpoint |
| [Smoke](results/smoke_01) · [initial local smoke](results/local_smoke_01) · [screen](results/screen_01) | Single-worker correctness and cost exploration |
| [Four-repeat eight-spin grid](results/confirm_01) · [32- and 512-spin repeats](results/confirm_large_01) · [optional Newton](results/global_newton_01) | Timing distributions, solver failures and earlier validated Newton extension |
| [Worker table](analysis/workers.csv) · [endpoints](analysis/endpoints.csv) · [failures](analysis/failures.csv) · [frontier](analysis/frontier.csv) · [summary](analysis/summary.json) | Auditable one-row-per-worker and per-endpoint views, settings that hit each target, archive and certificate counts |
| [Six figures](figures) | Generated from the native data and analytic predictions, using [analysis code](analyze.py) and [oracle checks](oracle_validation.py) |

Every common grid is stored before execution in its manifest alongside the Git revision, hashes of source modules, interpreter, libraries and thread settings. Early local pilot runs began before the first adapter commit. Their manifests record the actual source bytes and worker-level changes. The original replay and dedicated validation harnesses record their own provenance. Each common run has a write-once result directory, a JSON-lines measurement ledger and a sealed ZIP with worker configuration, diagnostic event logs, checkpoint scores and exceptions. Git stores the worker archives once. Archive checksums and complete index/repeat coverage are verified by the analysis. The optional Newton manifest and other custom grids can have the default suite label `smoke`; the directory and explicit configuration identify those cohorts.

The common runner launches one subprocess per configuration and repeat. Its production timer starts after imports and includes initialization, physical references, predictor/lift, correction, audit contractions and synchronous checkpoint diagnostics. Future-root scoring is timed separately. The outer process remainder includes interpreter startup, imports, shutdown and final serialization. Compression of the parent archive is outside both timers. The monitoring watchdog samples RSS every 40 ms. Per-worker wall limits, nominal bond caps and allocation limits are recorded explicitly. Local construction checks the cache and largest dense operands in advance, and the global algebra checks product/addition core allocations. Intermediate work and actual stored bonds appear in the worker diagnostics. Exploratory suites could share the machine with other workers. Within a repeated suite the workers ran one at a time in alternating forward/reverse order.

The frontier requires the exact requested time horizon, success of every independent repeat, at least three repetitions, and the specified actual endpoint error in each. Groups match method, physical parameters, valid protocol and the complete immutable configuration. Timing cells are the minimum median among eligible measured settings. The table gives the setting, error, repeat count, range, independent scoring time and process overhead alongside each success. A failed or incomplete worker makes its setting ineligible. A method without a hit has an explicit status and no matched-time ratio. Full-chain fitting data, whole-chain lifts, diagnostic completion and inactive-site audits are excluded. Active-only and full-sweep local costs have their own rows. Initial density preparation, metric access and global generator sweeps are charged as used. [Benchmark CLI and worker](benchmark.py)
