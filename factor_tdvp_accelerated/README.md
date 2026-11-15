# Matrix-free factor-TDVP

Read [the research report](report.md) and [repository handoff](../HANDOFF.md).
The solver accelerates the original density-residual factor objective. It
uses real tangent coordinates and MPO-output LSQR, retaining factor positivity.
The existing finite-size rank-aware driver accepts it as an injected backend.

## Reproduce

From the repository root, using Python 3.9+ with NumPy, SciPy and Matplotlib:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest factor_tdvp.test_factor_tdvp factor_tdvp.test_rank_bridge factor_tdvp_accelerated.test_matrix_free_solver -v
python3 -m factor_tdvp_accelerated.run_benchmarks --suite all --out run_outputs/accelerated-new --timeout 120
python3 -m factor_tdvp_accelerated.run_ablations --out run_outputs/accelerated-ablations
python3 -m factor_tdvp_accelerated.product_scaling --n 64 --out run_outputs/product64.json
```

Create `run_outputs/` before the last command. The benchmark suite requires a
fresh output directory and writes a result after each isolated worker; failures
and timeouts remain in its tables. `--suite force`, `trajectories`,
`matched_witness`, or `witness` selects a subset. `all` includes the same-grid
D3 witness controls; `witness` separately reproduces historical native grids.
Measured environments are stored alongside results; this run used the available
NumPy 2.0.2 / SciPy 1.13.1 / Matplotlib 3.9.4 installation. Existing requirements
files describe older experiments and have not been silently rewritten.

```python
from factor_tdvp import factor_tdvp as fd
from factor_tdvp_accelerated.matrix_free_solver import factor_velocity
state, events = fd.evolve(
    initial_factor, hamiltonian, jumps, times, chi=4, work_bond=16,
    velocity_solver=factor_velocity,
    solver_options={"tolerance": 1e-10, "maxiter": 400},
)
```

The Hamiltonian/jumps above use the existing dense finite-size driver API.
For an MPO-only force use `factor_velocity(H, hamiltonian_mpo, jump_mpos,
work_bond=...)`. Large chains must declare a work bond. The force uses no dense
exports, but fixed bonds do not guarantee convergence or sufficient accuracy.
The compatibility `rcond` argument is not an LSQR singular-value cutoff.

Damping changes the objective. Coordinate scaling can change the selected
null-space representative. A warm start must be projected into the current
tangent frame using `TangentCoordinates(H).encode(previous_velocity)`; previous
coordinate vectors cannot be reused across changed canonical frames blindly.
Strict compression and stationarity guards are on by default. Disabling strict
mode is a diagnostic experiment, not a safe evolution setting.

`run_ablations.py` uses dense two-spin oracles and is explicitly separate from
the production solver. `product_scaling.py` checks only a force on a product
path, not an integrated trajectory or a correlated scalability claim.

## Report and figures

`plot_results.py` builds the figures and primary matched-accuracy tables from
saved authoritative datasets. `build_report_pdf.py` converts the Markdown note
to LaTeX and runs an installed Tectonic with cached packages. The PDF build
requires Tectonic and its TeX bundle; it does not rerun numerical experiments.


## Increasing-chain-size comparison with root TDVP and the positive witness

The reproducible [size-scaling benchmark runner](size_scaling.py) executes
correlated mixed-chain D trajectories in isolated, single-BLAS-thread workers.
Small-system direct factor and conventional root trajectories use the existing
finite-reference adapter. The native root+witness and correction-only methods
also run at larger chain lengths without importing dense reference scoring.
Failed, time-limited, and uncertified runs remain in the output.

Run from the repository root:

```sh
python -m factor_tdvp_accelerated.size_scaling \
  --suite smoke --timeout 90 --out run_outputs/size-scaling-smoke
python -m factor_tdvp_accelerated.size_scaling \
  --suite full --timeout 90 --out run_outputs/size-scaling-full
```

The [automated benchmark](../.github/workflows/factor-vs-witness-scaling.yml)
runs the full suite and records the source-branch measurement CSV and
[benchmark report](size_scaling_results/report.md). The complete raw worker
logs are uploaded as a GitHub Actions artifact. Missing rows or rows that did
not meet target accuracy must **not** be represented as measured infinite time.

**Important scope:** The corrected-root methods access separately evolved
physical density and an inverse-bound metric, whereas direct factor and
ordinary root methods do not. The factor evolution driver retains dense
rank/support checks even though its tangent force is matrix-free. For large N,
the witness supplies a global certified bound under its reference assumptions,
not an actual globally measured error. Therefore size scaling can establish
an empirical *current-implementation* cost-to-accuracy ranking on successful
overlaps, not universal or asymptotic TDVP superiority.
