# TDVP implementation handoff

Updated 2026-11-14. This file is the starting point for a fresh Codex chat.

## Repository and task

- Fork: `xiaoyanghuangphys-create/canonical-purification-dynamics`.
- Base: `research/factor-tdvp-benchmarks`, commit `facc0b7c85f2fbf816607f6365d87233a05e4054`.
- Working branch: `research/factor-tdvp-accelerated` (already existed at the base; continued without resetting it).
- The benchmark branch is not an ancestor of `origin/main` as checked at clone time.
- Do not modify the upstream `yijian-physics` repository or merge a pull request automatically.

## History recovered

All available text in Implement TDVP Method (5 turns), Summarize TDVP Results
(5 turns), and two Continue TDVP Implementation chats (1 turn each) was read.
All four reads ended with no further pages or message truncation. The two
continuation chats contained only initiating user requests. Hidden reasoning,
complete original tool logs and old sandbox files are not supplied by that API.
The current uploaded acceleration prompt is byte-identical to the historical
unanswered acceleration attachment (SHA256
`1b387087176fbaf2a1ae4cd8794602802c5ca6a897aec065ba1cb18728bb66b8`).
Raw chat archives are not published in this repository.

The completed predecessor work was the finite-size factor solver, benchmark
comparison, and math/PDF rendering repair at the base commit. There was no
accelerated implementation to recover. Prior claims are historical evidence;
new measurements live in `factor_tdvp_accelerated/results/`.

## Current implementation

- `factor_tdvp_accelerated/matrix_free_solver.py`: real-coordinate one-site
  tangent maps, MPO Jacobian/adjoint, MPO-output LSQR with residual checks,
  compression/nonconvergence guards, optional damping, diagonal coordinate
  scaling, and projected warm-start support. No dense output-space vectors or
  explicit Jacobian columns in this production force.
- `factor_tdvp/tangent_solver.py`: optional `velocity_solver` / `solver_options`
  injection into midpoint integration. The explicit backend stays the default.
- `factor_tdvp/factor_tdvp.py`: forwards injection while retaining the dense
  first/repeated/delayed support audits, positive bridge, and rank-loss guards.
- `factor_tdvp_accelerated/test_matrix_free_solver.py`: independent regression
  tests; `run_benchmarks.py`: isolated-worker baseline comparisons;
  `run_ablations.py`: explicitly dense small-system conditioning/metric oracles;
  `product_scaling.py`: analytically checked MPO-only product-path force sweep.
- See the accelerated README for commands and report for measured conclusions.

## Interpretation and remaining research

This is a force acceleration, not a generic scalable end-to-end factor method.
The preserved rank audits and general bridge remain dense. LS stationarity can
miss physically important small-singular-value components; interacting-loss
accuracy must be checked separately. A product-state force check through 64
sites is not a correlated 64-site trajectory. No local error replaces a global
canonical-root error in the results.

Next priorities: physical-error-aware stopping and adaptive stepping; a useful
local preconditioner in the restricted tangent; scalable rank/support handling;
compressed-Krylov error control; two-site/adaptive bond growth; and independent
reference/certification cost at matched global error. ALS/local sweeps and
alternative root-metric evolution are not production implementations here.
Never overwrite old measurements; choose fresh output directories.

## Verified checkpoint

- Combined predecessor + accelerated tests: **48 passed**; see
  `factor_tdvp_accelerated/results/tests_full.log`.
- Final D6 profiled force: explicit 14.827 s, MPO-LSQR 0.547 s (27.1x),
  relative velocity difference 3.83e-9.
- D3 minimum observed cost at global CP error <=1e-7: matrix-free 2.092 s,
  conventional root 0.131 s, witness 0.097 s, correction-only 0.047 s.
  Witness comparisons use the same available uniform endpoint grids but extra
  independent density/metric correction information.
- F_loss3 remains an accuracy failure for the default matrix-free tolerance:
  CP error 0.00427; tolerance 1e-12 improves it to 1.58e-4, versus explicit
  5.94e-5. Do not hide this by reporting stationarity or density error alone.
- MPO-only product-force checks completed through 64 sites with global relative
  velocity error below 5e-14. These are not integrated correlated trajectories.
- Authoritative dataset selection is in
  `factor_tdvp_accelerated/results/BENCHMARK_MANIFEST.md`. Preliminary data are
  retained, including the documented correction to residual aggregation.

The typeset report is `factor_tdvp_accelerated/report.pdf`; its pages were rendered
and visually checked. Build source is retained alongside the Markdown.

## Publication

Implementation and measured results were pushed in commit
`001ccc1b426827c407c8fd16dcca72a62477e51c`.
[PR #2](https://github.com/xiaoyanghuangphys-create/canonical-purification-dynamics/pull/2)
is open against `research/factor-tdvp-benchmarks`; it has not been merged.
The local branch tracks `origin/research/factor-tdvp-accelerated`.
