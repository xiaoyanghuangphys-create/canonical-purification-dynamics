# Paired trajectory protocol

The D/E trajectories use the repository’s original two-stage ambient Sylvester predictor (A) and the preparation-preconditioned, constrained two-site Galerkin predictor (B). Every comparison of old and new endpoint updates within a predictor uses the same predictor code, inner tolerance, target, initialization, root bond and witness budget. After an accepted endpoint the two trajectories may generate different predictions because their carried roots differ. The frozen comparisons provide byte-for-byte identical inputs when comparing a prescribed A or B prediction. None of these paths uses a future exact root to choose a direction.

## Physical trajectories

- D has polarization 0.12 and preparation angle 0.35. The stronger preparation uses 0.35 and 0.8. The independently propagated density applies the local dephasing channel to the initial circuit-prepared MPO. The A and B root bonds both start at four.
- E has parity coefficient 0.6 and transverse coefficient 0.4. Onsite dephasing multiplies the transverse coefficient by $e^{-t}$. The stored root starts at bond two.
- A uses root-force work bonds 10 for D and 8 for E. B uses 64 and 16 respectively. B uses the earlier study’s selected relative tolerance for the family, size, number of intervals and endpoint target. For more than two intervals the harness uses the earlier two-interval choice. A setting absent from the selection file uses $10^{-9}$. Both members of each B pair use that choice.
- The fixed legacy witness anchors, Jensen metric and analytic unital channel floor come from the original implementation. The same weighted global certificate controls every accepted endpoint. Initial preparation still constructs the legacy anchors for both algorithms. D above eight spins has certificate-only scoring. At smaller D sizes the eigendecomposition of the propagated density is for scoring. E uses its analytic spectral root for scoring at every size.
- Except for explicitly labeled one-site tests, the Newton variation is two-site. The factor/root/work cap triples in each worker configuration reproduce the old level schedule, including the two additional high-bond levels for tight D. The loose D schedule permits up to four iterations per level. E allows eight. An attempted grid with no passing endpoint stops at the previous committed time. These fixed-grid cost comparisons measure the prescribed number of endpoints. Adaptive bisection is a separate controller.

## Measurement and replay

Each worker runs in a fresh Python process. The five thread-count environment variables in the manifest are set to one before importing NumPy. Each cohort directory must be new. Its manifest records the CLI, full input grid, source SHA and source-file digests. The measurement index records wall time including worker startup and any differences from the initial manifest’s source digests. Research modules under active development outside the timed path can appear in these differences. The per-worker ZIP contains the configuration, attempts, endpoint certificates, independent scores, timing details and the factor/root at every accepted step. A timeout or watchdog exit keeps the last progress and event logs, plus all completed checkpoints. The sampled resident memory limit can overshoot between polls.

Within the timed worker, production seconds exclude the independent scoring call and construction of a frozen prefix. Production includes the physical density, predictor, carried-state and predictor audits, fitting, physical correction, diagnostic contractions and checkpoint/event writes. Witness seconds cover the full endpoint call. The endpoint’s six duration categories are exclusive. Inner-solve fields are nested in the fit or correction time. The certificate engine updates its cached Jensen metric within certification. The native-worker cohort reproduces the predecessor’s original monitoring and clock boundaries and has its own manifest. It serves as a separate implementation control. The end-to-end total and sampled peak include scoring.

The parser in `paired_analysis.py` reads the ZIP archives, including on a fresh checkout. A configuration gets a timing median for the matched target only if every requested repeat completes with a passing bound. Failed trajectories have a last accepted time and diagnostic bounds. Summaries of screening times are distinct from repeated comparisons. Four confirmation passes alternate the complete order, so each old/new pair appears in each order twice. Exploratory grids can overlap other exploratory processes. Use the serial confirmation cohorts for timing comparisons.

The `paired_frozen` cohort rebuilds a common legacy A trajectory to obtain its late input. Dstress could not construct the prefix at the required tolerance. `paired_frozen_replay` instead loads the certified factor and root at $t=3.5$ from the observed strong-D A-new trajectory. The loader checks the original physical grid, saved time and endpoint certificate. The independently scored record stores hashes of the carried factor, carried root, prediction and density target. This compares the same previous state at $t=3.5$ against the same target at $t=4$ with old/old, new/new, mixed fit/correction, fit alone and correction alone. The spectra of the density relative to its initial anchor are computed after direction selection.

## Commands

From the repository root, choose a fresh output directory for each invocation. Each recorded manifest preserves the full command and limits for its cohort.

```sh
python3 -m tangent_newton_witness.paired --out /tmp/paired-native --native --sizes 8 --steps 1 2 --labels A-old B-old
python3 -m tangent_newton_witness.paired --out /tmp/paired-screen --sizes 8 --steps 1 2 --tolerances 1e-5 1e-7 --timeout 30
python3 -m tangent_newton_witness.paired --out /tmp/paired-confirm --sizes 8 --steps 1 --tolerances 1e-5 --repeats 4 --phase confirm
python3 -m tangent_newton_witness.paired --out /tmp/paired-late --families Dstress E --sizes 3 --steps 8 --total 4 --tolerances 1e-7 --repeats 4 --phase confirm
python3 -m tangent_newton_witness.paired --out /tmp/paired-replay --families Dstress --sizes 3 --steps 8 --total 4 --freeze-step 8 --checkpoint tangent_newton_witness/results/paired_late/r01_001_Dstress3_A-new.zip --labels A-old A-new B-old B-new Old-only New-only A-fit A-oldfit A-newfit A-newcorr --tolerances 1e-7
python3 -m tangent_newton_witness.paired_analysis tangent_newton_witness/results/paired_confirm1 tangent_newton_witness/results/paired_frozen_replay --out /tmp/paired-tables
```

`A-fit` and `A-oldfit` use the new and old square fits alone. `A-newfit` and `A-newcorr` replace only the indicated component of the legacy endpoint. `Old-only` and `New-only` omit the predictor and fit. `A-gn` and `GN-only` select the separately labeled real-linear least-squares endpoint. The horizon sweeps compare initial segments of the longer trajectory at constant time step. The step-refinement cohort compares different grids at the same final time. No random inputs enter these deterministic D/E experiments.
