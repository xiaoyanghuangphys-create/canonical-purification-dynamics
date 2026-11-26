# Measurements of the two local lifts

At $`N=6`$, $`p=0.6`$, $`b_0=0.4`$, and $`T=1`$, the native marginal-based unitary at $`W=2`$, $`R=3`$ gave a global canonical-root error of $`0.01928021377671`$. Its local fit residual was $`2.63\times10^{-13}`$. A global-target unitary oracle on the same two spins attains $`0.00913725795001`$. The native half-Petz map gave a raw error of $`0.00426878058765`$ for every proper active width tested in the dense validations. Its normalized diagnostic was $`0.00364006653784`$. All numbers here use the stored amplitude at the endpoint.

## Data and scheduling

The input is a normalized canonical-gauge MPS. Both adapters contract their physical marginals from this input. Every subsequent physical update uses the supplied Kraus channel on every cached region containing the acted-on site. The trajectory dependence enters through the timestep and these independently evolved regions. The input MPS remains available to the caller.

The nominally active site is the first, at the open boundary. Physical site indices start at zero in the implementation. In a uniform sweep, all sites are processed from left to right. A window of actual width $`W`$ starts at $`\max(0,\min(j-\lfloor(W-1)/2\rfloor,N-W))`$. The same convention gives the containing fitting region $`R`$. Active-only operation allocates the first site's cache. The uniform identity policy walks the same site schedule and applies a gate only at sites independent model information designates as changing. All three protocols record the nominal, applied and skipped counts and the number of updated regional caches. Per-gate arithmetic comes from the original local kernels. The comparison schedule differs from the colored layers of the original general-purpose fit driver.

Half-Petz evolves the physical factor by stacking the Kraus-transformed columns, retains singular values above $`10^{-13}`$ relative to the largest and uses their inverses on that support. Each updated block is split at the requested bond cap and a $`10^{-14}`$ relative cutoff. The native amplitude is retained after each operation. The step diagnostics expose the largest supported inverse norm, cumulative discarded cache mass and sum of truncation bounds. The fit uses the original positive marginal-root routine, compact gate, $`10^{-13}`$ relative split cutoff and normalization at the orthogonality center after every gate. The timing fields separate cache construction at initialization, gate construction, block application including splitting, and subsequent physical cache propagation. The estimated short dimension of the pretruncation split matrix is recorded as an upper bound. An optional inactive-site audit compares centered blocks before and after an update, with identical exterior tensors. Its Frobenius norm equals the global amplitude change in this gauge.

Dense whole-system inputs have a separate role in the controls. For example, using a whole fitting region allows the physical cache to reveal $`p`$ even when the fitted gate acts on a proper window. The physical dimension and the doubled-site vector grow exponentially with the chosen region. A preallocation guard covers the local physical cache, contraction operands and the reduced Procrustes cross matrix. [Adapter](local.py) · [Original practical fit](../local_cp_large/local_cp_large.py) · [Original half-Petz map](../half_petz_evolution/half_petz_evolution.py)

## Why the native practical fit misses the optimal proper-window error

Write the two normalized left Schmidt vectors as $`u_p(b)=\cos\alpha_p(b)\,I_W+\sin\alpha_p(b)\,Q_W`$ and $`P_W`$. Here $`I_W,Q_W,P_W`$ denote normalized operators, and $`\alpha_p(b)=\mathrm{atan2}(c(b)b,a(b))`$. A proper physical fitting region erases $`p`$. Its square root is a unit vector with angle $`\alpha_0(b)=\frac12\arcsin b`$. Across the cut between $`W`$ and $`R\setminus W`$, the input and target training matrices each have rank one, with the same exterior identity vector. This also holds for an empty training exterior when $`R=W`$.

The code takes the thin SVD of the concatenated source and target training columns. The retained basis $`B`$ has relative threshold $`10^{-13}`$. In the reduced space the native rotation $`V`$ is the unitary polar factor of the target/source cross matrix plus $`\lambda I`$, where $`\lambda=10^{-12}`$. It implements $`I+B(V-I)B^\dagger`$. For a nonstationary proper fitting region the joint space is the $`I_W,Q_W`$ plane. The unregularized cross matrix has rank one. The native completion rotates this plane and preserves $`P_W`$. If $`d=\alpha_0(b_T)-\alpha_0(b_0)`$, its rotation angle is $`d_\lambda=\mathrm{atan2}(\sin d,\cos d+2\lambda)`$. This follows from the polar factor of the real matrix with first column $`(\cos d+\lambda,\sin d)^T`$ and second column $`(0,\lambda)^T`$ in the source-aligned basis.

Set $`q_i=\sqrt{1-w_i}`$ and $`\Delta\alpha_p=\alpha_p(b_T)-\alpha_p(b_0)`$. Adding the squared orthogonal branch errors gives

```math
\epsilon_{\mathrm{fit}}^2
= \epsilon_U^2 + 2 q_0 q_T\big[1-\cos(d_\lambda-\Delta\alpha_p)\big].
```

The extra term is the angular mismatch between marginal-root alignment and global Schmidt-vector alignment. Refinement composes planar rotations on the same support. For the unregularized proper fit the marginal-root angles telescope. The native bias in the angle per step is $`O(\lambda |d|)`$ for these parameters.

| $`N=6`$, active $`W`$ | Fitting $`R`$ | Source rank | Unregularized cross rank | Joint space dimension | Measured local residual | Measured global error |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 1 | 1 | 2 | $`2.63\times10^{-13}`$ | 0.01928021378 |
| 2 | 3 | 1 | 1 | 2 | $`2.63\times10^{-13}`$ | 0.01928021378 |
| 2 | 5 | 1 | 1 | 2 | $`2.63\times10^{-13}`$ | 0.01928021378 |
| 2 | 6 | 2 | 2 | 3 | 0.00913725795 | 0.00913725795 |
| 6 | 6 | 1 | 1 | 2 | $`2.83\times10^{-13}`$ | $`2.83\times10^{-13}`$ |

All proper fitting regions in this table give equal input cache matrices for $`p=+0.6`$ and $`p=-0.6`$ within roundoff. Their initial global roots differ by $`0.6521700`$. At $`W=2,R=3`$ the evolved cache differs from the exact physical marginal by $`2.1\times10^{-16}`$, while the first-site marginal computed from the actual output differs by $`0.0216416`$. An independently propagated physical cache or tiny local fitting residual cannot certify the global amplitude. [Rank, marginal and amplitude measurements](results/local_controls_01/measurements.jsonl)

## Sensitivity to the action on unseen directions

As a separate diagnostic, we retained the same fitted rotation while replacing the identity on the orthogonal complement of the joint training span by minus identity. Both completions have equal training residual to roundoff. Their actions on the globally occupied $`P_W`$ branch differ by a sign. The alternative can move the state when the prescribed physical channel has no effect on its exact trajectory.

At $`N=5`$, $`b_0=0`$, $`W=2`$, $`R=3`$, and $`p=0.6`$, the native full sweep changed the stationary amplitude by $`1.02\times10^{-15}`$ at its endpoint. The reflected completion changed the amplitude by $`0.6324555320`$ and had a maximum local residual of $`1.58\times10^{-16}`$. Designating all five sites as identities gave a $`5.68\times10^{-16}`$ difference from the reference for either completion. At $`N=6`$, $`b_0=0.4`$, the reflected completion changed the amplitude by up to $`0.6521700`$ on an inactive-site update, yet its endpoint error after the complete sweep was $`0.0192802`$. Its active-only error was $`0.6437493`$. Several parity sign flips can cancel before an endpoint check. In the native completion cohort the largest measured individual inactive-site change was $`1.02\times10^{-15}`$. The fixed-support unitary bound applies to active-only updates, and more generally to a proper union of gate supports. A uniform sweep can have full-chain support. [Per-site motion diagnostics](results/local_controls_01/measurements.jsonl)

## Independent dense checks and locally reversible control

Across 13 half-Petz checks on $`N=2,3,4,6,8`$, the largest difference between the stored MPS and an independently evaluated full-chain Kraus amplitude was $`2.58\times10^{-15}`$. Whole-chain windows at $`N=2,3,4,6`$ reproduced the exact root to $`2.73\times10^{-15}`$. A separate test uses a complex transverse field obtained by a diagonal phase rotation, compares all amplitude entries against dense evolution, and verifies the physical cache after three unequal intervals. [Validation tests](tests/test_local.py) · [Dense validation data](results/local_controls_01/measurements.jsonl)

The original cluster control starts from the controlled-$`Z`$ circuit applied to a product of plus states. Uniform $`Z`$ dephasing commutes with that circuit, so the analytic canonical root at the endpoint is the circuit conjugate of a product of one-site roots. A recovery channel on one noisy spin and its nearest neighbors can locally undo the incident controlled-$`Z`$ gates, restore the known one-site state, and reapply those gates. We applied two full sweeps of $`h=0.3`$ to $`N=8`$, with a stored bond cap of four, and compared all amplitude entries with the predecessor's analytic-root implementation.

| Native method | $`W`$ | $`R`$ | Raw global error at $`T=0.6`$ |
|---|---:|---:|---:|
| Half-Petz | 2 | 2 | 0.7711672 |
| Half-Petz | 3 | 3 | $`5.47\times10^{-15}`$ |
| Practical fit | 3 | 5 | 0.6279644 |
| Practical fit | 3 | 7 | $`8.32\times10^{-11}`$ |

All four outputs had bond four. The narrower settings show that support size and fitting data matter even on this control. The pure initial state lies outside the full-rank witness comparison. The unmodified predecessor half-Petz runner independently passed six-spin cluster and non-Clifford controlled-phase checks at five sweeps, with respective endpoint errors $`7.22\times10^{-15}`$ and $`1.69\times10^{-14}`$. The original practical-fit sanity runner also passed its channel and evolved-cache assertions. [Cluster measurements](results/local_controls_01/measurements.jsonl) · [Predecessor results](results/local_predecessors_01/)

The validation grid contains 46 fresh one-thread subprocesses, with 46 completed. Its timings include validation work and are excluded from the production cost comparison. The grid manifest records configurations, resource guards, environment and source hashes. To reproduce the local controls, choose a fresh output directory and run from the repository root.

```sh
python3 -m parity_local_lift_comparison.local_controls --output parity_local_lift_comparison/results/local_controls_replay
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s parity_local_lift_comparison/tests -p 'test_local.py' -v
```

[Validation grid and hashes](results/local_controls_01/manifest.json) · [Control worker](local_controls.py)
