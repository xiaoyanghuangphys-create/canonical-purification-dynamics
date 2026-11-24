# Script regressions and local equation rendering

All three predecessor scripts exited with status 0. Each ran from the repository root using Python 3.9.6 with `PYTHONDONTWRITEBYTECODE=1`. The OpenBLAS, OpenMP, MKL, Apple Accelerate and NumExpr thread settings were all 1. Only the `--out` or `--outdir` destination changed. All predecessor output CSVs and captured stdout/stderr remain in the corresponding directories.

```sh
python3 scalable_cp_witness/code/validate_scalable_cp.py --out tangent_newton_witness/results/validation_01/validate_scalable_cp
python3 scalable_cp_witness/code/cp_accuracy_refinement.py check --out tangent_newton_witness/results/validation_01/cp_accuracy_check
python3 automatic_rank_opening/test_automatic_rank_opening.py --outdir tangent_newton_witness/results/validation_01/automatic_rank
```

- The scalable witness suite wrote seven dense certificate cases. For the negative root at five spins, the density residual was about $10^{-15}$, the canonical-root error was 2 and the witness bound was 4. The additional singular rank-opening case had error 0.020001 and gap-free bound 0.056569. [Dense output](validate_scalable_cp/dense_checks.csv)
- The accuracy-refinement script checked complex Hermitian rounding, real tensor truncation and its independent canonical-purification audit. [Captured output](cp_accuracy_check/run.log)
- The support suite asserted all nine detection classifications and wrote six evolution comparisons with dense physical propagation. It also recorded six opening-tail scaling samples. [Detection](automatic_rank/detection.csv) · [Evolution](automatic_rank/evolution.csv) · [Tail scaling](automatic_rank/opening_scaling.csv)

The initial local render check examined the file digests recorded in `math_links_01.json`. KaTeX 0.16.21 produced MathML for all 183 inline expressions and four display equations. All 44 relative links/images resolved in the workspace. The extractor excludes Markdown code and checks for unbalanced math delimiters. GitHub uses MathJax. The local check establishes TeX parser acceptance and file existence; final GitHub rendering is a separate check.

To check fresh files, select a new JSON output path. The validator uses a locally installed KaTeX package, or the VS Code markdown-math notebook extension on macOS.

```sh
node tangent_newton_witness/results/validation_01/check_report.mjs /tmp/witness-math-replay.json tangent_newton_witness/report.md tangent_newton_witness/conditioning.md tangent_newton_witness/paired_methods.md tangent_newton_witness/frozen_report.md
```

The final main report produced 130 inline and four display math elements in GitHub's GFM renderer. The first hosted page inspection exposed seven unparsed inline expressions. Replacing ambiguous norm and star notation resolved every raw dollar span in the render check. The final local check also covers the README and handoff. A separate clean Git clone passed all 40 new tests, verified all 500 paired archives, and completed two fresh endpoint smoke runs. [GitHub rendering record](github_render.json) · [Final local parsing and links](final_render_and_links.json) · [Clean clone](clean_checkout.json)
