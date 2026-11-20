# Publication and validation

The writable fork's [draft PR #5](https://github.com/xiaoyanghuangphys-create/canonical-purification-dynamics/pull/5) targets `research/factor-vs-witness-scaling`. The published [report snapshot](https://github.com/xiaoyanghuangphys-create/canonical-purification-dynamics/blob/a2fd54130cad289e324123422c7d812729cabba0/galerkin_sylvester_tdvp/report.md) was inspected in GitHub's Markdown Preview. All five display equations rendered as mathematical symbols. The preview showed the Sylvester map, tangent metric/right-hand side, both sums with their limits and bond subscripts, and the constrained Galerkin equation with its multiplier. The comparison tables rendered as tables and the report exposed the cost figure.

The prepublication and analysis checks covered the following items.

- The origin fork has research base `b337e8b7fb86fc3cccec64e13e2ddb35cc14c4e1`. Its source tree is identical to main at inspection. The upstream repository was read solely for context. The published diff from the research base contains only additions under `galerkin_sylvester_tdvp/`.
- In the definitive 180-worker comparison and 51-worker matched interacting study, worker-level numerical source hashes are consistent. Missing or nonfinite measured errors, failed or incomplete endpoints, unexpected repeat IDs, and different source/settings manifests are ineligible for the repeated frontier. Regenerating the tables and plot from the archived worker measurements gave the published outputs.
- Raw main-driver archives were compared byte-for-byte with the worker files before recording their SHA-256 checksums. The supplemental oracle and integration notes identify their own definitive runs and record the pilot data separately.
- The two published scientific figures were inspected for readable axes, legends and labels. All research-module and predecessor tests passed. See [new test log](method_b_tests.log) and [predecessor test log](predecessor_tests.log).
