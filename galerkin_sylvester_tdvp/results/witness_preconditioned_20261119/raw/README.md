# Raw witness workers

Each ZIP groups all enrolled A and B processes for one physical family and spin count. Member paths preserve the original process directories. The manifest records SHA-256 checksums for each ZIP, each member and the two top-level measurement indexes. Timeout, resource and numerical failures retain their original records.

From the repository root, set `RUN` to the directory containing this `raw` folder. Verify the archives, then expand a family/size into a fresh destination.

```sh
python3 -m galerkin_sylvester_tdvp.witness_pair_archive verify "$RUN"
unzip "$RUN/raw/D08.zip" -d /tmp/new-witness-workers
```
