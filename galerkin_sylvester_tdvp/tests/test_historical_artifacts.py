"""File-preservation regressions pinned to the declared research base b337e8b."""
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class HistoricalArtifacts(unittest.TestCase):
    def test_published_controls(self):
        fixtures = {
            "cp_tdvp/tdvp_core.py": "2aa8cc3550ac96ca0a498f204deb4d470deefe330b4d00d130d040de4834b7db",
            "cp_tdvp/results/dense.csv": "7f4c1e49470729c4928854acf7fab6b0b198c5fc1b2605163796bcee4853f651",
            "scalable_cp_witness/n512_t0.1_s4_p0.12_a0.35_jensen_tdvp.csv": "0d4629d6f48cbbdba3299d8ed2879ac5ccfac87855168ef26b4331263d1dd4be",
            "factor_tdvp/results/measurements.csv": "5256591858d548d15a5b7435f52f129215d1b6a9be23ab946dc7394d7a41ffd0",
            "factor_tdvp_comparison/results/matched_accuracy.csv": "c0c420f9e0a692ca8573a9bba8883b97b6d1303d4a5a2cbedcb708ae73d44f7c",
            "factor_tdvp_accelerated/results/BENCHMARK_MANIFEST.md": "f0cfec4ca1031eaca78c9f0a12bb113161ded623e072aca2740a15c493534463",
        }
        for file, expected in fixtures.items():
            with self.subTest(file=file):
                self.assertEqual(hashlib.sha256((ROOT/file).read_bytes()).hexdigest(), expected)

    def test_recovered_actions_cohort(self):
        folder = ROOT/"factor_tdvp_accelerated/scaling_runs/actions_43745844867"
        # Flat lexicographic relative PATH (including slash), independent of OS traversal order.
        paths = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()
                       and not p.name.startswith("."))
        self.assertEqual(len(paths), 168)
        digest = hashlib.sha256()
        for relative in paths:
            digest.update(relative.encode()+b"\0")
            digest.update(hashlib.sha256((folder/relative).read_bytes()).digest())
        self.assertEqual(digest.hexdigest(), "aa75f8f3b013a7e32c8d17e8c84d0779cca2d4c89bb194a7a149b8dc650bd9ab")
