from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.build_execution_baseline_smoke import build_report, write_report


class FrozenBaselineReplaySmokeTests(unittest.TestCase):
    def test_three_baselines_replay_all_200_dev_tapes_deterministically(self) -> None:
        first = build_report()
        second = build_report()
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "PASS")
        self.assertEqual(first["tape_count"], 200)
        self.assertEqual(first["allocator_count"], 3)
        self.assertEqual(first["allocator_tape_runs"], 600)
        self.assertFalse(first["training_started"])
        self.assertFalse(first["model_effectiveness_evaluated"])
        self.assertTrue(all(item["decision_count"] == 280 for item in first["results"]))
        self.assertTrue(all(item["invariant_failures"] == 0 for item in first["results"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.json"
            write_report(path, first)
            first_bytes = path.read_bytes()
            write_report(path, second)
            self.assertEqual(first_bytes, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
