from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_dynamic_preemption_allocators import run


class AllocatorReplayTests(unittest.TestCase):
    def test_two_deterministic_allocators_replay_same_200_tapes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        tape_root = root / "experiments" / "dynamic_preemption" / "dev_v1" / "tapes"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            first = run(tape_root, output)
            first_bytes = output.read_bytes()
            second = run(tape_root, output)
            self.assertEqual(first_bytes, output.read_bytes())
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "PASS")
        self.assertEqual(first["tape_count"], 200)
        self.assertEqual(first["allocator_tape_runs"], 400)
        self.assertFalse(first["training_started"])
        self.assertFalse(first["model_effectiveness_evaluated"])
        self.assertTrue(all(item["decision_count"] == 280 for item in first["results"]))
        self.assertTrue(all(item["invariant_failures"] == 0 for item in first["results"]))


if __name__ == "__main__":
    unittest.main()
