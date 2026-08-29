from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.build_execution_graph_smoke import build


class GraphSmokeTests(unittest.TestCase):
    def test_scale_smoke_is_deterministic_and_training_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.json"
            first = build(path)
            first_bytes = path.read_bytes()
            second = build(path)
            self.assertEqual(first, second)
            self.assertEqual(first_bytes, path.read_bytes())
        self.assertEqual(first["status"], "PASS")
        self.assertEqual(first["uav_counts"], [4, 8, 16, 32])
        self.assertFalse(first["training_started"])
        self.assertFalse(first["old_checkpoint_compatible"])
        self.assertTrue(first["requires_new_training_contract"])
        for scale in first["scales"]:
            self.assertEqual(
                scale["action_candidate_count"],
                scale["expected_action_candidate_count"],
            )
            self.assertTrue(scale["noop_present"])


if __name__ == "__main__":
    unittest.main()
