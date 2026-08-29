from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.run_dynamic_preemption_deferred_parity import run


ROOT = Path(__file__).resolve().parents[1]


class DeferredParityTests(unittest.TestCase):
    def test_two_allocators_match_direct_replay_on_all_200_tapes(self) -> None:
        tape_root = ROOT / "experiments" / "dynamic_preemption" / "dev_v1" / "tapes"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "parity.json"
            first = run(tape_root, output)
            first_bytes = output.read_bytes()
            second = run(tape_root, output)
            self.assertEqual(first_bytes, output.read_bytes())
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "PASS")
        self.assertEqual(first["allocator_tape_runs"], 400)
        self.assertFalse(first["training_started"])
        self.assertFalse(first["model_framework_loaded"])
        for row in first["results"]:
            self.assertEqual(row["decision_count"], 280)
            self.assertEqual(row["allocation_request_count"], 80)
            self.assertEqual(row["decision_parity_pass_count"], 200)
            self.assertEqual(row["state_sha256_parity_pass_count"], 200)


if __name__ == "__main__":
    unittest.main()

