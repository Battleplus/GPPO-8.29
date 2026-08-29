from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_execution_policy_adapter_smoke.py"


class PolicyAdapterSmokeTests(unittest.TestCase):
    def test_smoke_is_deterministic_and_pretraining_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / "a.json", Path(temporary) / "b.json"]
            for path in paths:
                subprocess.run(
                    [sys.executable, str(SCRIPT), "--output", str(path)],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
            report = json.loads(paths[0].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["uav_counts"], [4, 8, 16, 32])
            self.assertEqual(len(report["scales"]), 4)
            self.assertTrue(all(row["status"] == "PASS" for row in report["scales"]))
            self.assertTrue(all(row["shared_action_space"] for row in report["scales"]))
            self.assertTrue(all(row["flat_observation_dimension"] == 37976
                                for row in report["scales"]))
            self.assertTrue(all(row["action_capacity"] == 3073 for row in report["scales"]))
            for key in (
                "model_framework_loaded",
                "checkpoint_loaded",
                "training_allowed",
                "training_started",
                "validation_started",
                "freeze_started",
                "test_started",
                "hidden_evaluation_started",
            ):
                self.assertFalse(report[key])


if __name__ == "__main__":
    unittest.main()

