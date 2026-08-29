from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_execution_framework_rollout_smoke.py"


class FrameworkRolloutSmokeTests(unittest.TestCase):
    def test_smoke_is_deterministic_and_never_trains(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / "a.json", Path(temporary) / "b.json"]
            for path in paths:
                subprocess.run(
                    [sys.executable, str(SCRIPT), "--output", str(path)],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=True,
                    timeout=90,
                )
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
            report = json.loads(paths[0].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["run_count"], 4)
            self.assertEqual(report["torch_tensor_conversion"], "PASS")
            self.assertEqual(report["gym_reset_step_contract"], "PASS")
            self.assertFalse(report["pyg_health"]["required_for_current_gppo"])
            self.assertTrue(all(run["status"] == "PASS" for run in report["runs"]))
            self.assertTrue(all(run["mask_violations"] == 0 for run in report["runs"]))
            self.assertTrue(all(run["resource_conflicts"] == 0 for run in report["runs"]))
            for key in (
                "optimizer_created",
                "model_weights_loaded",
                "checkpoint_loaded",
                "checkpoint_written",
                "training_allowed",
                "training_started",
                "validation_started",
                "freeze_started",
                "test_started",
                "hidden_evaluation_started",
            ):
                self.assertFalse(report[key])
            self.assertEqual(report["optimizer_step_count"], 0)


if __name__ == "__main__":
    unittest.main()
