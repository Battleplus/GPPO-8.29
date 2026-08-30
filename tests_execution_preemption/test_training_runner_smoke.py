from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_execution_training_runner_smoke.py"


class TrainingRunnerSmokeTests(unittest.TestCase):
    def test_machine_smoke_covers_all_four_methods_without_formal_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "smoke.json"
            subprocess.run(
                [sys.executable, str(SCRIPT), "--output", str(output)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["learned_method_count"], 4)
            self.assertTrue(report["real_optimizer_updates"])
            self.assertFalse(report["formal_training_allowed"])
            self.assertFalse(report["formal_training_started"])
            self.assertFalse(report["validation_started"])
            self.assertFalse(report["hidden_evaluation_started"])
            self.assertTrue(all(item["status"] == "PASS" for item in report["methods"]))
            self.assertTrue(all(item["optimizer_step_count"] == 2 for item in report["methods"]))
            self.assertTrue(all(item["checkpoint_file_sha_verified"] for item in report["methods"]))
            self.assertTrue(all(item["same_seed_state_determinism"] for item in report["methods"]))


if __name__ == "__main__":
    unittest.main()
