from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_execution_training_contract.py"


class TrainingContractSmokeTests(unittest.TestCase):
    def test_script_is_deterministic_and_never_enables_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_a = Path(temporary) / "a.json"
            output_b = Path(temporary) / "b.json"
            for output in (output_a, output_b):
                subprocess.run(
                    [sys.executable, str(SCRIPT), "--output", str(output)],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            self.assertEqual(output_a.read_bytes(), output_b.read_bytes())
            report = json.loads(output_a.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(
                report["classification"],
                "training_precondition_contract_smoke_not_model_evidence",
            )
            self.assertFalse(report["training_allowed"])
            self.assertFalse(report["training_started"])
            self.assertFalse(report["validation_started"])
            self.assertFalse(report["freeze_started"])
            self.assertFalse(report["test_started"])
            self.assertFalse(report["hidden_evaluation_started"])
            self.assertFalse(report["checkpoint_selection"])
            self.assertFalse(report["legacy_checkpoint_compatible"])
            self.assertEqual(report["contract"]["path"], "configs/execution_training_contract_v1.json")
            self.assertEqual(report["contract"]["learned_run_count"], 36)
            self.assertEqual(report["contract"]["checkpoint_count"], 72)
            self.assertEqual(report["acceptance_smoke"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
