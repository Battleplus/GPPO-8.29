from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_execution_launch_gate.py"


def _load_builder_module():
    specification = importlib.util.spec_from_file_location(
        "build_execution_launch_gate_for_test", SCRIPT
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load launch gate builder")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class LaunchGateBuilderTests(unittest.TestCase):
    def test_all_committed_pretraining_artifacts_pass_strict_validation(self) -> None:
        builder = _load_builder_module()
        validators = (
            builder._validate_dev_manifest,
            builder._validate_allocator_replay,
            builder._validate_graph_smoke,
            builder._validate_adapter_smoke,
            builder._validate_deferred_parity,
            builder._validate_training_contract_smoke,
            builder._validate_framework_smoke,
        )
        results = [validator() for validator in validators]
        self.assertTrue(
            all(result["status"] == "PASS" for result in results),
            results,
        )


if __name__ == "__main__":
    unittest.main()
