"""Machine P0 gate contract tests.

These tests ensure the gate is based on executable checks and frozen hashes,
not hand-edited status fields.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def _load_gate_module():
    path = ROOT / "scripts" / "build_p0_gate.py"
    spec = importlib.util.spec_from_file_location("build_p0_gate_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class P0GateContractTests(unittest.TestCase):
    def test_required_suites_include_round_two_tests(self):
        gate = _load_gate_module()
        labels = {item[0] for item in gate.REQUIRED_TEST_SUITES}
        self.assertTrue({
            "event_runtime_integration", "confirmation_timelines", "concurrency_invariants",
        } <= labels)

    def test_gate_machine_checks_required_protocol(self):
        gate = _load_gate_module()
        invariants = gate.run_invariant_checks()
        required = {
            "burst_atomicity", "reward_invariant_four_modes", "confirmation_timeline_contracts",
            "concurrency_stale_rejection", "concurrency_exclusive_holder",
            "concurrency_duplicate_assignment", "concurrency_late_ack_resurrection",
            "concurrency_fencing_monotonicity", "single_snapshot_identity",
            "overlap_received_order", "unseen_isolation", "model_save_load_determinism",
        }
        self.assertTrue(required <= set(invariants))
        self.assertTrue(all("passed" in invariants[name] for name in required))

    def test_hash_attestation_contains_commit_and_tree(self):
        gate = _load_gate_module()
        hashes = gate.compute_hashes()
        self.assertRegex(hashes["git_commit_sha"], r"^[0-9a-f]{40}$|^UNAVAILABLE$")
        self.assertRegex(hashes["source_tree_hash"], r"^[0-9a-f]{64}$")
        self.assertEqual(len(hashes["protocol"]), 64)
        self.assertEqual(len(hashes["seed_manifest"]), 64)

    def test_config_contract_is_frozen(self):
        gate = _load_gate_module()
        result = gate.verify_config_contract()
        self.assertTrue(result["passed"], result)


if __name__ == "__main__":
    unittest.main()
