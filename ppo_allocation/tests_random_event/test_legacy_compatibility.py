"""Compatibility checks for the repository-pinned legacy fixture and C++ bridge.

The fixture is not a formal experiment result. It is a deterministic compatibility
artifact used only to exercise LegacyMLPPPOPolicy and the machine-readable bridge.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from random_event.environment import RandomEventAllocationEnv
from random_event.legacy_adapter import LegacyMLPPPOPolicy


PPO_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PPO_DIR.parent
MODEL = Path(__file__).resolve().parent / "fixtures/legacy_maskable_ppo_checkpoint.zip"
EXPECTED_FIXTURE_SHA256 = "5a9be7153d33532ce99c61f13c8151549cc6cc919ec75fad150d05bc78dec5da"


class LegacyCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODEL.is_file(), f"missing repository fixture: {MODEL}")
        import hashlib
        digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_FIXTURE_SHA256)

    def test_pinned_legacy_fixture_selects_legal_edge(self):
        env = RandomEventAllocationEnv(initial_seed=1, event_seed=2, events_per_episode=3)
        graph, _ = env.reset()
        policy = LegacyMLPPPOPolicy(MODEL)
        action = policy.select_action(env, graph)
        self.assertTrue(bool(graph.action_mask[action]))
        _, reward, _, _, info = env.step(action)
        self.assertIsInstance(reward, float)
        self.assertFalse(info["invalid_action"])

    def test_cpp_bridge_returns_machine_readable_success_json(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            request_path = temp / "request.json"
            output_path = temp / "output.json"
            request_path.write_text(
                json.dumps(
                    {
                        "model_path": str(MODEL),
                        "preallocation_path": str(PPO_DIR / "scenarios/output_template.json"),
                        "event": {"event_type": "UAV_DAMAGE", "uav_id": 1},
                        "output_path": str(output_path),
                        "deterministic": True,
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [sys.executable, str(PPO_DIR / "cpp_bridge.py"), "--request-file", str(request_path)],
                cwd=REPO_DIR,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            response = json.loads(completed.stdout)
            self.assertTrue(response["success"])
            self.assertTrue(output_path.exists())
            self.assertIn("region_assignments", response["result"])
            self.assertIn("uav_tasks", response["result"])


if __name__ == "__main__":
    unittest.main()
