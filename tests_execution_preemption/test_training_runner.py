from __future__ import annotations

import json
from pathlib import Path
from unittest import mock
import tempfile
import unittest

import torch

from execution_preemption.training import (
    TrainingRunConfig,
    formal_run_relative_path,
    planned_formal_runs,
    train_run,
    verify_checkpoint,
    verify_training_run,
)


class TrainingRunnerTests(unittest.TestCase):
    def _config(self, method: str, *, steps: int = 2) -> TrainingRunConfig:
        return TrainingRunConfig(
            method_id=method,
            policy_seed=1101,
            uav_count=4,
            accepted_decision_steps=steps,
            checkpoint_steps=tuple(range(1, steps + 1)),
            rollout_steps=1,
            update_epochs=1,
            hidden_dim=8,
            relation_layers=1,
            formal=False,
        )

    def test_tiny_ppo_and_gppo_runs_update_and_seal_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for method in (
                "ppo_mlp_reactive_v1",
                "gppo_adaptive_rule_arbiter_v1",
            ):
                output = root / method
                report = train_run(self._config(method), output)
                self.assertEqual(report["status"], "PASS")
                self.assertEqual(report["accepted_decision_steps"], 2)
                self.assertEqual(report["optimizer_step_count"], 2)
                self.assertEqual(report["checkpoint_steps"], [1, 2])
                self.assertFalse(report["formal"])
                self.assertFalse(report["validation_started"])
                self.assertFalse(report["hidden_evaluation_started"])
                progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
                self.assertEqual(progress["status"], "COMPLETE")
                inventory = json.loads(
                    (output / "sha256_inventory.json").read_text(encoding="utf-8")
                )
                self.assertEqual(inventory["status"], "PASS")
                self.assertEqual(len(inventory["files"]), 5)
                sealed = verify_training_run(output, expected_config=self._config(method))
                self.assertEqual(sealed["status"], "PASS")
                self.assertEqual(sealed["checkpoint_count"], 2)
                for checkpoint in report["checkpoints"]:
                    verified = verify_checkpoint(
                        output / "checkpoints" / checkpoint["path"],
                        expected_file_sha256=checkpoint["sha256"],
                        expected_method_id=method,
                        expected_policy_seed=1101,
                        expected_uav_count=4,
                        expected_step=checkpoint["step"],
                    )
                    self.assertEqual(verified["status"], "PASS")
                    self.assertEqual(
                        verified["model_state_sha256"],
                        checkpoint["model_state_sha256"],
                    )
                    self.assertEqual(
                        verified["rng_state_sha256"],
                        checkpoint["rng_state_sha256"],
                    )

    def test_same_seed_has_same_model_and_rng_state_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config("ppo_mlp_reactive_v1")
            first = train_run(config, root / "first")
            second = train_run(config, root / "second")
            self.assertEqual(
                [item["model_state_sha256"] for item in first["checkpoints"]],
                [item["model_state_sha256"] for item in second["checkpoints"]],
            )
            self.assertEqual(
                [item["rng_state_sha256"] for item in first["checkpoints"]],
                [item["rng_state_sha256"] for item in second["checkpoints"]],
            )

    def test_formal_gate_runs_before_output_or_optimizer(self) -> None:
        config = TrainingRunConfig(
            method_id="ppo_mlp_reactive_v1",
            policy_seed=1101,
            uav_count=4,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist"
            with mock.patch(
                "execution_preemption.training._check_execution_launch_gate",
                side_effect=RuntimeError("gate red"),
            ), mock.patch.object(torch.optim, "Adam") as optimizer:
                with self.assertRaisesRegex(RuntimeError, "gate red"):
                    train_run(config, output)
            self.assertFalse(output.exists())
            optimizer.assert_not_called()

    def test_formal_campaign_has_36_unique_worker_paths(self) -> None:
        runs = planned_formal_runs()
        paths = [
            formal_run_relative_path(item.method_id, item.policy_seed, item.uav_count)
            for item in runs
        ]
        self.assertEqual(len(runs), 36)
        self.assertEqual(len(set(paths)), 36)
        self.assertEqual({item.uav_count for item in runs}, {4, 8, 16})
        self.assertEqual({item.policy_seed for item in runs}, {1101, 2202, 3303})
        self.assertTrue(all(item.accepted_decision_steps == 50_000 for item in runs))
        self.assertTrue(all(item.checkpoint_steps == (25_000, 50_000) for item in runs))

    def test_output_reuse_and_contract_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            train_run(self._config("ppo_mlp_reactive_v1", steps=1), output)
            with self.assertRaises(FileExistsError):
                train_run(self._config("ppo_mlp_reactive_v1", steps=1), output)
            progress = output / "progress.json"
            progress.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not complete"):
                verify_training_run(
                    output,
                    expected_config=self._config("ppo_mlp_reactive_v1", steps=1),
                )
        with self.assertRaisesRegex(ValueError, "budget drift"):
            TrainingRunConfig(
                method_id="ppo_mlp_reactive_v1",
                policy_seed=1101,
                uav_count=4,
                accepted_decision_steps=49_999,
                checkpoint_steps=(25_000, 49_999),
            ).validate()


if __name__ == "__main__":
    unittest.main()
