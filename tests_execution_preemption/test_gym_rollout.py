from __future__ import annotations

import json
import math
from pathlib import Path
import unittest

import numpy as np

from execution_preemption.adapter import AdapterValidationError
from execution_preemption.framework import flat_to_torch, hetero_to_torch
from execution_preemption.gym_env import ExecutionPreemptionGymEnv
from execution_preemption.policy_models import ExecutionGPPOAdaptive, ExecutionPPOMLP


ROOT = Path(__file__).resolve().parents[1]
TAPES = ROOT / "experiments" / "dynamic_preemption" / "dev_v1" / "tapes"


def load_tape(scenario: str, index: int = 0) -> dict:
    path = next((TAPES / scenario).glob(f"{scenario}-{index:02d}-*.json"))
    return json.loads(path.read_text(encoding="utf-8"))


class GymRolloutTests(unittest.TestCase):
    def test_one_request_tape_completes_with_reward_and_metrics(self) -> None:
        tape = load_tape("execution_uav_destroyed")
        env = ExecutionPreemptionGymEnv(tape, allocator_id="gym_test")
        observation, info = env.reset(seed=tape["case_seed"])
        self.assertEqual(observation.shape, (37976,))
        self.assertEqual(observation.dtype, np.float32)
        mask = env.action_masks()
        self.assertFalse(mask[0])
        action = int(np.flatnonzero(mask)[0])
        next_observation, reward, terminated, truncated, final = env.step_with_metadata(
            action, inference_latency_ms=1.25
        )
        self.assertEqual(next_observation.shape, (37976,))
        self.assertTrue(math.isfinite(reward))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(final["training_started"])
        self.assertEqual(final["policy_action_count"], 1)
        metrics = final["episode_metrics"]
        self.assertEqual(metrics["accepted_decision_count"], 1)
        self.assertEqual(metrics["p0_handling_rate"], 1.0)
        self.assertEqual(metrics["normal_task_recovery_rate"], 1.0)
        self.assertEqual(metrics["inference_latency_mean_ms"], 1.25)
        env.runtime.validate_invariants()

    def test_two_actions_in_one_batch_keep_live_state_unchanged_until_final(self) -> None:
        tape = load_tape("simultaneous_p1")
        env = ExecutionPreemptionGymEnv(tape, allocator_id="batch_test")
        _, info = env.reset(seed=tape["case_seed"])
        live_before = info["live_runtime_sha256"]
        first_action = int(np.flatnonzero(env.action_masks())[0])
        _, first_reward, terminated, truncated, middle = env.step(first_action)
        self.assertEqual(first_reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(middle["reward_deferred"])
        self.assertEqual(middle["live_runtime_sha256"], live_before)
        self.assertEqual(middle["policy_action_count"], 1)

        second_action = int(np.flatnonzero(env.action_masks())[0])
        _, reward, terminated, truncated, final = env.step(second_action)
        self.assertTrue(math.isfinite(reward))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertNotEqual(final["live_runtime_sha256"], live_before)
        self.assertEqual(env.runtime.graph_version, 1)
        self.assertEqual(final["policy_action_count"], 2)
        self.assertEqual(final["episode_metrics"]["accepted_decision_count"], 2)
        self.assertEqual(sum(task.assigned_uav is not None for task in env.runtime.tasks.values()), 2)

    def test_masked_action_fails_without_live_mutation(self) -> None:
        tape = load_tape("execution_uav_destroyed")
        env = ExecutionPreemptionGymEnv(tape)
        _, info = env.reset(seed=tape["case_seed"])
        before = info["live_runtime_sha256"]
        self.assertFalse(env.action_masks()[0])
        with self.assertRaisesRegex(AdapterValidationError, "masked"):
            env.step(0)
        self.assertEqual(env.runtime.state_sha256(), before)

    def test_ppo_and_gppo_forward_actions_submit_through_same_env_contract(self) -> None:
        tape = load_tape("execution_uav_destroyed")
        for policy_name in ("ppo", "gppo"):
            with self.subTest(policy=policy_name):
                env = ExecutionPreemptionGymEnv(tape, allocator_id=policy_name)
                _, info = env.reset(seed=tape["case_seed"])
                if policy_name == "ppo":
                    import torch
                    torch.manual_seed(5)
                    model = ExecutionPPOMLP(hidden_dim=8).eval()
                    action, *_ = model.act(flat_to_torch(info["flat_observation"]))
                else:
                    import torch
                    torch.manual_seed(5)
                    model = ExecutionGPPOAdaptive(hidden_dim=8, layers=1).eval()
                    action, *_ = model.act(hetero_to_torch(info["hetero_observation"]))
                self.assertTrue(env.action_masks()[action])
                _, reward, terminated, _, final = env.step(action)
                self.assertTrue(terminated)
                self.assertTrue(math.isfinite(reward))
                self.assertFalse(final["training_started"])

    def test_reset_and_rollout_are_deterministic(self) -> None:
        tape = load_tape("execution_uav_destroyed")
        outputs = []
        for _ in range(2):
            env = ExecutionPreemptionGymEnv(tape, allocator_id="deterministic")
            observation, info = env.reset(seed=tape["case_seed"])
            action = int(np.flatnonzero(env.action_masks())[0])
            _, reward, terminated, truncated, final = env.step(action)
            outputs.append((
                observation.tobytes(), action, reward, terminated, truncated,
                final["live_runtime_sha256"], final["episode_metrics"],
            ))
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
