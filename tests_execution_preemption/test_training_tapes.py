from __future__ import annotations

import unittest

from execution_preemption.gym_env import ExecutionPreemptionGymEnv
from execution_preemption.tapes import build_development_bank, canonical_json_bytes
from execution_preemption.training_tapes import (
    TRAINING_BANK,
    TRAINING_NAMESPACE,
    build_training_tape,
    training_case_seed,
)


class TrainingTapeTests(unittest.TestCase):
    def test_training_stream_is_deterministic_scaled_and_namespace_isolated(self) -> None:
        development_seeds = {item["case_seed"] for item in build_development_bank()}
        seen: set[int] = set()
        for scale in (4, 8, 16):
            for episode in range(20):
                first = build_training_tape(
                    policy_seed=1101, uav_count=scale, episode_index=episode
                )
                second = build_training_tape(
                    policy_seed=1101, uav_count=scale, episode_index=episode
                )
                self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
                self.assertEqual(first["bank"], TRAINING_BANK)
                self.assertEqual(first["training_identity"]["namespace"], TRAINING_NAMESPACE)
                self.assertEqual(len(first["initial_state"]["uavs"]), scale)
                self.assertEqual(
                    len(first["initial_state"]["tasks"]),
                    scale * first["training_identity"]["tasks_per_uav"],
                )
                self.assertNotIn(first["case_seed"], development_seeds)
                self.assertNotIn(first["case_seed"], seen)
                seen.add(first["case_seed"])

    def test_seed_and_contract_drift_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "policy seed"):
            training_case_seed(9999, 4, 0)
        with self.assertRaisesRegex(ValueError, "UAV count"):
            training_case_seed(1101, 32, 0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            training_case_seed(1101, 4, -1)

    def test_reactive_and_rule_variants_share_actions_but_not_context(self) -> None:
        tape = build_training_tape(policy_seed=1101, uav_count=4, episode_index=3)
        reactive = ExecutionPreemptionGymEnv(tape, expose_rule_context=False)
        ruled = ExecutionPreemptionGymEnv(tape, expose_rule_context=True)
        _, reactive_info = reactive.reset(seed=tape["case_seed"])
        _, ruled_info = ruled.reset(seed=tape["case_seed"])
        self.assertEqual(
            reactive_info["flat_observation"].action_space,
            ruled_info["flat_observation"].action_space,
        )
        self.assertEqual(
            reactive_info["hetero_observation"].rule_context,
            (0.0,) * 16,
        )
        self.assertNotEqual(
            ruled_info["hetero_observation"].rule_context,
            (0.0,) * 16,
        )
        self.assertFalse(reactive_info["rule_context_exposed"])
        self.assertTrue(ruled_info["rule_context_exposed"])


if __name__ == "__main__":
    unittest.main()
