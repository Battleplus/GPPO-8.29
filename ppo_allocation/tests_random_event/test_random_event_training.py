"""Small end-to-end PPO training test kept separate from fast contracts."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from random_event.environment import RandomEventAllocationEnv
from random_event.events import EventTape, RandomEvent, RandomEventType
from random_event.trainer import PPOConfig, PPOTrainer


def training_env() -> RandomEventAllocationEnv:
    shock = RandomEvent(
        event_id="E0000",
        event_type=RandomEventType.REGION_VACANCY,
        occurred_at=0.0,
        observed_at=0.0,
        source_event="training_smoke",
        affected_uavs=(0,),
        affected_regions=(0,),
        severity=0.5,
        event_seed=91,
        state_version=0,
    )
    tape = EventTape(
        initial_seed=23,
        event_seed=91,
        mode="single",
        events=(shock,),
    )
    return RandomEventAllocationEnv(
        initial_seed=23,
        event_seed=91,
        event_tape=tape,
        events_per_episode=1,
        max_decisions=10,
    )


class PPOTrainingContractTests(unittest.TestCase):
    def test_minimal_adaptive_ppo_train_save_and_load(self):
        config = PPOConfig(
            rollout_steps=8,
            update_epochs=1,
            minibatch_size=4,
            seed=13,
            device="cpu",
        )
        trainer = PPOTrainer(training_env(), variant="GPPO-Adaptive", config=config)
        history = trainer.train(total_timesteps=8)
        self.assertEqual(trainer.total_steps, 8)
        self.assertEqual(trainer.update_count, 1)
        self.assertEqual(len(history), 1)
        for key in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "pre_mask_invalid_probability",
            "grad_norm",
        ):
            self.assertTrue(math.isfinite(float(history[0][key])), key)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "ppo_smoke.pt"
            trainer.save(checkpoint, extra={"test": "minimal"})
            restored, metadata = PPOTrainer.load(checkpoint, env=training_env(), device="cpu")
            self.assertEqual(restored.variant, "GPPO-Adaptive")
            self.assertEqual(restored.total_steps, 8)
            self.assertEqual(restored.update_count, 1)
            self.assertEqual(metadata["test"], "minimal")
            self.assertEqual(restored.model.config.adaptive_gate, True)


if __name__ == "__main__":
    unittest.main()
