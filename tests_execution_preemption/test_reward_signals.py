from __future__ import annotations

import copy
import unittest

from execution_preemption.models import TaskRuntime, TaskState, UAVRuntime
from execution_preemption.reward import HardSafetyViolation, compute_transition_reward
from execution_preemption.runtime import ExecutionRuntime
from execution_preemption.signals import derive_transition_signals


class RewardSignalDerivationTests(unittest.TestCase):
    @staticmethod
    def running() -> ExecutionRuntime:
        runtime = ExecutionRuntime()
        runtime.add_uav(UAVRuntime("U0", energy_ratio=0.9))
        runtime.add_uav(UAVRuntime("U1", energy_ratio=0.8))
        runtime.add_task(TaskRuntime(
            "T0", "SEARCH", 80, 20.0, progress=0.4, remaining_work=0.6
        ))
        runtime.assign_task("T0", "U0", at=0.0)
        return runtime

    def test_migration_loss_switch_and_load_are_derived(self) -> None:
        before = self.running()
        after = copy.deepcopy(before)
        after.migrate_task("T0", "U1", at=1.0, reason="test")
        signals = derive_transition_signals(before, after, now=1.0)
        self.assertAlmostEqual(signals.progress_loss, 0.04)
        self.assertAlmostEqual(signals.switch_time, 0.05)
        self.assertEqual(signals.load_gap, 1.0)
        self.assertEqual(signals.resource_conflicts, 0)
        self.assertEqual(signals.energy_safety_violations, 0)
        reward = compute_transition_reward(signals)
        self.assertAlmostEqual(reward.reward, -0.71)

    def test_progress_energy_vacancy_deadline_and_starvation_are_explicit(self) -> None:
        before = self.running()
        before.add_task(TaskRuntime("WAIT", "URGENT", 100, 0.5))
        after = copy.deepcopy(before)
        after.advance(
            1.0,
            now=1.0,
            work_rate_by_task={"T0": 0.1},
            energy_cost_by_uav={"U0": 0.1},
        )
        # Preserve an old queue timestamp so starvation exposure is observable.
        after.tasks["WAIT"].last_updated_at = 0.0
        signals = derive_transition_signals(before, after, now=101.0, delta_time=2.0)
        self.assertGreater(signals.weighted_progress_gain, 0.0)
        self.assertGreater(signals.weighted_vacancy_time, 0.0)
        self.assertEqual(signals.starvation_exposure, 1.0 / 1.8)
        self.assertEqual(signals.urgent_deadline_miss_rate, 1.0)
        self.assertAlmostEqual(signals.energy_consumed, 0.05)

    def test_energy_safety_violation_stops_reward(self) -> None:
        before = self.running()
        after = copy.deepcopy(before)
        after.uavs["U0"].energy_ratio = 0.1
        signals = derive_transition_signals(before, after, now=1.0)
        self.assertEqual(signals.energy_safety_violations, 1)
        with self.assertRaises(HardSafetyViolation):
            compute_transition_reward(signals)

    def test_invalid_time_or_distance_fails_closed(self) -> None:
        runtime = self.running()
        with self.assertRaisesRegex(ValueError, "delta_time"):
            derive_transition_signals(runtime, runtime, now=1.0, delta_time=-1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            derive_transition_signals(
                runtime, runtime, now=1.0, normalized_distance=float("nan")
            )


if __name__ == "__main__":
    unittest.main()
