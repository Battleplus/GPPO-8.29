"""Event runtime integration tests.

Covers:
- same seed deterministic replay
- different seed changes tape
- single snapshot identity
- bridge truth→observation→confirmation pipeline end-to-end
- bridge concurrency counters
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from event_runtime.events import EventType, TruthEvent

from random_event.environment import RandomEventAllocationEnv
from random_event.events import EventTape, RandomEvent, RandomEventType
from random_event.graph import build_graph_state


class DeterministicReplayTests(unittest.TestCase):
    def test_same_seed_produces_identical_tape(self):
        env_a = RandomEventAllocationEnv(initial_seed=42, event_seed=42001, mode="sequential", events_per_episode=5)
        env_b = RandomEventAllocationEnv(initial_seed=42, event_seed=42001, mode="sequential", events_per_episode=5)
        ga, _ = env_a.reset(seed=42)
        gb, _ = env_b.reset(seed=42)
        self.assertEqual(len(env_a.event_tape.events), len(env_b.event_tape.events))
        for ea, eb in zip(env_a.event_tape.events, env_b.event_tape.events):
            self.assertEqual(ea.event_id, eb.event_id)
            self.assertEqual(ea.event_type, eb.event_type)
            self.assertAlmostEqual(ea.occurred_at, eb.occurred_at)

    def test_different_seed_changes_tape(self):
        env_a = RandomEventAllocationEnv(initial_seed=42, event_seed=42001, mode="sequential", events_per_episode=5)
        env_b = RandomEventAllocationEnv(initial_seed=99, event_seed=42002, mode="sequential", events_per_episode=5)
        env_a.reset(seed=42)
        env_b.reset(seed=99)
        events_a = [(e.event_type, e.affected_uavs, e.affected_targets, e.affected_regions) for e in env_a.event_tape.events]
        events_b = [(e.event_type, e.affected_uavs, e.affected_targets, e.affected_regions) for e in env_b.event_tape.events]
        # Different seeds should produce different tapes (extremely high probability)
        self.assertNotEqual(events_a, events_b)


class SnapshotIdentityTests(unittest.TestCase):
    def test_same_seed_same_snapshot(self):
        env = RandomEventAllocationEnv(initial_seed=42, event_seed=42001, mode="single", events_per_episode=1)
        s1 = env.snapshot()
        env2 = RandomEventAllocationEnv(initial_seed=42, event_seed=42001, mode="single", events_per_episode=1)
        s2 = env2.snapshot()
        # Entity positions etc should match
        self.assertEqual(s1["random_event"]["graph_version"], s2["random_event"]["graph_version"])
        self.assertEqual(s1["random_event"]["pending_regions"], s2["random_event"]["pending_regions"])


class BridgeEndToEndTests(unittest.TestCase):
    def test_region_vacancy_confirms_and_modifies_env(self):
        env = RandomEventAllocationEnv(
            initial_seed=42, event_seed=42001, mode="single", events_per_episode=1,
            event_tape=EventTape(
                initial_seed=42, event_seed=42001, mode="single",
                events=(RandomEvent(
                    event_id="E0000", event_type=RandomEventType.REGION_VACANCY,
                    occurred_at=0.0, observed_at=0.0, source_event="test",
                    affected_uavs=(0,), affected_regions=(0,), severity=0.5,
                    event_seed=42001, state_version=0,
                ),),
            ),
            max_decisions=10,
        )
        graph, info = env.reset()
        self.assertEqual(len(env.event_records), 1)
        self.assertIn("E0000", env.event_queue)
        self.assertEqual(info["graph_version"], 1)
        self.assertTrue(env.regions[0].need_reassign)

    def test_uav_damage_confirms_and_kills_uav(self):
        env = RandomEventAllocationEnv(
            initial_seed=42, event_seed=42001, mode="single", events_per_episode=1,
            event_tape=EventTape(
                initial_seed=42, event_seed=42001, mode="single",
                events=(RandomEvent(
                    event_id="E0000", event_type=RandomEventType.UAV_DAMAGE,
                    occurred_at=0.0, observed_at=0.0, source_event="test",
                    affected_uavs=(0,), affected_regions=(), severity=0.5,
                    event_seed=42001, state_version=0,
                ),),
            ),
            max_decisions=10,
        )
        env.reset()
        self.assertFalse(env.uavs[0].alive)

    def test_bridge_observation_count_increments(self):
        env = RandomEventAllocationEnv(
            initial_seed=42, event_seed=42001, mode="single", events_per_episode=1,
            event_tape=EventTape(
                initial_seed=42, event_seed=42001, mode="single",
                events=(RandomEvent(
                    event_id="E0000", event_type=RandomEventType.REGION_VACANCY,
                    occurred_at=0.0, observed_at=0.0, source_event="test",
                    affected_uavs=(0,), affected_regions=(0,), severity=0.5,
                    event_seed=42001, state_version=0,
                ),),
            ),
            max_decisions=10,
        )
        env.reset()
        obs_count = env.runtime_bridge.get_observation_count()
        truth_count = env.runtime_bridge.get_truth_event_count()
        self.assertGreaterEqual(obs_count, 1)
        self.assertEqual(truth_count, 1)


class BridgeConcurrencyCountersTests(unittest.TestCase):
    def test_counters_initialized(self):
        env = RandomEventAllocationEnv(initial_seed=42, event_seed=42001, events_per_episode=1)
        env.reset()
        snapshot = env.runtime_bridge.snapshot_concurrency(0.0)
        self.assertIn("stale_rejected", snapshot)
        self.assertIn("late_ack_resurrections", snapshot)
        self.assertIn("exclusive_holder_violations", snapshot)
        self.assertEqual(snapshot["late_ack_resurrections"], 0)

    def test_ack_resurrection_count(self):
        env = RandomEventAllocationEnv(initial_seed=42, event_seed=42001, events_per_episode=1)
        env.reset()
        bridge = env.runtime_bridge
        bridge.adapter.concurrency.create_command("c1", "0", "0", graph_version=0, action_version=0, now=0.0)
        cmd = bridge.adapter.concurrency.commands["c1"]
        cmd.status = CommandStatus.REVOKED
        ok = bridge.receive_ack("c1", "0", cmd.fencing_token, 1.0)
        self.assertFalse(ok)
        self.assertEqual(bridge._cc["late_ack_resurrections"], 1)


from event_runtime.concurrency import CommandStatus


if __name__ == "__main__":
    unittest.main()
