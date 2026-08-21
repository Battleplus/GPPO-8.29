"""Round-2 confirmation timeline and isolation tests.

Covers:
- truth/belief isolation (unconfirmed observation does not change mask/graph)
- 3-of-5 discovery confirmation
- dual-path destruction (authoritative vs ordinary strong)
- heartbeat probe flow (SUSPECTED → PROBE → timeout/confirm)
- healthy telemetry false alarm
- duplicate doesn't increase evidence count
- unconfirmed event doesn't change graph_version
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# event_runtime is at the repository root level
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from config import NO_TARGET, NO_UAV, TaskType
from event_runtime.adapter import EventRuntimeAdapter
from event_runtime.events import ConfirmationStatus, EventType, TruthEvent
from event_runtime.observation import Observation
from event_runtime.state_machine import ConfirmationStateMachine

from random_event.environment import RandomEventAllocationEnv
from random_event.events import EventTape, RandomEvent, RandomEventType
from random_event.graph import build_graph_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs(
    event_id: str = "E0001",
    event_type: EventType = EventType.TARGET_DISCOVERED,
    source_id: str = "src-0",
    signal_type: str = "SENSOR_DETECTION",
    positive: bool = True,
    confidence: float = 0.95,
    emitted_at: float = 1.0,
    received_at: float = 1.5,
    occurred_at: float = 0.5,
    affected_uavs: tuple[str, ...] = ("0",),
    affected_regions: tuple[str, ...] = (),
    affected_targets: tuple[str, ...] = ("0",),
    duplicate_of: str | None = None,
    is_duplicate: bool = False,
    sequence: int = 1,
) -> Observation:
    return Observation(
        observation_id=f"obs-{event_id}-{source_id}-{sequence}",
        event_id=event_id,
        event_type=event_type,
        source_event="test",
        source_id=source_id,
        source_type="sensor",
        signal_type=signal_type,
        sequence=sequence,
        confidence=confidence,
        positive=positive,
        emitted_at=emitted_at,
        received_at=received_at,
        occurred_at=occurred_at,
        affected_uavs=affected_uavs,
        affected_regions=affected_regions,
        affected_targets=affected_targets,
        duplicate_of=duplicate_of,
        is_duplicate=is_duplicate,
    )


def _truth(
    event_id: str = "E0001",
    event_type: EventType = EventType.TARGET_DISCOVERED,
    affected_uavs: tuple[str, ...] = (),
    affected_targets: tuple[str, ...] = ("0",),
    affected_regions: tuple[str, ...] = (),
) -> TruthEvent:
    return TruthEvent(
        event_id=event_id,
        event_type=event_type,
        source_event="test",
        affected_uavs=affected_uavs,
        affected_targets=affected_targets,
        affected_regions=affected_regions,
        occurred_at=0.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TruthBeliefIsolationTests(unittest.TestCase):
    """Audit item 1: truth event changes True State but NOT belief/mask/graph."""

    def test_unconfirmed_observation_does_not_change_env(self):
        """Feed 2 of 5 discovery observations (not enough to confirm) → env unchanged."""
        env = RandomEventAllocationEnv(
            initial_seed=42, event_seed=42001, mode="single", events_per_episode=1,
            event_tape=EventTape(
                initial_seed=42, event_seed=42001, mode="single",
                events=(),
            ),
            max_decisions=10,
        )
        env.reset()
        gv_before = env.graph_version
        dv_before = env.decision_version
        graph_before = build_graph_state(env)
        mask_before = graph_before.action_mask.clone()

        # Feed only 2 of 5 observations → should NOT confirm (need 3)
        bridge = env.runtime_bridge
        te = TruthEvent(
            event_id="E0001", event_type=EventType.TARGET_DISCOVERED,
            source_event="test", affected_uavs=("0",), affected_targets=("0",),
            occurred_at=0.0,
        )
        bridge.truth_state.record_truth_event(te)
        for i in range(2):
            obs = _obs(event_id="E0001", source_id=f"src-{i}")
            bridge.adapter.process_observation(obs)

        # True State changed immediately; belief and decision state did not.
        self.assertTrue(bridge.truth_state.is_discovered(0))
        self.assertEqual(len(bridge.get_belief().confirmed_events), 0)
        self.assertEqual(env.graph_version, gv_before)
        self.assertEqual(env.decision_version, dv_before)
        graph_after = build_graph_state(env)
        self.assertTrue(torch.equal(mask_before, graph_after.action_mask))


class DiscoveryThreeOfFiveTests(unittest.TestCase):
    """Audit item 2: TARGET_DISCOVERED requires >=3/5 distinct-source evidence."""

    def test_three_of_five_confirms(self):
        sm = ConfirmationStateMachine(target_confirmation_count=3)
        confirmed = False
        for i in range(5):
            obs = _obs(source_id=f"src-{i}")
            result = sm.process(obs)
            if result.confirmed_event is not None:
                confirmed = True
                break
        self.assertTrue(confirmed)

    def test_two_of_five_not_confirmed(self):
        sm = ConfirmationStateMachine(target_confirmation_count=3)
        for i in range(2):
            sm.process(_obs(source_id=f"src-{i}"))
        rec = sm.get("E0001")
        self.assertIsNotNone(rec)
        self.assertIn(rec.status, {ConfirmationStatus.SUSPECTED, ConfirmationStatus.NORMAL})

    def test_duplicate_does_not_increase_count(self):
        sm = ConfirmationStateMachine(target_confirmation_count=3)
        obs1 = _obs(source_id="src-0")
        obs2 = _obs(source_id="src-0", sequence=2, duplicate_of=obs1.observation_id)
        sm.process(obs1)
        sm.process(obs2)
        rec = sm.get("E0001")
        # Only one distinct source → not confirmed
        self.assertEqual(len(rec.positive_evidence_sources), 1)


class DestructionDualPathTests(unittest.TestCase):
    """Audit item 3: TARGET_DESTROYED has authoritative (direct) and strong (>=2 sources) paths."""

    def test_authoritative_single_confirm(self):
        sm = ConfirmationStateMachine()
        obs = _obs(
            event_type=EventType.TARGET_DESTROYED,
            signal_type="AUTHORITATIVE_TARGET_DESTROYED",
            affected_targets=("0",),
        )
        result = sm.process(obs)
        self.assertIsNotNone(result.confirmed_event)

    def test_strong_one_source_not_confirmed(self):
        sm = ConfirmationStateMachine(destruction_confirmation_count=2)
        sm.process(_obs(
            event_type=EventType.TARGET_DESTROYED,
            signal_type="STRONG_TARGET_DESTROYED",
            source_id="src-a",
            affected_targets=("0",),
        ))
        rec = sm.get("E0001")
        self.assertIn(rec.status, {ConfirmationStatus.SUSPECTED})

    def test_strong_two_same_source_not_confirmed(self):
        sm = ConfirmationStateMachine(destruction_confirmation_count=2)
        sm.process(_obs(
            event_type=EventType.TARGET_DESTROYED,
            signal_type="STRONG_TARGET_DESTROYED",
            source_id="src-a",
            affected_targets=("0",),
        ))
        sm.process(_obs(
            event_type=EventType.TARGET_DESTROYED,
            signal_type="STRONG_TARGET_DESTROYED",
            source_id="src-a",
            sequence=2,
            affected_targets=("0",),
        ))
        rec = sm.get("E0001")
        self.assertEqual(len(rec.positive_evidence_sources), 1)
        self.assertIn(rec.status, {ConfirmationStatus.SUSPECTED})

    def test_strong_two_independent_confirms(self):
        sm = ConfirmationStateMachine(destruction_confirmation_count=2)
        sm.process(_obs(
            event_type=EventType.TARGET_DESTROYED,
            signal_type="STRONG_TARGET_DESTROYED",
            source_id="src-a",
            affected_targets=("0",),
        ))
        result = sm.process(_obs(
            event_type=EventType.TARGET_DESTROYED,
            signal_type="STRONG_TARGET_DESTROYED",
            source_id="src-b",
            affected_targets=("0",),
        ))
        self.assertIsNotNone(result.confirmed_event)


class HeartbeatProbeTests(unittest.TestCase):
    """Audit item 4: UAV_DAMAGE heartbeat/probe chain."""

    def test_one_miss_suspected(self):
        sm = ConfirmationStateMachine(heartbeat_miss_threshold=3)
        sm.process(_obs(signal_type="HEARTBEAT_MISSED", affected_uavs=("1",)))
        rec = sm.get("E0001")
        self.assertEqual(rec.status, ConfirmationStatus.SUSPECTED)
        self.assertEqual(rec.heartbeat_miss_count, 1)

    def test_three_misses_probe_required(self):
        sm = ConfirmationStateMachine(heartbeat_miss_threshold=3)
        for sequence in range(1, 4):
            sm.process(_obs(signal_type="HEARTBEAT_MISSED", affected_uavs=("1",), sequence=sequence))
        rec = sm.get("E0001")
        self.assertEqual(rec.status, ConfirmationStatus.PROBE_REQUIRED)
        self.assertIsNotNone(rec.probe_started_at)

    def test_healthy_telemetry_false_alarm(self):
        sm = ConfirmationStateMachine(heartbeat_miss_threshold=3)
        sm.process(_obs(signal_type="HEARTBEAT_MISSED", affected_uavs=("1",), sequence=1))
        sm.process(_obs(signal_type="HEARTBEAT_MISSED", affected_uavs=("1",), sequence=2))
        rec = sm.get("E0001")
        self.assertEqual(rec.status, ConfirmationStatus.SUSPECTED)
        sm.process(_obs(signal_type="HEALTHY_TELEMETRY", source_id="telemetry-1", affected_uavs=("1",)))
        rec = sm.get("E0001")
        self.assertEqual(rec.status, ConfirmationStatus.FALSE_ALARM)

    def test_probe_timeout_confirms(self):
        sm = ConfirmationStateMachine(heartbeat_miss_threshold=3, probe_timeout=2.0)
        for sequence in range(1, 4):
            sm.process(_obs(signal_type="HEARTBEAT_MISSED", affected_uavs=("1",), received_at=1.0, sequence=sequence))
        rec = sm.get("E0001")
        self.assertEqual(rec.status, ConfirmationStatus.PROBE_REQUIRED)
        sm.advance(5.0)  # well past probe_timeout
        rec = sm.get("E0001")
        self.assertEqual(rec.status, ConfirmationStatus.CONFIRMED)

    def test_second_independent_source_confirms(self):
        sm = ConfirmationStateMachine(heartbeat_miss_threshold=3)
        sm.process(_obs(signal_type="HEARTBEAT_MISSED", source_id="src-a", affected_uavs=("1",)))
        result = sm.process(_obs(
            signal_type="HEARTBEAT_MISSED", source_id="src-b", affected_uavs=("1",),
        ))
        self.assertIsNotNone(result.confirmed_event)
        self.assertEqual(result.confirmed_event.status, ConfirmationStatus.CONFIRMED)


class UnconfirmedNoGraphVersionChangeTests(unittest.TestCase):
    """Audit item 5: unconfirmed observation must not change graph_version."""

    def test_burst_three_confirmed_single_increment(self):
        env = RandomEventAllocationEnv(
            initial_seed=42, event_seed=42001, mode="burst", events_per_episode=3,
            event_tape=EventTape(
                initial_seed=42, event_seed=42001, mode="burst",
                events=tuple(RandomEvent(
                    event_id=f"E{i:04d}", event_type=RandomEventType.REGION_VACANCY,
                    occurred_at=0.0, observed_at=0.0, source_event="test",
                    affected_uavs=(i,), affected_regions=(i,), severity=0.5,
                    event_seed=42001 + i, state_version=i,
                ) for i in range(3)),
            ),
            max_decisions=20,
        )
        graph, info = env.reset()
        # 3 burst events → all confirm (REGION_LEASE_VACANT direct) → graph_version should be 1
        self.assertEqual(env.graph_version, 1)

    def test_single_region_vacancy_one_increment(self):
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
        self.assertEqual(env.graph_version, 1)


import torch


if __name__ == "__main__":
    unittest.main()
