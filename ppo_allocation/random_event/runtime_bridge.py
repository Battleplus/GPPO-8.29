"""Runtime bridge connecting event_runtime into RandomEventAllocationEnv.

This module implements the mandatory event flow:

    TruthEvent
      -> True State (applied immediately to internal tracker)
      -> EventDetector (simulates observation delay, loss, duplicates)
      -> Observation
      -> ConfirmationStateMachine
      -> ConfirmedEvent
      -> EventQueue (atomic batch merge)
      -> BeliefState
      -> env state mutation (only confirmed events change belief)

Hard constraints enforced:
    - TruthEvent only modifies true state tracker
    - Observation only enters ConfirmationStateMachine
    - Unconfirmed events CANNOT change belief state, mask, or trigger rescheduling
    - Only ConfirmedEvent may modify belief and trigger re-decision
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

# event_runtime is at the repository root level
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from event_runtime.adapter import EventRuntimeAdapter, BeliefState
from event_runtime.events import (
    ConfirmationStatus,
    ConfirmedEvent,
    EventType as RuntimeEventType,
    TruthEvent,
)
from event_runtime.observation import Observation, WeakCommunicationProfile
from event_runtime.concurrency import ConcurrencyManager, CommandStatus


# ---------------------------------------------------------------------------
# Truth-state tracker (separate from belief state)
# ---------------------------------------------------------------------------

@dataclass
class TruthStateTracker:
    """Tracks the ground-truth state without exposing it to the policy.

    This is the single source of truth for what HAS happened.  The policy
    only sees the belief state derived from confirmed observations.
    """

    alive_uavs: set[int] = field(default_factory=set)
    damaged_uavs: set[int] = field(default_factory=set)
    discovered_targets: set[int] = field(default_factory=set)
    destroyed_targets: set[int] = field(default_factory=set)
    tracked_targets: dict[int, int] = field(default_factory=dict)  # target -> tracker uav
    vacant_regions: set[int] = field(default_factory=set)
    truth_events: list = field(default_factory=list)

    def record_truth_event(self, event: TruthEvent) -> None:
        """Record a truth event without modifying belief state."""
        self.truth_events.append(event)

    def to_canonical_json(self) -> bytes:
        """Byte-stable canonical JSON of truth state."""
        state = {
            "alive_uavs": sorted(self.alive_uavs),
            "damaged_uavs": sorted(self.damaged_uavs),
            "discovered_targets": sorted(self.discovered_targets),
            "destroyed_targets": sorted(self.destroyed_targets),
            "tracked_targets": {str(k): v for k, v in sorted(self.tracked_targets.items())},
            "vacant_regions": sorted(self.vacant_regions),
        }
        return json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def canonical_hash(self) -> str:
        return hashlib.sha256(self.to_canonical_json()).hexdigest()


# ---------------------------------------------------------------------------
# Detector simulation (truth -> observation with channel effects)
# ---------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    """Deterministic detector parameters seeded from the event seed."""

    loss_rate: float = 0.0
    duplicate_rate: float = 0.0
    false_positive_rate: float = 0.0
    out_of_order_max_delay: float = 0.0
    partition_sources: list = field(default_factory=list)


class DeterministicDetector:
    """Simulates observation-channel effects using a seeded RNG.

    All randomness is derived from a single seed so that the same truth tape
    always produces the same observation tape.
    """

    def __init__(self, seed: int, config: DetectorConfig | None = None):
        self.rng = np.random.default_rng(seed)
        self.config = config or DetectorConfig()
        self._partition_active: dict[str, bool] = {}
        self._partition_end: dict[str, float] = {}

    def generate_observation(
        self,
        truth_event: TruthEvent,
        observation_delay: float,
        current_time: float,
        source_override: str | None = None,
    ) -> Observation | None:
        """Convert a truth event into an observation, applying channel effects.

        ``source_override`` lets multi-evidence events emit observations from
        distinct sources (3-of-5 / >=2 independent sources).  Returns None if
        the observation is lost in transit.
        """
        source_id = source_override or truth_event.source_event

        # Check partition
        if source_id in self.config.partition_sources:
            if self.rng.random() < 0.3:
                self._partition_active[source_id] = True
                self._partition_end[source_id] = current_time + self.rng.uniform(1.0, 3.0)
            if self._partition_active.get(source_id, False):
                if current_time < self._partition_end.get(source_id, 0):
                    return None
                else:
                    self._partition_active[source_id] = False

        # Loss check
        if self.rng.random() < self.config.loss_rate:
            return None

        # False positive check
        is_false_positive = self.rng.random() < self.config.false_positive_rate

        # Out-of-order delay
        order_jitter = 0.0
        if self.config.out_of_order_max_delay > 0:
            order_jitter = self.rng.uniform(0, self.config.out_of_order_max_delay)

        emitted_at = truth_event.occurred_at + observation_delay
        received_at = emitted_at + order_jitter

        obs = Observation(
            observation_id=f"OBS-{truth_event.event_id}-{int(self.rng.integers(0, 10**9))}",
            event_id=truth_event.event_id,
            event_type=RuntimeEventType(truth_event.event_type.value),
            source_event=truth_event.source_event,
            source_id=source_id,
            source_type="sensor",
            signal_type=self._signal_type_for(truth_event),
            sequence=1,
            confidence=0.95 if not is_false_positive else 0.3,
            positive=True if not is_false_positive else bool(self.rng.integers(0, 2)),
            emitted_at=emitted_at,
            received_at=received_at,
            occurred_at=truth_event.occurred_at,
            affected_uavs=tuple(str(u) for u in truth_event.affected_uavs),
            affected_regions=tuple(str(r) for r in truth_event.affected_regions),
            affected_targets=tuple(str(t) for t in truth_event.affected_targets),
            severity=truth_event.severity,
            event_seed=truth_event.event_seed,
            state_version=truth_event.state_version,
            is_false_positive=is_false_positive,
        )
        return obs

    @staticmethod
    def _signal_type_for(event: TruthEvent) -> str:
        mapping = {
            "UAV_DAMAGE": "ACTIVE_FAILURE_REPORT",
            "TARGET_DISCOVERED": "SENSOR_DETECTION",
            "TARGET_DESTROYED": "AUTHORITATIVE_TARGET_DESTROYED",
            "REGION_VACANCY": "REGION_LEASE_VACANT",
        }
        return mapping.get(event.event_type.value, "GENERIC")


# ---------------------------------------------------------------------------
# Runtime Bridge (main integration point)
# ---------------------------------------------------------------------------

class RuntimeBridge:
    """Bridges event_runtime into RandomEventAllocationEnv.

    This is the single integration point.  The environment calls:

        1. bridge.ingest_truth_event(event, current_time)
           -> records truth, generates observation, runs confirmation

        2. bridge.get_confirmed_events()
           -> returns events confirmed since last call

        3. bridge.apply_confirmed_to_env(env)
           -> mutates env state ONLY for confirmed events

        4. bridge.get_belief()
           -> returns current belief state for graph construction

    The environment's _apply_random_event() is REPLACED by this flow.
    """

    def __init__(
        self,
        *,
        merge_window: float = 0.10,
        suspicion_timeout: float = 5.0,
        heartbeat_miss_threshold: int = 3,
        target_confirmation_count: int = 3,
        destruction_confirmation_count: int = 2,
        detector_seed: int = 42,
        detector_config: DetectorConfig | None = None,
    ):
        self.adapter = EventRuntimeAdapter(
            merge_window=merge_window,
            suspicion_timeout=suspicion_timeout,
            heartbeat_miss_threshold=heartbeat_miss_threshold,
            target_confirmation_count=target_confirmation_count,
            destruction_confirmation_count=destruction_confirmation_count,
        )
        self.truth_state = TruthStateTracker()
        self.detector = DeterministicDetector(
            seed=detector_seed,
            config=detector_config or DetectorConfig(),
        )
        self._confirmed_since_last: list = []
        self._observation_count = 0
        self._truth_to_observation_delay = 0.5

    def reset(self) -> None:
        """Reset all bridge state for a new episode."""
        self.adapter = EventRuntimeAdapter(
            merge_window=self.adapter.event_queue.merge_window,
            suspicion_timeout=self.adapter.state_machine.suspicion_timeout,
            heartbeat_miss_threshold=self.adapter.state_machine.heartbeat_miss_threshold,
            target_confirmation_count=self.adapter.state_machine.target_confirmation_count,
            destruction_confirmation_count=self.adapter.state_machine.destruction_confirmation_count,
        )
        self.truth_state = TruthStateTracker()
        self._confirmed_since_last = []
        self._observation_count = 0

    # Evidence requirements per event type (Phase D).
    # TARGET_DISCOVERED requires 3-of-5 distinct-source evidence.
    # TARGET_DESTROYED requires >=2 independent strong evidence sources.
    _EVIDENCE_COUNT: dict = {
        "TARGET_DISCOVERED": 3,
        "TARGET_DESTROYED": 2,
    }

    def ingest_truth_event(
        self,
        truth_event: TruthEvent,
        current_time: float,
        observation_delay: float | None = None,
    ) -> ConfirmedEvent | None:
        """Process a truth event through the full pipeline.

        Multi-evidence event types (TARGET_DISCOVERED 3-of-5, TARGET_DESTROYED
        >=2 sources) generate observations from DISTINCT sources so the
        confirmation state machine can reach its required evidence count.
        Single-evidence types (UAV_DAMAGE trusted failure, REGION_VACANCY)
        confirm on the first observation.

        Returns the ConfirmedEvent if confirmation succeeds immediately,
        otherwise None (confirmation pending).
        """
        # 1. Record truth (does NOT affect belief)
        self.truth_state.record_truth_event(truth_event)

        # 2. Generate observation(s) with channel effects
        delay = observation_delay if observation_delay is not None else self._truth_to_observation_delay
        evidence_count = self._EVIDENCE_COUNT.get(truth_event.event_type.value, 1)
        confirmed_event: ConfirmedEvent | None = None
        for source_index in range(evidence_count):
            obs = self.detector.generate_observation(
                truth_event,
                delay,
                current_time,
                source_override=f"src-{source_index}",
            )
            if obs is None:
                continue
            self._observation_count += 1

            # 3. Run through confirmation state machine
            result = self.adapter.process_observation(obs)
            if result is not None:
                confirmed_event = result

        if confirmed_event is not None:
            self._confirmed_since_last.append(confirmed_event)
        return confirmed_event

    def advance_time(self, now: float) -> list:
        """Advance time and collect any timeout-triggered confirmations."""
        confirmed = self.adapter.advance_time(now)
        self._confirmed_since_last.extend(confirmed)
        return confirmed

    def get_confirmed_events(self) -> list:
        """Get events confirmed since last call and clear the buffer."""
        events = list(self._confirmed_since_last)
        self._confirmed_since_last = []
        return events

    def apply_confirmed_to_env(self, env: Any, confirmed: ConfirmedEvent) -> None:
        """Apply a confirmed event to the environment state.

        This is the ONLY path through which events modify env state.
        The env's _apply_random_event() should NOT be called directly.
        """
        from config import EventType, NO_TARGET, NO_UAV, TaskType

        kind = confirmed.event_type
        actual: set[int] = set()

        if kind == RuntimeEventType.UAV_DAMAGE:
            if confirmed.affected_uavs:
                uid = int(confirmed.affected_uavs[0])
                if uid in env.uavs:
                    uav = env.uavs[uid]
                    actual.update(uav.regions)
                    if uav.alive:
                        if uav.task == TaskType.TRACK and uav.target_id != NO_TARGET:
                            target = env.targets[uav.target_id]
                            target.tracked = False
                            target.tracker_id = NO_UAV
                        uav.alive = False
                        uav.task = TaskType.IDLE
                        uav.target_id = NO_TARGET
                        for rid in tuple(uav.regions):
                            env._clear_region_assignment(rid)

        elif kind == RuntimeEventType.TARGET_DISCOVERED:
            if confirmed.affected_targets and confirmed.affected_uavs:
                tid = int(confirmed.affected_targets[0])
                uid = int(confirmed.affected_uavs[0])
                if tid in env.targets and uid in env.uavs:
                    target, uav = env.targets[tid], env.uavs[uid]
                    if not target.destroyed and uav.alive:
                        actual.update(uav.regions)
                        for rid in tuple(uav.regions):
                            env._clear_region_assignment(rid)
                        target.discovered = True
                        target.tracked = True
                        target.tracker_id = uid
                        uav.task = TaskType.TRACK
                        uav.target_id = tid

        elif kind == RuntimeEventType.TARGET_DESTROYED:
            if confirmed.affected_targets:
                tid = int(confirmed.affected_targets[0])
                if tid in env.targets:
                    target = env.targets[tid]
                    uid = int(target.tracker_id)
                    target.discovered = True
                    target.destroyed = True
                    target.tracked = False
                    target.tracker_id = NO_UAV
                    if uid != NO_UAV and uid in env.uavs and env.uavs[uid].alive:
                        env.uavs[uid].task = TaskType.IDLE
                        env.uavs[uid].target_id = NO_TARGET

        elif kind == RuntimeEventType.REGION_VACANCY:
            if confirmed.affected_regions:
                rid = int(confirmed.affected_regions[0])
                if rid in env.regions:
                    actual.add(rid)
                    env._clear_region_assignment(rid)

        for rid in actual:
            env.regions[rid].need_reassign = True
            env.pending_regions.add(rid)
            env.vacancy_duration.setdefault(rid, 0.0)

        # NOTE: graph_version is NOT incremented here.  Burst batches are
        # committed atomically by the environment with exactly one increment
        # per batch (Phase G).  Incrementing here would double-count.
        env.communication_trigger_count += 1
        env.communication_bytes += len(
            json.dumps(confirmed.to_dict(), sort_keys=True).encode("utf-8")
        )

    def get_belief(self) -> BeliefState:
        return self.adapter.belief

    def get_observation_count(self) -> int:
        return self._observation_count

    def get_truth_event_count(self) -> int:
        return len(self.truth_state.truth_events)

    def get_confirmation_stats(self) -> dict[str, Any]:
        sm = self.adapter.state_machine
        return {
            "total_observations": self._observation_count,
            "total_truth_events": len(self.truth_state.truth_events),
            "false_alarm_count": sm.false_alarm_count,
            "duplicate_observation_count": sm.duplicate_observation_count,
            "late_observation_count": sm.late_observation_count,
            "confirmed_events": len(self.adapter.belief.confirmed_events),
            "pending_queue_size": len(self.adapter.event_queue),
            "truth_canonical_hash": self.truth_state.canonical_hash(),
        }
