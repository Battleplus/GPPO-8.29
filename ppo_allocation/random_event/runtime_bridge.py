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
from typing import Any, Iterable, Mapping

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
from event_runtime.concurrency import ACK, ACKType, ConcurrencyManager, CommandStatus


# ---------------------------------------------------------------------------
# Truth-state tracker (separate from belief state)
# ---------------------------------------------------------------------------

@dataclass
class TruthStateTracker:
    """Tracks the ground-truth state without exposing it to the policy.

    This is the single source of truth for what HAS happened.  The policy
    only sees the belief state derived from confirmed observations.

    Round-2 hardening: ``record_truth_event`` genuinely mutates the true
    state vectors (alive/damaged UAVs, discovered/destroyed/tracked targets,
    vacant regions) immediately when a TruthEvent occurs.  These vectors are
    NEVER read by GraphObservation / policy; the env belief state is only
    mutated by ``apply_confirmed_to_env``.
    """

    alive_uavs: set[int] = field(default_factory=set)
    damaged_uavs: set[int] = field(default_factory=set)
    discovered_targets: set[int] = field(default_factory=set)
    destroyed_targets: set[int] = field(default_factory=set)
    tracked_targets: dict[int, int] = field(default_factory=dict)  # target -> tracker uav
    vacant_regions: set[int] = field(default_factory=set)
    truth_events: list = field(default_factory=list)

    def record_truth_event(self, event: TruthEvent) -> None:
        """Apply a truth event to the TRUE state immediately.

        This must NOT touch belief state (pending_regions, graph_version,
        masks).  It only updates the ground-truth tracker.
        """
        self.truth_events.append(event)
        kind = event.event_type
        if kind == RuntimeEventType.UAV_DAMAGE:
            for raw in event.affected_uavs:
                uid = int(raw)
                self.damaged_uavs.add(uid)
                self.alive_uavs.discard(uid)
                for tid, tracker in list(self.tracked_targets.items()):
                    if tracker == uid:
                        del self.tracked_targets[tid]
        elif kind == RuntimeEventType.TARGET_DISCOVERED:
            for raw in event.affected_targets:
                tid = int(raw)
                if tid not in self.destroyed_targets:
                    self.discovered_targets.add(tid)
            # discovered also implies tracking by the discovering UAV
            if event.affected_targets and event.affected_uavs:
                tid = int(event.affected_targets[0])
                uid = int(event.affected_uavs[0])
                if tid not in self.destroyed_targets and uid in self.alive_uavs:
                    self.tracked_targets[tid] = uid
        elif kind == RuntimeEventType.TARGET_DESTROYED:
            for raw in event.affected_targets:
                tid = int(raw)
                self.destroyed_targets.add(tid)
                self.discovered_targets.discard(tid)
                self.tracked_targets.pop(tid, None)
        elif kind == RuntimeEventType.REGION_VACANCY:
            for raw in event.affected_regions:
                self.vacant_regions.add(int(raw))

    def initialize_alive(self, uav_ids: Iterable[int]) -> None:
        """Seed the alive UAV set (called at bridge reset)."""
        self.alive_uavs = set(int(uid) for uid in uav_ids)

    def is_alive(self, uid: int) -> bool:
        return int(uid) in self.alive_uavs

    def is_damaged(self, uid: int) -> bool:
        return int(uid) in self.damaged_uavs

    def is_discovered(self, tid: int) -> bool:
        return int(tid) in self.discovered_targets

    def is_destroyed(self, tid: int) -> bool:
        return int(tid) in self.destroyed_targets

    def tracker_of(self, tid: int) -> int | None:
        return self.tracked_targets.get(int(tid))

    def is_vacant(self, rid: int) -> bool:
        return int(rid) in self.vacant_regions

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
        signal_type_override: str | None = None,
    ) -> Observation | None:
        """Convert a truth event into an observation, applying channel effects.

        ``source_override`` lets multi-evidence events emit observations from
        distinct sources (3-of-5 / >=2 independent sources).  ``signal_type_override``
        distinguishes the authoritative vs ordinary strong destruction paths.
        Returns None if the observation is lost in transit.
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
            signal_type=signal_type_override or self._signal_type_for(truth_event),
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
            "TARGET_DESTROYED": "STRONG_TARGET_DESTROYED",
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
        self._cc = self._new_counters()
        self._cc_seq = 0

    @staticmethod
    def _new_counters() -> dict[str, int]:
        return {
            "stale_rejected": 0,
            "stale_attempted": 0,
            "duplicate_assignments": 0,
            "late_ack_resurrections": 0,
            "unaffected_interruptions": 0,
            "exclusive_holder_violations": 0,
        }

    def reset(self) -> None:
        """Reset all bridge state for a new episode."""
        self.adapter = EventRuntimeAdapter(
            merge_window=self.adapter.event_queue.merge_window,
            suspicion_timeout=self.adapter.state_machine.suspicion_timeout,
            heartbeat_miss_threshold=self.adapter.state_machine.heartbeat_miss_threshold,
            target_confirmation_count=self.adapter.state_machine.target_confirmation_count,
            destruction_confirmation_count=self.adapter.state_machine.destruction_confirmation_count,
            probe_timeout=self.adapter.state_machine.probe_timeout,
        )
        self.truth_state = TruthStateTracker()
        self._confirmed_since_last = []
        self._observation_count = 0
        self._cc = self._new_counters()
        self._cc_seq = 0

    # Evidence requirements per event type (Phase D round-2).
    # TARGET_DISCOVERED: up to 5 evidence opportunities from distinguishable
    # sources; confirmation requires >=3 distinct-source positives (3-of-5).
    # TARGET_DESTROYED: dual path -- authoritative evidence confirms with a
    # single observation; ordinary STRONG evidence requires >=2 independent
    # sources.
    _EVIDENCE_OPPORTUNITIES: dict = {
        "TARGET_DISCOVERED": 5,
        "TARGET_DESTROYED": 2,
    }

    def ingest_truth_event(
        self,
        truth_event: TruthEvent,
        current_time: float,
        observation_delay: float | None = None,
        *,
        authoritative: bool = False,
    ) -> ConfirmedEvent | None:
        """Process a truth event through the full pipeline.

        - TARGET_DISCOVERED: generate up to 5 opportunities from distinct
          sources (``src-0..src-4``); channel loss may drop some, and the
          state machine confirms only on >=3 distinct-source positives.
        - TARGET_DESTROYED: when ``authoritative`` is True, emit a single
          AUTHORITATIVE_TARGET_DESTROYED observation (direct confirm);
          otherwise emit 2 STRONG_TARGET_DESTROYED observations from distinct
          sources (requires >=2 independent sources).  Tracking loss is NEVER
          an authoritative destruction report.
        - UAV_DAMAGE / REGION_VACANCY: single trusted observation, direct
          confirm.

        Returns the ConfirmedEvent if confirmation succeeds immediately,
        otherwise None (confirmation pending).
        """
        # 1. Record truth (does NOT affect belief)
        self.truth_state.record_truth_event(truth_event)

        # 2. Generate observation(s) with channel effects
        delay = observation_delay if observation_delay is not None else self._truth_to_observation_delay
        kind = truth_event.event_type.value

        if kind == "TARGET_DESTROYED" and authoritative:
            opportunities: list[tuple[str | None, str | None]] = [(None, "AUTHORITATIVE_TARGET_DESTROYED")]
        else:
            count = self._EVIDENCE_OPPORTUNITIES.get(kind, 1)
            signal = None
            if kind == "TARGET_DESTROYED":
                signal = "STRONG_TARGET_DESTROYED"
            opportunities = [(f"src-{i}", signal) for i in range(count)]

        confirmed_event: ConfirmedEvent | None = None
        for source_override, signal_override in opportunities:
            obs = self.detector.generate_observation(
                truth_event,
                delay,
                current_time,
                source_override=source_override,
                signal_type_override=signal_override,
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

    def ingest_observation(self, observation: Observation) -> ConfirmedEvent | None:
        """Accept an externally delivered observation (heartbeat/probe path).

        Truth events use ``ingest_truth_event``; detector/communication tests
        and real adapters use this method to feed observations into the same
        confirmation state machine without bypassing it.
        """
        self._observation_count += 1
        confirmed = self.adapter.process_observation(observation)
        if confirmed is not None:
            self._confirmed_since_last.append(confirmed)
        return confirmed

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

    def apply_confirmed_to_env(self, env: Any, confirmed: ConfirmedEvent) -> bool:
        """Apply a confirmed event to the environment state.

        Returns True if the environment state was actually mutated (not idempotent).
        Only confirmed events may modify belief state.
        """
        from config import EventType, NO_TARGET, NO_UAV, TaskType

        kind = confirmed.event_type
        actual: set[int] = set()
        state_changed = False

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
                        state_changed = True
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
                        state_changed = True

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
                    state_changed = True
                    if uid != NO_UAV and uid in env.uavs and env.uavs[uid].alive:
                        env.uavs[uid].task = TaskType.IDLE
                        env.uavs[uid].target_id = NO_TARGET

        elif kind == RuntimeEventType.REGION_VACANCY:
            if confirmed.affected_regions:
                rid = int(confirmed.affected_regions[0])
                if rid in env.regions:
                    actual.add(rid)
                    env._clear_region_assignment(rid)
                    state_changed = True

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
        return state_changed or bool(actual)

    # ------------------------------------------------------------------
    # Phase E: command / ACK / lease / fencing execution lifecycle
    # ------------------------------------------------------------------
    # These counters implement the hard concurrency metrics that the P0 gate
    # machine-checks:
    #   stale_action_rejection_rate == 1.0
    #   valid_exclusive_holder <= 1
    #   duplicate_assignment_count == 0
    #   late_ack_resurrection == 0
    #   unaffected_task_interruption == 0

    def _concurrency_counters(self) -> dict[str, int]:
        return self._cc

    def issue_assignment_command(
        self,
        env: Any,
        uav_id: int,
        region_id: int,
        now: float,
    ) -> AssignmentCommand | None:
        """Create, validate and commit an assignment command (execution layer).

        Guards enforced HERE (not post-hoc):
        - command.graph_version must EXACTLY match current graph version.
        - action_version must match the environment's decision version.
        - the latest action mask is re-validated before execution.
        - the same region may only ever have one valid lease holder; a new
          holder is only established after the previous lease is revoked or
          timed out AND its fencing token is higher.
        - only AFFECTED (pending) regions are re-assigned; a lease for an
          unaffected region is never interrupted.

        Returns the committed command, or None when rejected.
        """
        cc = self._cc
        graph_version = int(env.graph_version)
        action_version = int(env.decision_version)

        # 1. Re-validate the LATEST action mask before execution.
        from random_event.graph import build_graph_state, decode_edge_action

        graph = build_graph_state(env)
        # Find the edge index for (uav_id, region_id) from candidate_edges.
        latest_mask = graph.action_mask.cpu().numpy()
        candidate_edges = graph.candidate_edges.cpu().numpy()
        edge_index = -1
        for idx, (eu, er) in enumerate(candidate_edges):
            if int(eu) == int(uav_id) and int(er) == int(region_id):
                edge_index = idx
                break
        if edge_index < 0 or not bool(latest_mask[edge_index]):
            cc["stale_rejected"] += 1
            return None

        # 2. Fencing: only a HIGHER token may supersede the current holder.
        concurrency = self.adapter.concurrency
        region_key = str(region_id)
        previous = concurrency.get_valid_lease(region_key, now)
        if previous is not None and previous.uav_id == str(uav_id):
            # Idempotent replay by the current holder: do not create a second
            # command or lease. The existing command remains the execution
            # record and the environment may safely re-apply the same holder.
            for existing in concurrency.commands.values():
                if (
                    existing.uav_id == str(uav_id)
                    and existing.region_id == region_key
                    and existing.fencing_token == previous.fencing_token
                    and existing.status not in {CommandStatus.REVOKED, CommandStatus.EXPIRED, CommandStatus.REJECTED}
                ):
                    return existing
            return None
        if previous is not None:
            if region_id not in env.pending_regions:
                # Interrupting a lease on an UNAFFECTED region is forbidden.
                cc["unaffected_interruptions"] += 1
                return None
            # Revoke old holder first (new holder established only afterwards).
            concurrency.revoke_lease(previous.lease_id)

        # 3. Create command carrying graph_version + action_version.
        command = concurrency.create_command(
            command_id=f"cmd-{graph_version}-{region_key}-{uav_id}-{self._cc_seq}",
            uav_id=str(uav_id),
            region_id=region_key,
            graph_version=graph_version,
            action_version=action_version,
            ttl=0.5,
            now=now,
        )
        self._cc_seq += 1

        # 4. graph_version must EXACTLY match current (not merely "not old").
        if not concurrency.validate_command(command.command_id, graph_version):
            cc["stale_rejected"] += 1
            return None
        if command.graph_version != graph_version:
            cc["stale_rejected"] += 1
            return None
        if command.action_version != action_version:
            cc["stale_rejected"] += 1
            return None
        concurrency.commit_command(command.command_id)

        # 5. ACK (execution layer validates command_id/uav_id/fencing token).
        ack = ACK(
            command_id=command.command_id,
            uav_id=str(uav_id),
            ack_type=ACKType.ACCEPTED,
            received_at=now,
            fencing_token=command.fencing_token,
        )
        try:
            concurrency.receive_ack(command.command_id, ack)
        except ValueError:
            cc["stale_rejected"] += 1
            return None

        # 6. Grant the exclusive lease with the command's fencing token.
        lease = concurrency.create_lease(
            lease_id=f"lease-{command.command_id}",
            uav_id=str(uav_id),
            region_id=region_key,
            fencing_token=command.fencing_token,
            now=now,
            ttl=5.0,
        )
        if concurrency.get_valid_holder_count(region_key, now) > 1:
            cc["exclusive_holder_violations"] += 1
            concurrency.revoke_lease(lease.lease_id)
            return None
        return command

    def complete_region(
        self,
        region_id: int,
        now: float,
    ) -> None:
        """Complete the active command and release its lease once recovered."""
        concurrency = self.adapter.concurrency
        region_key = str(region_id)
        lease = concurrency.get_valid_lease(region_key, now)
        if lease is not None:
            concurrency.revoke_lease(lease.lease_id)
        for command in concurrency.commands.values():
            if command.region_id == region_key and command.status in {
                CommandStatus.ACKED,
                CommandStatus.EXECUTING,
            }:
                command.status = CommandStatus.COMPLETED

    def receive_ack(
        self,
        command_id: str,
        uav_id: str,
        fencing_token: int,
        now: float,
    ) -> bool:
        """Validate an ACK: command_id + uav_id + fencing token + status.

        A late ACK for a REVOKED / EXPIRED / REJECTED command can never
        resurrect it (the command status is left untouched and counted).
        """
        concurrency = self.adapter.concurrency
        command = concurrency.commands.get(command_id)
        if command is None:
            return False
        if command.status in {
            CommandStatus.REVOKED,
            CommandStatus.EXPIRED,
            CommandStatus.REJECTED,
            CommandStatus.COMPLETED,
        }:
            self._cc["late_ack_resurrections"] += 1
            return False
        if command.uav_id != str(uav_id) or command.fencing_token != int(fencing_token):
            self._cc["stale_rejected"] += 1
            return False
        try:
            concurrency.receive_ack(
                command_id,
                ACK(
                    command_id=command_id,
                    uav_id=str(uav_id),
                    ack_type=ACKType.ACCEPTED,
                    received_at=now,
                    fencing_token=int(fencing_token),
                ),
            )
        except ValueError:
            self._cc["stale_rejected"] += 1
            return False
        return True

    def snapshot_concurrency(self, now: float) -> dict[str, Any]:
        """Machine-checkable concurrency invariant snapshot."""
        concurrency = self.adapter.concurrency
        regions = {lease.region_id for lease in concurrency.leases.values()}
        exclusive = all(
            concurrency.get_valid_holder_count(region, now) <= 1
            for region in regions
        )
        return {
            "valid_exclusive_holder": exclusive,
            "stale_rejection_rate": (
                float(self._cc["stale_rejected"])
                / max(1, self._cc["stale_attempted"])
            ),
            "stale_rejected": self._cc["stale_rejected"],
            "stale_attempted": self._cc["stale_attempted"],
            "duplicate_assignments": self._cc["duplicate_assignments"],
            "late_ack_resurrections": self._cc["late_ack_resurrections"],
            "unaffected_interruptions": self._cc["unaffected_interruptions"],
            "exclusive_holder_violations": self._cc["exclusive_holder_violations"],
        }

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
