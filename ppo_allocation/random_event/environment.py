"""Multi-event, event-triggered environment for the 4-UAV allocation task.

The class subclasses the legacy environment only to reuse its entity creation
and assignment invariants.  It deliberately replaces the one-event/one-step
termination logic with a replayable event tape, a pending-region queue and
graph-versioned edge actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import sys
from typing import Any, Mapping

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# The original package uses top-level ``config/env/policy`` imports.  Preserve
# that public execution style while also allowing ``ppo_allocation.random_event``
# imports from the repository root.
_PPO_DIR = Path(__file__).resolve().parents[1]
if str(_PPO_DIR) not in sys.path:
    sys.path.insert(0, str(_PPO_DIR))

from config import EventType, NO_TARGET, NO_UAV, TaskType  # noqa: E402
from env.event import Event as LegacyEvent  # noqa: E402
from env.uav_env import UAVTaskAllocationEnv  # noqa: E402

from .events import EventTape, RandomEvent, RandomEventType
from .graph import HeteroGraphState, build_graph_state, decode_edge_action
from .reward import CostWeights, assignment_map, compute_cost, cost_difference_reward
from .runtime_bridge import RuntimeBridge, DetectorConfig
from .scheduler import RandomEventScheduler, SchedulerState


@dataclass
class EventRuntime:
    event: RandomEvent
    actual_affected_regions: tuple[int, ...] = ()
    status: str = "scheduled"
    applied_at: float | None = None
    resolved_at: float | None = None
    application_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.event.to_dict(),
            "actual_affected_regions": list(self.actual_affected_regions),
            "status": self.status,
            "applied_at": self.applied_at,
            "resolved_at": self.resolved_at,
            "recovery_delay": None if self.resolved_at is None else self.resolved_at - self.event.observed_at,
            "application_note": self.application_note,
        }


class StaleDecisionError(RuntimeError):
    """Raised by the strict submission API when the graph changed in flight."""


@dataclass(frozen=True)
class DecisionContext:
    """Snapshot of the decision-time state returned by ``begin_decision``.

    Carries the graph, graph_version, and action_version that were current at
    the instant the policy was asked to make a decision.  These values must be
    passed back via ``submit_action`` so the execution layer can detect stale
    submissions.
    """
    graph: HeteroGraphState
    graph_version: int
    action_version: int


@dataclass(frozen=True)
class ActionSubmission:
    """Unified action submission contract for versioned submission.

    Both ``PPOTrainer.collect_rollout`` and ``run_episode`` produce an
    ``ActionSubmission`` that carries the action together with the decision-time
    versions.  The environment validates both before executing.

    Both ``LightweightEnvAdapter`` and ``SFC/IsaacSimAdapter`` consume this
    contract.
    """
    action: int
    graph_version: int
    action_version: int

    @classmethod
    def from_decision(cls, action: int, ctx: DecisionContext) -> "ActionSubmission":
        """Create an ActionSubmission from a DecisionContext."""
        return cls(
            action=int(action),
            graph_version=ctx.graph_version,
            action_version=ctx.action_version,
        )


class RandomEventAllocationEnv(UAVTaskAllocationEnv):
    """Event-tape environment whose action is a single UAV--Region edge."""

    def __init__(
        self,
        *,
        initial_seed: int = 42,
        event_seed: int = 42001,
        mode: str = "sequential",
        events_per_episode: int = 5,
        event_tape: EventTape | None = None,
        max_decisions: int = 100,
        max_time: float = 200.0,
        decision_duration: float = 1.0,
        cost_weights: CostWeights | None = None,
    ) -> None:
        super().__init__(max_decision_steps=max_decisions, seed=initial_seed, random_event_mode=False)
        self.initial_seed = int(initial_seed)
        self.event_seed = int(event_seed)
        self.mode = str(mode)
        self.events_per_episode = int(events_per_episode)
        self.supplied_tape = event_tape
        self.max_time = float(max_time)
        self.decision_duration = float(decision_duration)
        self.cost_weights = cost_weights or CostWeights()
        self.scheduler = RandomEventScheduler(event_count=events_per_episode)
        self.action_space = spaces.Discrete(4 * 4 + 1)

        self.event_tape: EventTape | None = None
        self.current_time = 0.0
        self.graph_version = 0
        self.decision_version = 0
        self.next_event_index = 0
        self._observation_order: list[int] = []
        self.pending_regions: set[int] = set()
        self.vacancy_duration: dict[int, float] = {}
        self.event_queue: list[str] = []
        self.event_records: dict[str, EventRuntime] = {}
        self.communication_trigger_count = 0
        self.communication_bytes = 0
        self.repair_count = 0
        self.stale_rejection_count = 0
        self.last_event: RandomEvent | None = None
        self.runtime_bridge: RuntimeBridge | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Avoid UAVTaskAllocationEnv.reset(): it mutates the state by generating
        # one legacy random event before the replay tape is installed.
        selected_seed = self.initial_seed if seed is None else int(seed)
        gym.Env.reset(self, seed=selected_seed)
        self.rng = np.random.default_rng(selected_seed)
        self.decision_step = 0
        self.steps_since_event = 0
        self._init_regions()
        self._init_uavs()
        self._init_targets()
        self._initial_assignment()

        self.current_time = 0.0
        self.graph_version = 0
        self.decision_version = 0
        self.next_event_index = 0
        self._observation_order = []
        self.pending_regions = set()
        self.vacancy_duration = {rid: 0.0 for rid in self.regions}
        self.event_queue = []
        self.event_records = {}
        self.communication_trigger_count = 0
        self.communication_bytes = 0
        self.repair_count = 0
        self.stale_rejection_count = 0
        self.last_event = None
        self.current_event = LegacyEvent(EventType.REGION_VACANCY, [], description="awaiting event")
        # Phase C: Initialize runtime bridge for truth→observation→confirmation flow
        self.runtime_bridge = RuntimeBridge(
            merge_window=0.10,
            detector_seed=self.event_seed,
            detector_config=DetectorConfig(
                loss_rate=0.0,
                duplicate_rate=0.0,
                false_positive_rate=0.0,
                out_of_order_max_delay=0.0,
            ),
        )
        # Seed the true-state tracker separately from the environment belief.
        # This does not expose truth to graph construction or policy input.
        self.runtime_bridge.truth_state.initialize_alive(self.uavs.keys())

        if self.supplied_tape is None:
            initial_state = SchedulerState.from_entities(
                list(self.uavs.values()), list(self.regions.values()), list(self.targets.values())
            )
            self.event_tape = self.scheduler.generate_tape(
                initial_state,
                initial_seed=selected_seed,
                event_seed=self.event_seed,
                mode=self.mode,
                event_count=self.events_per_episode,
            )
        else:
            self.event_tape = self.supplied_tape
            if self.event_tape.initial_seed != selected_seed:
                # A supplied tape is an immutable experiment case.  Its initial
                # state seed wins unless the caller explicitly reset with it.
                selected_seed = self.event_tape.initial_seed
                self.rng = np.random.default_rng(selected_seed)
                self._init_regions(); self._init_uavs(); self._init_targets(); self._initial_assignment()

        assert self.event_tape is not None
        # Keep the immutable tape in occurrence/generation order for exact
        # replay evidence, but process detector reports in observation order.
        # Under weak communication E1 may be observed before an earlier E0.
        self._observation_order = sorted(
            range(len(self.event_tape.events)),
            key=lambda index: (
                self.event_tape.events[index].observed_at,
                self.event_tape.events[index].occurred_at,
                self.event_tape.events[index].event_id,
            ),
        )
        if self.event_tape.events:
            self.current_time = min(event.observed_at for event in self.event_tape.events)
            self._ingest_observed_events()
        graph = build_graph_state(self)
        return graph, self._info(graph, reset=True)

    # ------------------------------------------------------------------
    # Real-time event ingestion and versioning
    # ------------------------------------------------------------------

    def advance_time(self, delta: float) -> list[str]:
        """Advance the monitoring clock and ingest all newly observed events.

        This method models the situation where another event is detected while
        a policy is still computing.  A later ``submit_action`` with the old
        graph version will then be rejected.
        """

        if delta < 0:
            raise ValueError("delta must be non-negative")
        for rid in self.pending_regions:
            self.vacancy_duration[rid] = self.vacancy_duration.get(rid, 0.0) + float(delta)
        self.current_time += float(delta)
        return self._ingest_observed_events()

    def _ingest_observed_events(self) -> list[str]:
        """Ingest events through the runtime bridge pipeline with burst batching.

        Phase C: Events flow through TruthEvent → Observation → Confirmation
        → BeliefState.  Only confirmed events modify env state.

        Phase G (burst atomicity): events whose observed_at falls inside the
        burst window (100ms) are collected into ONE atomic batch.  The batch
        is committed with a single graph_version increment and a single policy
        call afterwards; there are no intermediate policy calls inside the
        window.
        """
        assert self.event_tape is not None
        burst_window = 0.1 if self.mode == "burst" else 0.0
        ingested: list[str] = []
        while self.next_event_index < len(self._observation_order):
            tape_index = self._observation_order[self.next_event_index]
            event = self.event_tape.events[tape_index]
            if event.observed_at > self.current_time + 1e-12:
                break

            # Collect the full atomic batch (all events inside the burst window).
            batch: list[RandomEvent] = []
            window_cutoff = event.observed_at + burst_window + 1e-12
            while self.next_event_index < len(self._observation_order):
                candidate_index = self._observation_order[self.next_event_index]
                candidate = self.event_tape.events[candidate_index]
                if candidate.observed_at > self.current_time + 1e-12 and candidate.observed_at > window_cutoff:
                    break
                if candidate.observed_at <= self.current_time + 1e-12 or burst_window > 0:
                    self.next_event_index += 1
                    batch.append(candidate)
                else:
                    break

            version_before = self.graph_version
            batch_state_changed = False
            for item in batch:
                if self.runtime_bridge is not None:
                    # Phase C: truth -> observation -> confirmation -> belief.
                    # ONLY confirmed events mutate env state; unconfirmed events
                    # must NOT change belief, mask, or trigger rescheduling.
                    truth_event = self._random_event_to_truth_event(item)
                    confirmed = self.runtime_bridge.ingest_truth_event(
                        truth_event, self.current_time,
                        observation_delay=item.observed_at - item.occurred_at,
                    )
                    if confirmed is not None:
                        changed = self.runtime_bridge.apply_confirmed_to_env(self, confirmed)
                        batch_state_changed = batch_state_changed or changed
                        self._record_confirmed_event(item)
                else:
                    # Legacy fallback: direct application without bridge.
                    self._apply_random_event(item, increment_version=False)
                ingested.append(item.event_id)

            # Atomic commit: graph_version only increments when at least one
            # confirmed event actually changed belief/decision-relevant state.
            # Unconfirmed observations must NEVER change graph_version (audit item 5).
            if batch and batch_state_changed:
                self.graph_version = version_before + 1
        return ingested

    def _record_confirmed_event(self, event: RandomEvent) -> None:
        """Create an EventRuntime record for a bridge-confirmed event.

        Mirrors the bookkeeping inside ``_apply_random_event`` (status, actual
        affected regions, event queue membership) without double-applying the
        state mutation that the bridge already performed.
        """

        actual = {rid for rid in event.affected_regions}
        runtime = EventRuntime(
            event=event,
            actual_affected_regions=tuple(sorted(actual)),
            status="pending" if actual else "resolved",
            applied_at=self.current_time,
            resolved_at=None if actual else self.current_time,
            application_note="confirmed-via-bridge",
        )
        self.event_records[event.event_id] = runtime
        if actual:
            self.event_queue.append(event.event_id)
        self.last_event = event
        self.current_event = self._legacy_event_view(event, actual)

    def _random_event_to_truth_event(self, event: RandomEvent) -> TruthEvent:
        """Convert a RandomEvent tape event to a TruthEvent for the bridge."""
        from event_runtime.events import EventType as RuntimeEventType, TruthEvent
        type_map = {
            RandomEventType.UAV_DAMAGE: RuntimeEventType.UAV_DAMAGE,
            RandomEventType.TARGET_DISCOVERED: RuntimeEventType.TARGET_DISCOVERED,
            RandomEventType.TARGET_DESTROYED: RuntimeEventType.TARGET_DESTROYED,
            RandomEventType.REGION_VACANCY: RuntimeEventType.REGION_VACANCY,
        }
        return TruthEvent(
            event_id=event.event_id,
            event_type=type_map[event.event_type],
            source_event=event.source_event,
            occurred_at=event.occurred_at,
            affected_uavs=event.affected_uavs,
            affected_regions=event.affected_regions,
            affected_targets=event.affected_targets,
            severity=event.severity,
            event_seed=event.event_seed,
            state_version=event.state_version,
        )

    def _apply_random_event(self, event: RandomEvent, *, increment_version: bool = True) -> None:
        actual: set[int] = set()
        note = "applied"
        kind = event.event_type

        if kind is RandomEventType.UAV_DAMAGE:
            uid = int(event.affected_uavs[0])
            uav = self.uavs[uid]
            actual.update(uav.regions)
            if uav.alive:
                if uav.task == TaskType.TRACK and uav.target_id != NO_TARGET:
                    target = self.targets[uav.target_id]
                    target.tracked = False
                    target.tracker_id = NO_UAV
                uav.alive = False
                uav.task = TaskType.IDLE
                uav.target_id = NO_TARGET
                for rid in tuple(uav.regions):
                    self._clear_region_assignment(rid)
            else:
                note = "idempotent: UAV already unavailable"

        elif kind is RandomEventType.TARGET_DISCOVERED:
            tid = int(event.affected_targets[0])
            uid = int(event.affected_uavs[0])
            target, uav = self.targets[tid], self.uavs[uid]
            if target.destroyed or not uav.alive:
                note = "temporarily infeasible discovery subject"
            else:
                actual.update(uav.regions)
                for rid in tuple(uav.regions):
                    self._clear_region_assignment(rid)
                target.discovered = True
                target.tracked = True
                target.tracker_id = uid
                uav.task = TaskType.TRACK
                uav.target_id = tid

        elif kind is RandomEventType.TARGET_DESTROYED:
            tid = int(event.affected_targets[0])
            target = self.targets[tid]
            uid = int(target.tracker_id)
            target.discovered = True
            target.destroyed = True
            target.tracked = False
            target.tracker_id = NO_UAV
            if uid != NO_UAV and self.uavs[uid].alive:
                self.uavs[uid].task = TaskType.IDLE
                self.uavs[uid].target_id = NO_TARGET
            else:
                note = "idempotent: no live tracker to release"

        elif kind is RandomEventType.REGION_VACANCY:
            rid = int(event.affected_regions[0])
            actual.add(rid)
            self._clear_region_assignment(rid)
        else:  # pragma: no cover - enum guards this path
            raise ValueError(f"unsupported event {kind}")

        for rid in actual:
            self.regions[rid].need_reassign = True
            self.pending_regions.add(rid)
            self.vacancy_duration.setdefault(rid, 0.0)

        runtime = EventRuntime(
            event=event,
            actual_affected_regions=tuple(sorted(actual)),
            status="pending" if actual else "resolved",
            applied_at=self.current_time,
            resolved_at=None if actual else self.current_time,
            application_note=note,
        )
        self.event_records[event.event_id] = runtime
        if actual:
            self.event_queue.append(event.event_id)
        self.last_event = event
        self.current_event = self._legacy_event_view(event, actual)
        if increment_version:
            self.graph_version += 1
        self.communication_trigger_count += 1
        self.communication_bytes += len(json.dumps(event.to_dict(), sort_keys=True).encode("utf-8"))

    def _legacy_event_view(self, event: RandomEvent, actual: set[int]) -> LegacyEvent:
        if event.event_type is RandomEventType.UAV_DAMAGE:
            return LegacyEvent(EventType.UAV_DAMAGE, sorted(actual), damaged_uav=event.affected_uavs[0])
        if event.event_type is RandomEventType.TARGET_DESTROYED:
            released = event.affected_uavs[0] if event.affected_uavs else NO_UAV
            return LegacyEvent(EventType.TARGET_DESTROYED, [], released_uav=released)
        return LegacyEvent(EventType.REGION_VACANCY, sorted(actual), description=event.event_type.value)

    # ------------------------------------------------------------------
    # Versioned edge decisions
    # ------------------------------------------------------------------

    def begin_decision(self) -> DecisionContext:
        """Snapshot the current decision context.

        Returns a frozen ``DecisionContext`` carrying the graph, graph_version,
        and action_version at this instant.  The caller passes the context back
        to ``submit_action`` so the execution layer can detect stale or
        future-version submissions.
        """
        graph = build_graph_state(self)
        return DecisionContext(
            graph=graph,
            graph_version=int(self.graph_version),
            action_version=int(self.decision_version),
        )

    def submit_action(
        self,
        action: int | ActionSubmission,
        expected_graph_version: int | None = None,
        expected_action_version: int | None = None,
        *,
        strict: bool = False,
    ):
        """Submit a decision with staleness guards.

        Accepts either an ``ActionSubmission`` (preferred) or separate arguments.

        ``expected_graph_version`` must match ``self.graph_version`` exactly.
        If ``expected_action_version`` is provided, it must also match
        ``self.decision_version`` exactly.  A mismatch on either count causes
        an immediate reject (reward=0, no state mutation, no PPO transition).

        ``strict=True`` raises ``StaleDecisionError`` instead of returning a
        zero-reward tuple.
        """
        if isinstance(action, ActionSubmission):
            sub = action
            action_int = sub.action
            gv = sub.graph_version
            expected_action_version = sub.action_version
        else:
            action_int = int(action)
            gv = int(expected_graph_version) if expected_graph_version is not None else self.graph_version
        if gv != self.graph_version:
            self.stale_rejection_count += 1
            if strict:
                raise StaleDecisionError(
                    f"decision used graph v{gv}, current graph is v{self.graph_version}"
                )
            graph = build_graph_state(self)
            info = self._info(graph, stale_decision=True, rejected_action=action_int)
            return graph, 0.0, False, False, info
        if expected_action_version is not None and int(expected_action_version) != self.decision_version:
            self.stale_rejection_count += 1
            if strict:
                raise StaleDecisionError(
                    f"decision used action_version={expected_action_version}, current is {self.decision_version}"
                )
            graph = build_graph_state(self)
            info = self._info(graph, stale_decision=True, rejected_action=action_int)
            return graph, 0.0, False, False, info
        return self._step_current(action_int, expected_graph_version=gv, expected_action_version=int(expected_action_version) if expected_action_version is not None else None)

    def step(self, action):
        if isinstance(action, Mapping):
            return self.submit_action(
                int(action["action"]),
                int(action.get("graph_version", self.graph_version)),
                expected_action_version=action.get("action_version"),
            )
        return self._step_current(int(action))

    def _step_current(self, action: int, *, expected_graph_version: int | None = None, expected_action_version: int | None = None):
        graph_before = build_graph_state(self)
        before_assignments = assignment_map(self)
        before_cost = compute_cost(self, self.cost_weights, reference_assignments=before_assignments)

        repaired = action
        invalid = not (0 <= action < graph_before.num_actions and bool(graph_before.action_mask[action]))
        if invalid:
            legal = np.flatnonzero(graph_before.action_mask.cpu().numpy())
            repaired = int(legal[0])
            self.repair_count += 1

        # Decision latency accumulates for currently vacant regions.  Events
        # are not ingested until after commit in this atomic step; callers that
        # need true in-flight arrival use begin_decision/advance_time/submit.
        for rid in self.pending_regions:
            self.vacancy_duration[rid] = self.vacancy_duration.get(rid, 0.0) + self.decision_duration

        decoded = decode_edge_action(graph_before, repaired)
        if decoded is not None:
            uid, rid = decoded
            # Phase E: command/ACK/lease/fencing lifecycle at the execution
            # layer. A rejected command must not mutate the environment.
            command_accepted = True
            if self.runtime_bridge is not None:
                bridge = self.runtime_bridge
                command_accepted = bridge.issue_assignment_command(
                    self, uid, rid, self.current_time,
                    expected_graph_version=expected_graph_version,
                    expected_action_version=expected_action_version,
                ) is not None
                if not command_accepted:
                    self.stale_rejection_count += 1
                    repaired = self.noop_action
            if command_accepted:
                self._assign_region_to_uav(rid, uid)
                if self._valid_search_assign(uid, rid):
                    self.pending_regions.discard(rid)
                    self.regions[rid].need_reassign = False

        self.decision_step += 1
        self.decision_version += 1
        self.graph_version += 1
        self._resolve_completed_events()
        after_cost = compute_cost(self, self.cost_weights, reference_assignments=before_assignments)
        reward, reward_trace = cost_difference_reward(before_cost, after_cost)

        self.current_time += self.decision_duration
        new_events = self._ingest_observed_events()
        if not self.pending_regions and self.next_event_index < len(self._observation_order):
            assert self.event_tape is not None
            tape_index = self._observation_order[self.next_event_index]
            next_time = self.event_tape.events[tape_index].observed_at
            if next_time > self.current_time:
                self.current_time = next_time
            new_events.extend(self._ingest_observed_events())

        final_infeasible = self._is_final_infeasible()
        terminated = self._episode_complete() or final_infeasible
        truncated = (not terminated) and (
            self.decision_step >= self.max_decision_steps or self.current_time >= self.max_time
        )
        graph_after = build_graph_state(self)
        info = self._info(
            graph_after,
            raw_action=action,
            repaired_action=repaired,
            invalid_action=invalid,
            reward_trace=reward_trace,
            new_events=new_events,
            temporary_infeasible=self._is_temporarily_infeasible(),
            final_infeasible=final_infeasible,
        )
        return graph_after, reward, terminated, truncated, info

    def _resolve_completed_events(self) -> None:
        still_pending: list[str] = []
        for event_id in self.event_queue:
            runtime = self.event_records[event_id]
            resolved = all(
                self.regions[rid].assigned_uav != NO_UAV
                and self._valid_search_assign(self.regions[rid].assigned_uav, rid)
                for rid in runtime.actual_affected_regions
            )
            if resolved:
                runtime.status = "resolved"
                runtime.resolved_at = self.current_time + self.decision_duration
            else:
                still_pending.append(event_id)
        self.event_queue = still_pending

    def _no_legal_edge(self) -> bool:
        graph = build_graph_state(self)
        return not bool(graph.action_mask[:-1].any())

    def _future_release_exists(self) -> bool:
        assert self.event_tape is not None
        return any(
            self.event_tape.events[tape_index].event_type is RandomEventType.TARGET_DESTROYED
            for tape_index in self._observation_order[self.next_event_index :]
        )

    def _is_temporarily_infeasible(self) -> bool:
        return bool(self.pending_regions and self._no_legal_edge() and self._future_release_exists())

    def _is_final_infeasible(self) -> bool:
        return bool(self.pending_regions and self._no_legal_edge() and not self._future_release_exists())

    def _episode_complete(self) -> bool:
        assert self.event_tape is not None
        return self.next_event_index >= len(self._observation_order) and not self.pending_regions and not self.event_queue

    def _info(self, graph: HeteroGraphState, **extra) -> dict[str, Any]:
        info = {
            "graph_version": self.graph_version,
            "decision_version": self.decision_version,
            "current_time": self.current_time,
            "pending_regions": sorted(self.pending_regions),
            "event_queue": list(self.event_queue),
            "event_index": self.next_event_index,
            "events_total": 0 if self.event_tape is None else len(self.event_tape.events),
            "action_mask": graph.action_mask.cpu().numpy(),
            "repair_count": self.repair_count,
            "stale_rejection_count": self.stale_rejection_count,
            "communication_trigger_count": self.communication_trigger_count,
            "communication_bytes": self.communication_bytes,
            "event_records": {key: value.to_dict() for key, value in self.event_records.items()},
        }
        info.update(extra)
        return info

    def snapshot(self) -> dict[str, Any]:
        base = super().snapshot()
        base["random_event"] = {
            "current_time": self.current_time,
            "graph_version": self.graph_version,
            "decision_version": self.decision_version,
            "pending_regions": sorted(self.pending_regions),
            "event_queue": list(self.event_queue),
            "event_tape": None if self.event_tape is None else self.event_tape.to_dict(),
            "event_records": {key: value.to_dict() for key, value in self.event_records.items()},
        }
        return base

    def legacy_observation(self) -> np.ndarray:
        """Return the original 165-D observation for the unchanged MLP model."""

        return self._get_obs()
