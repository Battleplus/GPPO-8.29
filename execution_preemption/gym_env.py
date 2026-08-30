"""Gymnasium rollout shell for the Execution-Preemption V1 contract."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .adapter import (
    ACTION_CAPACITY,
    FLAT_OBSERVATION_DIMENSION,
    FlatPolicyObservation,
    HeteroPolicyObservation,
    build_flat_observation,
    build_hetero_observation,
    proposal_from_policy_action,
)
from .controller import PreemptionController
from .graph import build_execution_graph
from .metrics import ExecutionMetricAccumulator
from .models import EventPriority, RuntimeEvent, RuntimeEventType, TaskState
from .reward import TransitionSignals, compute_transition_reward
from .runtime import ExecutionRuntime, PendingEventBatchTransaction
from .signals import derive_transition_signals
from .tapes import runtime_from_tape, validate_tape


def _runtime_event(item: Mapping[str, Any]) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=str(item["event_id"]),
        event_type=RuntimeEventType(item["event_type"]),
        priority=EventPriority[str(item["priority"])],
        occurred_at=float(item["occurred_at"]),
        received_at=float(item["received_at"]),
        task_id=item.get("task_id"),
        uav_id=item.get("uav_id"),
        task_priority=int(item.get("task_priority", 0)),
        deadline=item.get("deadline"),
        confidence=float(item.get("confidence", 1.0)),
        payload=item.get("payload", {}),
    )


def runtime_event_batches(tape: Mapping[str, Any]) -> tuple[tuple[RuntimeEvent, ...], ...]:
    validate_tape(tape)
    grouped: list[list[RuntimeEvent]] = []
    indexes: dict[str, int] = {}
    for item in tape["events"]:
        batch_id = str(item["batch_id"])
        if batch_id not in indexes:
            indexes[batch_id] = len(grouped)
            grouped.append([])
        grouped[indexes[batch_id]].append(_runtime_event(item))
    return tuple(tuple(items) for items in grouped)


class ExecutionPreemptionGymEnv(gym.Env):
    """Replay one frozen tape through request-level policy actions.

    The public Gym observation is the fixed PPO vector.  The matching typed
    heterograph and immutable adapter objects are exposed through ``info`` for
    GPPO/Planner.  Multiple actions inside one event batch operate on staged
    state; the live runtime changes only after the final proposal succeeds.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        tape: Mapping[str, Any],
        *,
        allocator_id: str = "framework_rollout_v1",
        controller: PreemptionController | None = None,
        expose_rule_context: bool = True,
    ) -> None:
        super().__init__()
        validate_tape(tape)
        self.tape = copy.deepcopy(dict(tape))
        self.allocator_id = str(allocator_id)
        if not self.allocator_id:
            raise ValueError("allocator_id is required")
        self.controller = controller or PreemptionController()
        self.expose_rule_context = bool(expose_rule_context)
        self.action_space = spaces.Discrete(ACTION_CAPACITY)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(FLAT_OBSERVATION_DIMENSION,),
            dtype=np.float32,
        )
        self.runtime: ExecutionRuntime | None = None
        self._batches: tuple[tuple[RuntimeEvent, ...], ...] = ()
        self._batch_index = 0
        self._pending: PendingEventBatchTransaction | None = None
        self._batch_before: ExecutionRuntime | None = None
        self._flat: FlatPolicyObservation | None = None
        self._hetero: HeteroPolicyObservation | None = None
        self._terminated = False
        self._policy_action_count = 0
        self._metrics: ExecutionMetricAccumulator | None = None
        self._last_inference_latency_ms: float | None = None
        self._last_reward_trace: dict[str, object] | None = None

    @property
    def current_time(self) -> float:
        if self._pending is not None:
            return self._pending.now
        if self._batch_index:
            return max(item.received_at for item in self._batches[self._batch_index - 1])
        return 0.0

    def _terminal_observations(self) -> tuple[FlatPolicyObservation, HeteroPolicyObservation]:
        assert self.runtime is not None
        graph = build_execution_graph(self.runtime, now=self.current_time)
        return build_flat_observation(graph), build_hetero_observation(graph)

    def _prepare_pending_observations(self) -> None:
        assert self._pending is not None and self._pending.awaiting_allocation
        staged = self._pending.staged_runtime_copy()
        graph = build_execution_graph(
            staged,
            now=self._pending.now,
            events=self._pending.events,
            allocation_request=self._pending.allocation_request,
        )
        self._flat = build_flat_observation(
            graph,
            request=self._pending.allocation_request,
            decision=self._pending.decision if self.expose_rule_context else None,
        )
        self._hetero = build_hetero_observation(
            graph,
            request=self._pending.allocation_request,
            decision=self._pending.decision if self.expose_rule_context else None,
        )

    def _advance_until_request_or_terminal(self) -> None:
        assert self.runtime is not None
        while self._batch_index < len(self._batches):
            events = self._batches[self._batch_index]
            now = max(item.received_at for item in events)
            self._batch_before = copy.deepcopy(self.runtime)
            pending = self.runtime.begin_event_batch_transaction(
                events, self.controller, now=now
            )
            self._pending = pending
            if pending.awaiting_allocation:
                self._prepare_pending_observations()
                return
            self.runtime.commit_event_batch_transaction(pending)
            self._record_event_outcomes(pending, inference_latency_ms=None)
            self._batch_index += 1
            self._pending = None
            self._batch_before = None
        self._terminated = True
        self._flat, self._hetero = self._terminal_observations()
        assert self._metrics is not None
        for task in self.runtime.tasks.values():
            self._metrics.record_task_outcome(
                starved=task.state in {TaskState.PENDING, TaskState.PAUSED, TaskState.PREEMPTED}
            )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        case_seed = int(self.tape["case_seed"])
        if seed is not None and int(seed) != case_seed:
            raise ValueError("frozen tape reset seed must equal tape case_seed")
        self.runtime = runtime_from_tape(self.tape)
        self._batches = runtime_event_batches(self.tape)
        self._batch_index = 0
        self._pending = None
        self._batch_before = None
        self._flat = None
        self._hetero = None
        self._terminated = False
        self._policy_action_count = 0
        self._last_inference_latency_ms = None
        self._last_reward_trace = None
        self._metrics = ExecutionMetricAccumulator(
            self.allocator_id,
            str(self.tape["tape_id"]),
            len(self.runtime.uavs),
        )
        self._advance_until_request_or_terminal()
        assert self._flat is not None
        return np.asarray(self._flat.vector, dtype=np.float32), self._info(reset=True)

    def action_masks(self) -> np.ndarray:
        if self._flat is None:
            raise RuntimeError("environment must be reset before requesting mask")
        return np.asarray(self._flat.action_space.mask, dtype=np.bool_)

    def record_inference_latency(self, milliseconds: float) -> None:
        value = float(milliseconds)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("inference latency must be finite and non-negative")
        self._last_inference_latency_ms = value

    def _record_event_outcomes(
        self,
        pending: PendingEventBatchTransaction,
        *,
        inference_latency_ms: float | None,
    ) -> None:
        assert self.runtime is not None and self._metrics is not None
        decision_by_event = {item.event_id: item for item in pending.decisions}
        for event in pending.events:
            decision = decision_by_event[event.event_id]
            urgent = event.priority <= EventPriority.P1
            task = self.runtime.tasks.get(event.task_id or "")
            deadline_missed = bool(
                urgent
                and event.deadline is not None
                and pending.now > event.deadline
                and (task is None or task.state is not TaskState.COMPLETED)
            )
            p0 = event.priority is EventPriority.P0
            self._metrics.record_event(
                urgent=urgent,
                deadline_missed=deadline_missed,
                p0=p0,
                p0_handled=p0,
            )
            if decision.displaced_task_id is not None:
                displaced = self.runtime.tasks.get(decision.displaced_task_id)
                resumed = bool(displaced is not None and displaced.state is TaskState.RUNNING)
                self._metrics.record_displacement(
                    resumed=resumed,
                    recovery_latency=(
                        self.runtime.progress_policy.switch_time_cost if resumed else None
                    ),
                )

    def step_with_metadata(self, action: int, *, inference_latency_ms: float | None = None):
        if inference_latency_ms is not None:
            self.record_inference_latency(inference_latency_ms)
        return self.step(action)

    def step(self, action: int):
        if self._terminated:
            raise RuntimeError("cannot step a terminated tape; call reset")
        if self._pending is None or not self._pending.awaiting_allocation:
            raise RuntimeError("environment is not awaiting a policy action")
        assert self.runtime is not None and self._flat is not None and self._metrics is not None
        request = self._pending.allocation_request
        graph_sha = self._flat.graph_sha256
        proposal = proposal_from_policy_action(
            request,
            self._flat.action_space,
            int(action),
            allocator_id=self.allocator_id,
            current_graph_version=self._pending.graph_version_after,
            current_graph_sha256=graph_sha,
        )
        self.runtime.submit_event_batch_proposal(self._pending, proposal)
        self._policy_action_count += 1
        latency = self._last_inference_latency_ms
        self._last_inference_latency_ms = None

        if self._pending.awaiting_allocation:
            self._metrics.record_transition(
                TransitionSignals(), inference_latency_ms=latency
            )
            self._prepare_pending_observations()
            assert self._flat is not None
            info = self._info(reward_deferred=True)
            return np.asarray(self._flat.vector, dtype=np.float32), 0.0, False, False, info

        assert self._batch_before is not None
        committed = self._pending
        self.runtime.commit_event_batch_transaction(committed)
        signals = derive_transition_signals(
            self._batch_before,
            self.runtime,
            now=committed.now,
        )
        reward = compute_transition_reward(signals)
        self._last_reward_trace = reward.to_dict()
        self._metrics.record_transition(
            signals,
            inference_latency_ms=latency,
            preemption_response_latency=self.runtime.progress_policy.switch_time_cost,
        )
        self._record_event_outcomes(committed, inference_latency_ms=latency)
        self._batch_index += 1
        self._pending = None
        self._batch_before = None
        self._advance_until_request_or_terminal()
        assert self._flat is not None
        info = self._info(reward_deferred=False, reward_trace=reward.to_dict())
        return (
            np.asarray(self._flat.vector, dtype=np.float32),
            float(reward.reward),
            self._terminated,
            False,
            info,
        )

    def _info(self, **extra) -> dict[str, object]:
        assert self.runtime is not None and self._flat is not None and self._hetero is not None
        value: dict[str, object] = {
            "contract_id": "execution-preemption-training-v1",
            "tape_id": str(self.tape["tape_id"]),
            "scenario_id": str(self.tape["scenario_id"]),
            "graph_version": self._flat.graph_version,
            "graph_sha256": self._flat.graph_sha256,
            "flat_observation": self._flat,
            "hetero_observation": self._hetero,
            "action_mask": self.action_masks(),
            "policy_action_count": self._policy_action_count,
            "rule_context_exposed": self.expose_rule_context,
            "batch_index": self._batch_index,
            "batch_count": len(self._batches),
            "episode_terminated": self._terminated,
            "live_runtime_sha256": self.runtime.state_sha256(),
            "training_allowed": False,
            "training_started": False,
            "validation_started": False,
            "freeze_started": False,
            "test_started": False,
            "hidden_evaluation_started": False,
        }
        if self._terminated and self._metrics is not None:
            value["episode_metrics"] = self._metrics.finalize().to_dict()
        value.update(extra)
        return value


__all__ = ["ExecutionPreemptionGymEnv", "runtime_event_batches"]
