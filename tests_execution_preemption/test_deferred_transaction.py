from __future__ import annotations

import unittest

from execution_preemption.adapter import (
    AdapterValidationError,
    build_flat_observation,
    proposal_from_policy_action,
)
from execution_preemption.controller import PreemptionController
from execution_preemption.allocation import MaxEnergyMarginAllocator
from execution_preemption.graph import build_execution_graph
from execution_preemption.models import (
    DecisionType,
    EventPriority,
    RuntimeEvent,
    RuntimeEventType,
    TaskRuntime,
    UAVAvailability,
    UAVRuntime,
)
from execution_preemption.runtime import ExecutionRuntime, StaleExecutionCommand


class DeferredEventTransactionTests(unittest.TestCase):
    @staticmethod
    def idle_runtime() -> ExecutionRuntime:
        runtime = ExecutionRuntime()
        runtime.add_uav(UAVRuntime("U0", energy_ratio=0.9))
        runtime.add_uav(UAVRuntime("U1", energy_ratio=0.8))
        runtime.add_task(TaskRuntime("T0", "SEARCH", 50, 20.0))
        return runtime

    @staticmethod
    def arrival() -> RuntimeEvent:
        return RuntimeEvent(
            "E0",
            RuntimeEventType.TASK_ARRIVAL,
            EventPriority.P3,
            occurred_at=1.0,
            received_at=1.0,
            task_id="T0",
            task_priority=50,
            deadline=20.0,
        )

    def test_begin_is_read_only_and_commit_uses_policy_proposal(self) -> None:
        runtime = self.idle_runtime()
        before_sha = runtime.state_sha256()
        pending = runtime.begin_event_transaction(
            self.arrival(), PreemptionController(), now=1.0
        )
        self.assertEqual(runtime.state_sha256(), before_sha)
        self.assertEqual(runtime.graph_version, 0)
        self.assertEqual(pending.graph_version_after, 1)
        self.assertIsNotNone(pending.allocation_request)
        self.assertIsNone(pending.decision.selected_uav)

        staged = pending.staged_runtime_copy()
        graph = build_execution_graph(
            staged,
            now=1.0,
            events=[pending.event],
            allocation_request=pending.allocation_request,
        )
        observation = build_flat_observation(
            graph,
            request=pending.allocation_request,
            decision=pending.decision,
        )
        action = observation.action_space.bindings.index(("U1", "T0"))
        proposal = proposal_from_policy_action(
            pending.allocation_request,
            observation.action_space,
            action,
            allocator_id="ppo_mlp_rule_arbiter_v1",
            current_graph_version=pending.graph_version_after,
            current_graph_sha256=graph.sha256(),
        )
        result = runtime.commit_event_transaction(pending, proposal=proposal, now=1.0)
        self.assertEqual(result.graph_version_after, 1)
        self.assertEqual(runtime.tasks["T0"].assigned_uav, "U1")
        self.assertEqual(runtime.decision_log[-1].allocator_id, "ppo_mlp_rule_arbiter_v1")
        runtime.validate_invariants()

    def test_migration_request_is_overlaid_on_running_task_graph(self) -> None:
        runtime = ExecutionRuntime()
        runtime.add_uav(UAVRuntime("FAILED", energy_ratio=0.9))
        runtime.add_uav(UAVRuntime("REPLACEMENT", energy_ratio=0.8))
        runtime.add_task(TaskRuntime("RUN", "SEARCH", 50, 20.0, progress=0.4, remaining_work=0.6))
        runtime.assign_task("RUN", "FAILED", at=0.0)
        event = RuntimeEvent(
            "E-failure",
            RuntimeEventType.EXECUTION_FAILURE,
            EventPriority.P0,
            occurred_at=1.0,
            received_at=1.0,
            task_id="RUN",
            uav_id="FAILED",
        )
        pending = runtime.begin_event_transaction(event, PreemptionController(), now=1.0)
        self.assertEqual(pending.decision.decision, DecisionType.MIGRATE)
        self.assertEqual(pending.allocation_request.candidate_uav_ids, ("REPLACEMENT",))
        staged = pending.staged_runtime_copy()
        plain = build_execution_graph(staged, now=1.0, events=[event])
        self.assertNotIn(("REPLACEMENT", "RUN"), plain.action_candidates)
        graph = build_execution_graph(
            staged,
            now=1.0,
            events=[event],
            allocation_request=pending.allocation_request,
        )
        self.assertIn(("REPLACEMENT", "RUN"), graph.action_candidates)
        observation = build_flat_observation(graph, request=pending.allocation_request,
                                             decision=pending.decision)
        action = observation.action_space.bindings.index(("REPLACEMENT", "RUN"))
        proposal = proposal_from_policy_action(
            pending.allocation_request,
            observation.action_space,
            action,
            allocator_id="gppo_adaptive_rule_arbiter_v1",
            current_graph_version=pending.graph_version_after,
            current_graph_sha256=graph.sha256(),
        )
        runtime.commit_event_transaction(pending, proposal=proposal, now=1.0)
        self.assertEqual(runtime.tasks["RUN"].assigned_uav, "REPLACEMENT")
        self.assertEqual(runtime.uavs["FAILED"].availability, UAVAvailability.FAILED)
        self.assertAlmostEqual(runtime.tasks["RUN"].progress, 0.36)

    def test_progress_during_inference_rejects_pending_transaction(self) -> None:
        runtime = self.idle_runtime()
        pending = runtime.begin_event_transaction(self.arrival(), PreemptionController(), now=1.0)
        runtime.assign_task("T0", "U0", at=1.0)
        runtime.advance(1.0, now=2.0, work_rate_by_task={"T0": 0.1})
        with self.assertRaisesRegex(StaleExecutionCommand, "changed while policy"):
            runtime.commit_event_transaction(pending, proposal=None, now=2.0)

    def test_second_transaction_from_same_source_is_stale_after_first_commit(self) -> None:
        runtime = self.idle_runtime()
        first = runtime.begin_event_transaction(self.arrival(), PreemptionController(), now=1.0)
        second = runtime.begin_event_transaction(self.arrival(), PreemptionController(), now=1.0)
        request = first.allocation_request
        staged = first.staged_runtime_copy()
        graph = build_execution_graph(staged, now=1.0, allocation_request=request)
        observation = build_flat_observation(graph, request=request, decision=first.decision)
        action = next(index for index, enabled in enumerate(observation.action_space.mask)
                      if enabled and index)
        proposal = proposal_from_policy_action(
            request,
            observation.action_space,
            action,
            allocator_id="test",
            current_graph_version=first.graph_version_after,
            current_graph_sha256=graph.sha256(),
        )
        runtime.commit_event_transaction(first, proposal=proposal, now=1.0)
        with self.assertRaisesRegex(StaleExecutionCommand, "changed while policy"):
            runtime.commit_event_transaction(second, proposal=proposal, now=1.0)

    def test_direct_safety_plan_commits_without_policy(self) -> None:
        runtime = self.idle_runtime()
        runtime.assign_task("T0", "U0", at=0.0)
        event = RuntimeEvent(
            "E-energy",
            RuntimeEventType.UAV_LOW_ENERGY,
            EventPriority.P2,
            occurred_at=1.0,
            received_at=1.0,
            uav_id="U0",
        )
        pending = runtime.begin_event_transaction(event, PreemptionController(), now=1.0)
        self.assertIsNone(pending.allocation_request)
        self.assertEqual(pending.decision.decision, DecisionType.RTB)
        runtime.commit_event_transaction(pending, now=1.0)
        self.assertEqual(runtime.uavs["U0"].availability, UAVAvailability.RETURNING)
        self.assertIsNone(runtime.tasks["T0"].assigned_uav)

    def test_request_overlay_rejects_wrong_version(self) -> None:
        runtime = self.idle_runtime()
        pending = runtime.begin_event_transaction(self.arrival(), PreemptionController(), now=1.0)
        staged = pending.staged_runtime_copy()
        staged.graph_version += 1
        with self.assertRaisesRegex(ValueError, "graph_version"):
            build_execution_graph(staged, now=1.0, allocation_request=pending.allocation_request)

    def test_atomic_batch_can_pause_for_two_policy_actions_then_commit_once(self) -> None:
        runtime = ExecutionRuntime()
        runtime.add_uav(UAVRuntime("U0", energy_ratio=0.9, supported_task_types=frozenset({"URGENT"})))
        runtime.add_uav(UAVRuntime("U1", energy_ratio=0.8, supported_task_types=frozenset({"URGENT"})))
        runtime.add_task(TaskRuntime("A", "URGENT", 80, 10.0))
        runtime.add_task(TaskRuntime("B", "URGENT", 90, 5.0))
        events = [
            RuntimeEvent(
                "E-A", RuntimeEventType.TASK_ARRIVAL, EventPriority.P1,
                occurred_at=1.0, received_at=1.0, task_id="A",
                task_priority=80, deadline=10.0,
            ),
            RuntimeEvent(
                "E-B", RuntimeEventType.TASK_ARRIVAL, EventPriority.P1,
                occurred_at=1.0, received_at=1.0, task_id="B",
                task_priority=90, deadline=5.0,
            ),
        ]
        before = runtime.state_sha256()
        pending = runtime.begin_event_batch_transaction(events, PreemptionController(), now=1.0)
        self.assertTrue(pending.awaiting_allocation)
        self.assertEqual(pending.decision.event_id, "E-B")
        self.assertEqual(runtime.state_sha256(), before)

        allocator = MaxEnergyMarginAllocator()
        first_proposal = allocator.propose(pending.allocation_request)
        runtime.submit_event_batch_proposal(pending, first_proposal)
        self.assertTrue(pending.awaiting_allocation)
        self.assertEqual(pending.decision.event_id, "E-A")
        self.assertEqual(pending.allocation_request.candidate_uav_ids, ("U1",))
        self.assertEqual(runtime.state_sha256(), before)

        runtime.submit_event_batch_proposal(
            pending, allocator.propose(pending.allocation_request)
        )
        self.assertTrue(pending.complete)
        self.assertEqual(runtime.state_sha256(), before)
        result = runtime.commit_event_batch_transaction(pending)
        self.assertEqual(result.graph_version_before, 0)
        self.assertEqual(result.graph_version_after, 1)
        self.assertEqual([item.event_id for item in result.decisions], ["E-B", "E-A"])
        self.assertEqual(runtime.tasks["B"].assigned_uav, "U0")
        self.assertEqual(runtime.tasks["A"].assigned_uav, "U1")
        runtime.validate_invariants()

    def test_incomplete_batch_cannot_commit_and_live_change_rejects_submit(self) -> None:
        runtime = self.idle_runtime()
        pending = runtime.begin_event_batch_transaction(
            [self.arrival()], PreemptionController(), now=1.0
        )
        with self.assertRaisesRegex(ValueError, "still awaits"):
            runtime.commit_event_batch_transaction(pending)
        runtime.uavs["U0"].energy_ratio = 0.7
        with self.assertRaisesRegex(StaleExecutionCommand, "changed while batch"):
            runtime.submit_event_batch_proposal(
                pending, MaxEnergyMarginAllocator().propose(pending.allocation_request)
            )


if __name__ == "__main__":
    unittest.main()
