from __future__ import annotations

import json
from pathlib import Path
import unittest

from execution_preemption.adapter import (
    ACTION_CAPACITY,
    ADAPTER_ID,
    FLAT_OBSERVATION_DIMENSION,
    AdapterValidationError,
    adapter_layout_sha256,
    build_flat_observation,
    build_hetero_observation,
    decode_policy_action,
    proposal_from_policy_action,
)
from execution_preemption.allocation import (
    AllocationValidationError,
    build_allocation_request,
)
from execution_preemption.graph import build_execution_graph
from execution_preemption.models import (
    DecisionType,
    EventDecision,
    EventPriority,
    TaskRuntime,
    UAVRuntime,
)
from execution_preemption.runtime import ExecutionRuntime


ROOT = Path(__file__).resolve().parents[1]


class PolicyAdapterTests(unittest.TestCase):
    @staticmethod
    def runtime(uav_count: int = 4, tasks_per_uav: int = 2) -> ExecutionRuntime:
        runtime = ExecutionRuntime()
        for index in range(uav_count):
            runtime.add_uav(UAVRuntime(
                f"U{index:02d}",
                energy_ratio=0.9 - index * 0.001,
                supported_task_types=frozenset({"SEARCH"}),
            ))
        for index in range(uav_count * tasks_per_uav):
            runtime.add_task(TaskRuntime(
                f"T{index:03d}",
                "SEARCH",
                priority=10 + index,
                deadline=100.0 + index,
            ))
        return runtime

    def test_config_matches_frozen_dimensions_and_layout_hash(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "execution_policy_adapter_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["adapter_id"], ADAPTER_ID)
        self.assertEqual(config["flat_observation"]["dimension"], FLAT_OBSERVATION_DIMENSION)
        self.assertEqual(config["flat_observation"]["layout_sha256"], adapter_layout_sha256())
        self.assertEqual(config["action"]["capacity"], ACTION_CAPACITY)
        self.assertTrue(config["action"]["noop_masked_during_required_allocation"])
        self.assertFalse(config["training_allowed"])
        self.assertFalse(config["model_weights_loaded_in_this_stage"])
        self.assertTrue(config["deferred_atomic_transaction"]["multiple_policy_requests_per_event_batch"])
        self.assertEqual(config["deferred_atomic_transaction"]["graph_version_increment_per_batch"], 1)
        self.assertTrue(config["deferred_atomic_transaction"]["all_or_nothing_commit"])

    def test_flat_dimension_and_action_capacity_are_fixed_across_scales(self) -> None:
        for uav_count in (4, 8, 16, 32):
            with self.subTest(uav_count=uav_count):
                graph = build_execution_graph(self.runtime(uav_count), now=1.0)
                observation = build_flat_observation(graph)
                self.assertEqual(len(observation.vector), FLAT_OBSERVATION_DIMENSION)
                self.assertEqual(len(observation.action_space.mask), ACTION_CAPACITY)
                self.assertEqual(
                    observation.action_space.valid_action_count,
                    1 + len(graph.action_candidates),
                )
                self.assertTrue(all(0.0 <= value <= 1.0 for value in observation.vector))

    def test_flat_and_hetero_share_exact_action_space(self) -> None:
        graph = build_execution_graph(self.runtime(), now=2.0)
        flat = build_flat_observation(graph)
        hetero = build_hetero_observation(graph)
        self.assertEqual(flat.graph_sha256, graph.sha256())
        self.assertEqual(hetero.graph_sha256, graph.sha256())
        self.assertEqual(flat.action_space, hetero.action_space)
        self.assertEqual(set(hetero.node_ids), {"UAV", "Task", "Region", "Target", "Event"})
        self.assertEqual(len(hetero.edge_indices), 7)

    def test_request_narrows_mask_to_one_task_and_safe_candidates(self) -> None:
        runtime = self.runtime()
        graph = build_execution_graph(runtime, now=2.0)
        task = runtime.tasks["T000"]
        request = build_allocation_request(
            request_id="R0",
            graph_version=runtime.graph_version,
            task=task,
            uavs=tuple(runtime.uavs.values()),
            decision_type=DecisionType.CONTINUE,
            reason="adapter test",
            generated_at=2.0,
        )
        observation = build_flat_observation(graph, request=request)
        self.assertFalse(observation.action_space.mask[0])
        enabled = {
            binding
            for binding, active in zip(
                observation.action_space.bindings,
                observation.action_space.mask,
            )
            if active and binding != ("NOOP", "NOOP")
        }
        self.assertEqual(enabled, {(uav_id, "T000") for uav_id in runtime.uavs})
        self.assertNotIn(("U00", "T001"), enabled)

    def test_rule_context_is_explicit_and_version_bound(self) -> None:
        runtime = self.runtime()
        graph = build_execution_graph(runtime, now=2.0)
        decision = EventDecision(
            event_id="E0",
            priority=EventPriority.P1,
            information_age=2.0,
            confidence=0.8,
            decision=DecisionType.PREEMPT,
            displaced_task_id="T001",
            selected_uav=None,
            reason="urgent",
            graph_version=runtime.graph_version,
        )
        reactive = build_flat_observation(graph)
        ruled = build_flat_observation(graph, decision=decision)
        self.assertFalse(reactive.rule_context_present)
        self.assertTrue(ruled.rule_context_present)
        self.assertEqual(sum(reactive.vector[-16:]), 0.0)
        self.assertGreater(sum(ruled.vector[-16:]), 0.0)
        with self.assertRaisesRegex(AdapterValidationError, "EventDecision"):
            build_flat_observation(graph, decision=EventDecision(
                event_id="E1",
                priority=EventPriority.P1,
                information_age=0.0,
                confidence=1.0,
                decision=DecisionType.PREEMPT,
                displaced_task_id=None,
                selected_uav=None,
                reason="stale",
                graph_version=runtime.graph_version + 1,
            ))

    def test_decode_and_proposal_pass_same_graph_and_candidate_validation(self) -> None:
        runtime = self.runtime()
        graph = build_execution_graph(runtime, now=2.0)
        request = build_allocation_request(
            request_id="R0",
            graph_version=runtime.graph_version,
            task=runtime.tasks["T000"],
            uavs=tuple(runtime.uavs.values()),
            decision_type=DecisionType.CONTINUE,
            reason="adapter test",
            generated_at=2.0,
        )
        observation = build_flat_observation(graph, request=request)
        action_index = observation.action_space.bindings.index(("U02", "T000"))
        decoded = decode_policy_action(
            observation.action_space,
            action_index,
            current_graph_version=runtime.graph_version,
            current_graph_sha256=graph.sha256(),
        )
        self.assertEqual((decoded.uav_id, decoded.task_id), ("U02", "T000"))
        proposal = proposal_from_policy_action(
            request,
            observation.action_space,
            action_index,
            allocator_id="ppo_mlp_rule_arbiter_v1",
            current_graph_version=runtime.graph_version,
            current_graph_sha256=graph.sha256(),
        )
        self.assertEqual(proposal.uav_id, "U02")
        self.assertEqual(proposal.metadata["action_index"], action_index)

    def test_stale_hash_version_masked_and_noop_fail_closed(self) -> None:
        graph = build_execution_graph(self.runtime(), now=2.0)
        observation = build_flat_observation(graph)
        self.assertTrue(observation.action_space.mask[0])
        valid_index = next(index for index, active in enumerate(observation.action_space.mask) if active and index)
        with self.assertRaisesRegex(AdapterValidationError, "graph_version"):
            decode_policy_action(
                observation.action_space,
                valid_index,
                current_graph_version=graph.graph_version + 1,
                current_graph_sha256=graph.sha256(),
            )
        with self.assertRaisesRegex(AdapterValidationError, "graph hash"):
            decode_policy_action(
                observation.action_space,
                valid_index,
                current_graph_version=graph.graph_version,
                current_graph_sha256="0" * 64,
            )
        masked_index = next(index for index, active in enumerate(observation.action_space.mask) if not active)
        with self.assertRaisesRegex(AdapterValidationError, "masked"):
            decode_policy_action(
                observation.action_space,
                masked_index,
                current_graph_version=graph.graph_version,
                current_graph_sha256=graph.sha256(),
            )

        request = build_allocation_request(
            request_id="R0",
            graph_version=graph.graph_version,
            task=self.runtime().tasks["T000"],
            uavs=tuple(self.runtime().uavs.values()),
            decision_type=DecisionType.CONTINUE,
            reason="noop test",
            generated_at=2.0,
        )
        with self.assertRaisesRegex(AllocationValidationError, "NOOP"):
            proposal_from_policy_action(
                request,
                observation.action_space,
                0,
                allocator_id="test",
                current_graph_version=graph.graph_version,
                current_graph_sha256=graph.sha256(),
            )

    def test_adapter_is_byte_deterministic(self) -> None:
        graph = build_execution_graph(self.runtime(), now=2.0)
        first = build_flat_observation(graph)
        second = build_flat_observation(graph)
        self.assertEqual(first, second)
        self.assertEqual(build_hetero_observation(graph), build_hetero_observation(graph))

    def test_capacity_overflow_is_hard_error(self) -> None:
        graph = build_execution_graph(self.runtime(33, 1), now=2.0)
        with self.assertRaisesRegex(AdapterValidationError, "exceeds frozen capacity"):
            build_flat_observation(graph)


if __name__ == "__main__":
    unittest.main()
