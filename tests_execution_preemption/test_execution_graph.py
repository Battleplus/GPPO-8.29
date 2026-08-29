from __future__ import annotations

import json
from pathlib import Path
import unittest

from execution_preemption import (
    CommunicationState,
    EventPriority,
    ExecutionRuntime,
    RegionRuntime,
    RuntimeEvent,
    RuntimeEventType,
    TargetRuntime,
    TaskRuntime,
    UAVAvailability,
    UAVRuntime,
    build_execution_graph,
)


class ExecutionGraphTests(unittest.TestCase):
    def test_machine_schema_matches_implementation_dimensions(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs" / "execution_graph_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(config["schema_id"], "execution-preemption-heterograph-v1")
        self.assertEqual(
            {name: item["feature_dimension"] for name, item in config["node_types"].items()},
            {"UAV": 15, "Task": 17, "Region": 5, "Target": 6, "Event": 12},
        )
        self.assertFalse(config["old_checkpoint_compatible"])
        self.assertTrue(config["requires_new_training_contract"])

    @staticmethod
    def build_scaled_runtime(uav_count: int) -> ExecutionRuntime:
        runtime = ExecutionRuntime()
        for index in range(uav_count):
            runtime.add_uav(UAVRuntime(
                f"U{index:02d}",
                energy_ratio=0.8,
                supported_task_types=frozenset({"SEARCH"}),
            ))
        for index in range(uav_count * 2):
            runtime.add_task(TaskRuntime(
                f"T{index:03d}",
                "SEARCH",
                priority=index % 100,
                deadline=100.0 + index,
            ))
        return runtime

    def test_dynamic_scale_4_8_16_32_builds_without_fixed_dimensions(self) -> None:
        for uav_count in (4, 8, 16, 32):
            with self.subTest(uav_count=uav_count):
                runtime = self.build_scaled_runtime(uav_count)
                graph = build_execution_graph(runtime, now=0.0)
                self.assertEqual(len(graph.nodes["UAV"]), uav_count)
                self.assertEqual(len(graph.nodes["Task"]), uav_count * 2)
                self.assertEqual(len(graph.action_candidates), uav_count * uav_count * 2)
                graph.validate()

    def test_unsafe_uavs_are_absent_from_action_candidates(self) -> None:
        runtime = ExecutionRuntime()
        runtime.add_uav(UAVRuntime("SAFE", energy_ratio=0.8))
        runtime.add_uav(UAVRuntime(
            "LOST",
            energy_ratio=0.9,
            communication_state=CommunicationState.LOST,
            availability=UAVAvailability.COMMUNICATION_LOST,
        ))
        runtime.add_uav(UAVRuntime("LOW", energy_ratio=0.15, reserve_energy=0.1, estimated_rtb_energy=0.1))
        runtime.add_task(TaskRuntime("T0", "SEARCH", 10, 100.0))
        graph = build_execution_graph(runtime, now=1.0)
        self.assertEqual(graph.action_candidates, (("SAFE", "T0"),))

    def test_running_task_is_not_reassignable(self) -> None:
        runtime = ExecutionRuntime()
        runtime.add_uav(UAVRuntime("U0", energy_ratio=0.8))
        runtime.add_uav(UAVRuntime("U1", energy_ratio=0.8))
        runtime.add_task(TaskRuntime("RUN", "SEARCH", 10, 100.0))
        runtime.assign_task("RUN", "U0", at=0.0)
        graph = build_execution_graph(runtime, now=1.0)
        self.assertNotIn(("U1", "RUN"), graph.action_candidates)
        self.assertTrue(any(
            edge.relation_key == ("UAV", "executes", "Task")
            and edge.src_id == "U0" and edge.dst_id == "RUN"
            for edge in graph.edges
        ))

    def test_five_node_types_and_seven_relations_are_materialized(self) -> None:
        runtime = ExecutionRuntime()
        runtime.add_uav(UAVRuntime("U0", energy_ratio=0.9))
        runtime.add_uav(UAVRuntime("U1", energy_ratio=0.8))
        first = TaskRuntime("T0", "SEARCH", 20, 50.0, metadata={"region_id": "R0"})
        second = TaskRuntime(
            "T1", "SEARCH", 80, 10.0,
            metadata={"region_id": "R0", "depends_on": ["T0"]},
        )
        runtime.add_task(first)
        runtime.add_task(second)
        runtime.assign_task("T0", "U0", at=0.0)
        item = RuntimeEvent(
            "E0", RuntimeEventType.TASK_ARRIVAL, EventPriority.P1,
            occurred_at=1.0, received_at=1.2, task_id="T1", uav_id="U1",
        )
        graph = build_execution_graph(
            runtime,
            now=2.0,
            events=[item],
            regions=[RegionRuntime("R0", vacant=False, demand=0.8, priority=60)],
            targets=[TargetRuntime("X0", status="DETECTED", confidence=0.9, priority=70)],
            preemption_links=[("T1", "T0")],
        )
        self.assertEqual(set(graph.nodes), {"UAV", "Task", "Region", "Target", "Event"})
        relation_keys = {edge.relation_key for edge in graph.edges}
        self.assertEqual(relation_keys, {
            ("UAV", "executes", "Task"),
            ("UAV", "can_execute", "Task"),
            ("Task", "located_in", "Region"),
            ("Task", "depends_on", "Task"),
            ("Event", "affects", "UAV"),
            ("Event", "affects", "Task"),
            ("Task", "preempts", "Task"),
        })

    def test_graph_is_byte_stable_and_version_bound(self) -> None:
        runtime = self.build_scaled_runtime(4)
        first = build_execution_graph(runtime, now=3.0)
        second = build_execution_graph(runtime, now=3.0)
        self.assertEqual(first.sha256(), second.sha256())
        self.assertEqual(first.to_dict(), second.to_dict())
        runtime.graph_version += 1
        third = build_execution_graph(runtime, now=3.0)
        self.assertNotEqual(first.sha256(), third.sha256())
        self.assertEqual(third.graph_version, 1)

    def test_all_features_are_finite_normalized_and_exact_dimension(self) -> None:
        runtime = self.build_scaled_runtime(4)
        graph = build_execution_graph(
            runtime,
            now=1000.0,
            events=[RuntimeEvent(
                "E0", RuntimeEventType.UAV_COMM_LOST, EventPriority.P2,
                occurred_at=0.0, received_at=900.0, uav_id="U00",
                payload={"severity": 2.0},
            )],
            regions=[RegionRuntime("R0", vacant=True, demand=2.0, priority=200, uncertainty=2.0)],
            targets=[TargetRuntime("X0", status="TRACKED", confidence=2.0, priority=200)],
        )
        graph.validate()
        expected = {"UAV": 15, "Task": 17, "Region": 5, "Target": 6, "Event": 12}
        for node_type, nodes in graph.nodes.items():
            for node in nodes:
                self.assertEqual(len(node.features), expected[node_type])
                self.assertTrue(all(0.0 <= value <= 1.0 for value in node.features))

    def test_noop_is_explicit_even_when_no_safe_edge_exists(self) -> None:
        runtime = ExecutionRuntime()
        runtime.add_uav(UAVRuntime(
            "U0", energy_ratio=0.1,
            reserve_energy=0.1, estimated_rtb_energy=0.1,
        ))
        runtime.add_task(TaskRuntime("T0", "SEARCH", 10, 100.0))
        graph = build_execution_graph(runtime, now=0.0)
        self.assertEqual(graph.action_candidates, ())
        self.assertEqual(graph.noop_action, ("NOOP", "NOOP"))


if __name__ == "__main__":
    unittest.main()
