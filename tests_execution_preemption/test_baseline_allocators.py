from __future__ import annotations

import unittest

from execution_preemption import (
    AllocationValidationError,
    BeamMPCAllocator,
    DecisionType,
    GreedyPriorityAllocator,
    SeniorLegacyMethodAllocator,
    TaskRuntime,
    UAVRuntime,
    build_allocation_request,
    validate_proposal,
)


class FrozenBaselineAllocatorTests(unittest.TestCase):
    @staticmethod
    def request():
        task = TaskRuntime("CURRENT", "SEARCH", 50, 20.0)
        uavs = [
            UAVRuntime(
                "U0", energy_ratio=0.70,
                supported_task_types=frozenset({"SEARCH"}),
            ),
            UAVRuntime(
                "U1", energy_ratio=0.80,
                supported_task_types=frozenset({"SEARCH"}),
            ),
            UAVRuntime(
                "U2", energy_ratio=0.95,
                supported_task_types=frozenset({"SEARCH", "URGENT"}),
            ),
        ]
        return build_allocation_request(
            request_id="R-baseline",
            graph_version=4,
            task=task,
            uavs=uavs,
            decision_type=DecisionType.CONTINUE,
            reason="baseline differentiation",
            generated_at=1.0,
            metadata={
                "current_task": {
                    "task_id": "CURRENT", "task_type": "SEARCH",
                    "priority": 50, "deadline": 20.0, "remaining_work": 1.0,
                },
                "forecast_tasks": [{
                    "task_id": "URGENT-NEXT", "task_type": "URGENT",
                    "priority": 95, "deadline": 10.0, "remaining_work": 1.0,
                }],
            },
        )

    def test_three_methods_are_distinct_on_frozen_synthetic_request(self) -> None:
        request = self.request()
        self.assertEqual(SeniorLegacyMethodAllocator().propose(request).uav_id, "U0")
        self.assertEqual(GreedyPriorityAllocator().propose(request).uav_id, "U2")
        self.assertEqual(BeamMPCAllocator().propose(request).uav_id, "U1")

    def test_exact_method_ids(self) -> None:
        request = self.request()
        self.assertEqual(
            [item.propose(request).allocator_id for item in (
                SeniorLegacyMethodAllocator(), GreedyPriorityAllocator(), BeamMPCAllocator()
            )],
            ["senior_legacy_method_v1", "greedy_priority_v1", "beam_mpc_v1"],
        )

    def test_every_baseline_proposal_passes_shared_validation(self) -> None:
        request = self.request()
        for allocator in (
            SeniorLegacyMethodAllocator(), GreedyPriorityAllocator(), BeamMPCAllocator()
        ):
            proposal = allocator.propose(request)
            self.assertEqual(
                validate_proposal(request, proposal, current_graph_version=4), proposal
            )

    def test_every_baseline_is_deterministic(self) -> None:
        request = self.request()
        for allocator in (
            SeniorLegacyMethodAllocator(), GreedyPriorityAllocator(), BeamMPCAllocator()
        ):
            self.assertEqual(allocator.propose(request), allocator.propose(request))

    def test_beam_trace_records_public_forecast_assignment(self) -> None:
        proposal = BeamMPCAllocator().propose(self.request())
        self.assertEqual(proposal.metadata["horizon"], 3)
        self.assertIn("forecast:URGENT-NEXT:U2", proposal.metadata["winning_trace"])

    def test_beam_without_forecast_remains_valid(self) -> None:
        request = self.request()
        request.metadata["forecast_tasks"] = []
        proposal = BeamMPCAllocator().propose(request)
        self.assertIn(proposal.uav_id, request.candidate_uav_ids)

    def test_beam_rejects_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            BeamMPCAllocator(horizon=0)
        with self.assertRaises(ValueError):
            BeamMPCAllocator(beam_width=0)

    def test_shared_validator_still_rejects_stale_baseline_proposal(self) -> None:
        request = self.request()
        proposal = GreedyPriorityAllocator().propose(request)
        with self.assertRaises(AllocationValidationError):
            validate_proposal(request, proposal, current_graph_version=5)


if __name__ == "__main__":
    unittest.main()
