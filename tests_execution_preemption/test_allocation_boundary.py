from __future__ import annotations

import unittest

from execution_preemption import (
    AllocationProposal,
    AllocationValidationError,
    CallbackAllocator,
    CommunicationState,
    DecisionType,
    EventPriority,
    ExecutionRuntime,
    FirstAvailableAllocator,
    MaxEnergyMarginAllocator,
    PreemptionController,
    RuntimeEvent,
    RuntimeEventType,
    TaskRuntime,
    UAVAvailability,
    UAVRuntime,
    build_allocation_request,
    validate_proposal,
)


class AllocationBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = TaskRuntime("T0", "SEARCH", 10, 100.0)
        self.uavs = [
            UAVRuntime("U0", energy_ratio=0.55, reserve_energy=0.1, estimated_rtb_energy=0.2),
            UAVRuntime("U1", energy_ratio=0.90, reserve_energy=0.1, estimated_rtb_energy=0.2),
            UAVRuntime(
                "U2", energy_ratio=0.95,
                communication_state=CommunicationState.LOST,
                availability=UAVAvailability.COMMUNICATION_LOST,
            ),
            UAVRuntime("U3", energy_ratio=0.15, reserve_energy=0.1, estimated_rtb_energy=0.1),
        ]

    def request(self, *, version: int = 7):
        return build_allocation_request(
            request_id="R0",
            graph_version=version,
            task=self.task,
            uavs=self.uavs,
            decision_type=DecisionType.CONTINUE,
            reason="test",
            generated_at=1.0,
        )

    def test_candidate_set_excludes_unsafe_or_disconnected_uavs(self) -> None:
        request = self.request()
        self.assertEqual(request.candidate_uav_ids, ("U0", "U1"))

    def test_first_available_is_deterministic(self) -> None:
        request = self.request()
        proposal = FirstAvailableAllocator().propose(request)
        self.assertEqual(proposal.uav_id, "U0")
        self.assertEqual(proposal.allocator_id, "first_available_v1")

    def test_max_energy_margin_selects_largest_safe_margin(self) -> None:
        request = self.request()
        proposal = MaxEnergyMarginAllocator().propose(request)
        self.assertEqual(proposal.uav_id, "U1")
        self.assertAlmostEqual(proposal.score or 0.0, 0.6)

    def test_stale_proposal_is_rejected(self) -> None:
        request = self.request(version=7)
        proposal = MaxEnergyMarginAllocator().propose(request)
        with self.assertRaises(AllocationValidationError):
            validate_proposal(request, proposal, current_graph_version=8)

    def test_non_candidate_selection_is_rejected(self) -> None:
        request = self.request()
        proposal = AllocationProposal(
            request_id=request.request_id,
            graph_version=request.graph_version,
            task_id=request.task_id,
            uav_id="U2",
            allocator_id="malformed",
        )
        with self.assertRaises(AllocationValidationError):
            validate_proposal(request, proposal, current_graph_version=7)

    def test_callback_allocator_cannot_escape_candidate_mask(self) -> None:
        controller = PreemptionController(CallbackAllocator("ppo_stub", lambda _: "U2"))
        runtime = ExecutionRuntime()
        for uav in self.uavs:
            runtime.add_uav(uav)
        runtime.add_task(self.task)
        item = RuntimeEvent(
            event_id="E0",
            event_type=RuntimeEventType.TASK_ARRIVAL,
            priority=EventPriority.P3,
            occurred_at=1.0,
            received_at=1.0,
            task_id="T0",
        )
        with self.assertRaises(AllocationValidationError):
            runtime.process_event_batch([item], controller, now=1.0)
        self.assertEqual(runtime.graph_version, 0)
        self.assertIsNone(runtime.tasks["T0"].assigned_uav)

    def test_callback_allocator_controls_only_safe_uav_selection(self) -> None:
        controller = PreemptionController(CallbackAllocator("gppo_stub", lambda _: "U0"))
        runtime = ExecutionRuntime()
        for uav in self.uavs:
            runtime.add_uav(uav)
        runtime.add_task(self.task)
        item = RuntimeEvent(
            event_id="E0",
            event_type=RuntimeEventType.TASK_ARRIVAL,
            priority=EventPriority.P3,
            occurred_at=1.0,
            received_at=1.0,
            task_id="T0",
        )
        result = runtime.process_event_batch([item], controller, now=1.0)
        decision = result.decisions[0]
        self.assertEqual(decision.selected_uav, "U0")
        self.assertEqual(decision.allocator_id, "gppo_stub")
        self.assertIsNotNone(decision.allocation_request_id)
        self.assertEqual(runtime.tasks["T0"].assigned_uav, "U0")

    def test_safety_rtb_does_not_call_allocator(self) -> None:
        calls = []

        def forbidden(_):
            calls.append(True)
            raise AssertionError("allocator must not decide RTB")

        controller = PreemptionController(CallbackAllocator("forbidden", forbidden))
        runtime = ExecutionRuntime()
        runtime.add_uav(UAVRuntime("U0", energy_ratio=0.9))
        runtime.add_task(TaskRuntime("T0", "SEARCH", 10, 100.0))
        runtime.assign_task("T0", "U0", at=0.0)
        item = RuntimeEvent(
            event_id="E-rtb",
            event_type=RuntimeEventType.UAV_LOW_ENERGY,
            priority=EventPriority.P2,
            occurred_at=1.0,
            received_at=1.0,
            uav_id="U0",
        )
        result = runtime.process_event_batch([item], controller, now=1.0)
        self.assertEqual(result.decisions[0].decision, DecisionType.RTB)
        self.assertEqual(calls, [])

    def test_callback_returning_mismatched_proposal_is_rejected(self) -> None:
        request = self.request()

        def mismatch(_):
            return AllocationProposal(
                request_id="other",
                graph_version=7,
                task_id="T0",
                uav_id="U0",
                allocator_id="bad",
            )

        proposal = CallbackAllocator("bad", mismatch).propose(request)
        with self.assertRaises(AllocationValidationError):
            validate_proposal(request, proposal, current_graph_version=7)


if __name__ == "__main__":
    unittest.main()
