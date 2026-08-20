"""Test: MPPI failure → REPLAN → retry → eventual MISSION_FAILED.

Unlike MILP, MPPI failures are considered recoverable — the FSM enters
REPLAN and retries up to ``max_retry`` times before giving up.
"""

from __future__ import annotations

from brain.adapters import MILPTaskAllocator, PositionSelector
from brain.core import MissionBrain, MissionEvent, MissionState
from brain.domain.result import AlgorithmResult


class _FailingMPPIRecon:
    """MPPI adapter whose ``plan_recon_route`` always fails."""

    def plan_recon_route(self, context, recon_allocation):
        return AlgorithmResult.fail("MPPI planner: all trajectories in collision")

    def plan_action_route(self, context, action_allocation, selected_positions=None):
        return AlgorithmResult.ok([])


def test_mppi_recon_failure_with_retry(sample_context):
    """MPPI recon planning fails 3 times → MISSION_FAILED on 4th attempt."""
    from brain.core import MissionBrain

    brain = MissionBrain(
        sample_context,
        MILPTaskAllocator(),
        _FailingMPPIRecon(),
        PositionSelector(),
    )

    state = brain.start()
    # MILP recon allocation should succeed, then MPPI recon planning fails → REPLAN
    # After max_retry+1 attempts (4 total), MISSION_FAILED
    assert state == MissionState.MISSION_FAILED, f"Expected MISSION_FAILED, got {state}"

    # Verify retry count exceeded
    assert sample_context.retry_count > sample_context.max_retry, (
        f"retry_count={sample_context.retry_count}, max_retry={sample_context.max_retry}"
    )
    assert "MPPI" in sample_context.last_error.upper() or "collision" in sample_context.last_error.lower()

    # Check history contains REPLAN entries
    replan_entries = [e for e in sample_context.history if e["event"] == "REPLAN"]
    assert len(replan_entries) == sample_context.max_retry + 1, (
        f"Expected {sample_context.max_retry + 1} REPLAN entries, got {len(replan_entries)}"
    )
    print(f"PASS: MPPI recon failure → {sample_context.retry_count} retries → MISSION_FAILED.")


class _FailingMPPIAction:
    """MPPI adapter where recon plan works but action plan fails."""

    def plan_recon_route(self, context, recon_allocation):
        return AlgorithmResult.ok({"formation": "v_shape", "paths": []})

    def plan_action_route(self, context, action_allocation, selected_positions=None):
        return AlgorithmResult.fail("MPPI action: no feasible route to attack position")


def test_mppi_action_failure_with_retry(sample_context):
    """MPPI action planning fails → retry → MISSION_FAILED after retries exhausted."""
    brain = MissionBrain(
        sample_context,
        MILPTaskAllocator(),
        _FailingMPPIAction(),
        PositionSelector(),
    )

    state = brain.start()
    # Recon pipeline should succeed (MILP recon + MPPI recon)
    assert state == MissionState.RECON_PLAN_READY, f"Expected RECON_PLAN_READY, got {state}"

    # Dispatch recon events to get to the action pipeline
    brain.dispatch(MissionEvent.recon_plan_dispatched())
    brain.dispatch(MissionEvent.recon_finished())
    state = brain.dispatch(
        MissionEvent.recon_result_received(
            data={
                "detections": [
                    {"truth_id": "g1", "confidence": 0.9},
                    {"truth_id": "g2", "confidence": 0.85},
                ]
            }
        )
    )

    # Action pipeline: MILP action OK → Position OK → MPPI action FAIL → REPLAN → … → MISSION_FAILED
    assert state == MissionState.MISSION_FAILED, f"Expected MISSION_FAILED, got {state}"
    assert sample_context.retry_count > sample_context.max_retry
    print("PASS: MPPI action failure → retries → MISSION_FAILED.")
