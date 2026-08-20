"""Test: MILP failure → immediate MISSION_FAILED (no retry).

MILP is a core allocation step — when it fails there is no alternative
path, so the FSM transitions directly to MISSION_FAILED.
"""

from __future__ import annotations

from brain.adapters import MPPIFormationPlanner, PositionSelector
from brain.core import MissionBrain, MissionState
from brain.domain.result import AlgorithmResult
from brain.domain.task import ReconTask, StrikeTask


class _FailingMILPRecon:
    """MILP adapter whose ``allocate_recon`` always fails."""

    def allocate_recon(self, context):
        return AlgorithmResult.fail("MILP solver returned INFEASIBLE")

    def allocate_action(self, context):
        return AlgorithmResult.ok([])


def test_milp_recon_failure(sample_context):
    brain = MissionBrain(
        sample_context,
        _FailingMILPRecon(),
        MPPIFormationPlanner(),
        PositionSelector(),
    )

    state = brain.start()
    assert state == MissionState.MISSION_FAILED, f"Expected MISSION_FAILED, got {state}"
    assert state.is_terminal
    assert "MILP" in sample_context.last_error.upper() or "INFEASIBLE" in sample_context.last_error.upper()
    assert sample_context.last_error != ""
    print("PASS: MILP recon failure → MISSION_FAILED (no retry).")


class _FailingMILPAction:
    """MILP adapter whose ``allocate_recon`` succeeds but ``allocate_action`` fails."""

    def allocate_recon(self, context):
        return AlgorithmResult.ok(
            [ReconTask(platform="U1", cell="c0", sensor="SAR", role="area_scan")]
        )

    def allocate_action(self, context):
        return AlgorithmResult.fail("MILP action solver returned INFEASIBLE")


def test_milp_action_failure(sample_context):
    """MILP action allocation failure should also go to MISSION_FAILED."""
    # We need to get past the recon pipeline first
    from brain.core import MissionEvent

    brain = MissionBrain(
        sample_context,
        _FailingMILPAction(),
        MPPIFormationPlanner(),
        PositionSelector(),
    )

    state = brain.start()
    # Recon pipeline should succeed
    assert state == MissionState.RECON_PLAN_READY, f"Expected RECON_PLAN_READY, got {state}"

    # Dispatch recon events → UPDATE_WORLD → ACTION_ALLOCATING (fails)
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

    # After UPDATE_WORLD the auto chain hits ACTION_ALLOCATING → fails → MISSION_FAILED
    assert state == MissionState.MISSION_FAILED, f"Expected MISSION_FAILED, got {state}"
    assert sample_context.last_error != ""
    print("PASS: MILP action failure → MISSION_FAILED (no retry).")
