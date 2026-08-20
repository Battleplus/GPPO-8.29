"""Test: PositionSelector placeholder logic.

When ``perch`` is not yet available, ``PositionSelector``
returns a default attack position.  This test verifies that the
placeholder does **not** block the pipeline.
"""

from __future__ import annotations

from brain.adapters import MILPTaskAllocator, MPPIFormationPlanner, PositionSelector
from brain.core import MissionBrain, MissionEvent, MissionState


def test_position_selector_placeholder_does_not_block(brain):
    """Default PositionSelector returns a placeholder, pipeline proceeds."""
    ctx = brain.context

    state = brain.start()
    assert state == MissionState.RECON_PLAN_READY

    # Dispatch recon events to reach ACTION_PLAN_READY
    brain.dispatch(MissionEvent.recon_plan_dispatched())
    brain.dispatch(MissionEvent.recon_finished())
    state = brain.dispatch(
        MissionEvent.recon_result_received(
            data={"detections": [{"truth_id": "g1", "confidence": 0.9}]}
        )
    )
    assert state == MissionState.ACTION_PLAN_READY, f"Expected ACTION_PLAN_READY, got {state}"

    # Verify selected_positions is populated by FREA
    positions = ctx.selected_positions
    assert positions is not None, "selected_positions should not be None"
    assert len(positions) > 0, "selected_positions should contain at least one position"

    pos = positions[0]
    assert pos.kind == "attack"
    assert pos.x != 0.0 or pos.y != 0.0  # position should be non-trivial
    # FREA stores diagnostics in metadata
    assert "f_exposure" in pos.metadata, f"metadata missing f_exposure: {pos.metadata}"
    assert "f_range" in pos.metadata
    assert pos.metadata.get("g_violation", 1.0) == 0.0, "Position should satisfy all constraints"

    print(f"PASS: FREA position returned: {pos}")


def test_position_selector_failure_triggers_replan(sample_context):
    """When PositionSelector fails, the FSM should enter REPLAN."""
    from brain.domain.result import AlgorithmResult

    class _FailingPositionSelector:
        def select(self, context, action_allocation):
            return AlgorithmResult.fail("Position model not available for this terrain")

    brain = MissionBrain(
        sample_context,
        MILPTaskAllocator(),
        MPPIFormationPlanner(),
        _FailingPositionSelector(),
    )

    state = brain.start()
    assert state == MissionState.RECON_PLAN_READY

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

    # PositionSelector fails → REPLAN → retries → MISSION_FAILED
    assert state == MissionState.MISSION_FAILED, f"Expected MISSION_FAILED, got {state}"
    assert sample_context.retry_count > sample_context.max_retry
    assert "Position" in sample_context.last_error
    print("PASS: PositionSelector failure → REPLAN → MISSION_FAILED.")
