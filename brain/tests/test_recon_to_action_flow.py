"""Integration test: full successful flow from INIT to MISSION_COMPLETE.

Verifies the complete pipeline::

    INIT → RECON_ALLOCATING → RECON_PLANNING → RECON_PLAN_READY
         → RECON_EXECUTING → WAIT_RECON_RESULT → UPDATE_WORLD
         → ACTION_ALLOCATING → POSITION_SELECTING → ACTION_PLANNING
         → ACTION_PLAN_READY → ACTION_EXECUTING → MISSION_COMPLETE
"""

from __future__ import annotations

from brain.core import MissionEvent, MissionState


def test_full_flow_state_sequence(brain):
    """Walk the full mission and verify the state sequence."""
    ctx = brain.context

    # -- Phase 1: start ----------------------------------------------------
    state = brain.start()
    assert state == MissionState.RECON_PLAN_READY, f"Expected RECON_PLAN_READY, got {state}"
    assert ctx.recon_allocation is not None
    assert ctx.recon_formation_plan is not None
    assert len(ctx.recon_allocation) > 0
    assert ctx.recon_formation_plan.success is True

    # -- Phase 2: recon execution events -----------------------------------
    state = brain.dispatch(MissionEvent.recon_plan_dispatched())
    assert state == MissionState.RECON_EXECUTING

    state = brain.dispatch(MissionEvent.recon_finished())
    assert state == MissionState.WAIT_RECON_RESULT

    state = brain.dispatch(
        MissionEvent.recon_result_received(
            data={
                "detections": [
                    {"truth_id": "g1", "confidence": 0.92},
                    {"truth_id": "g2", "confidence": 0.88},
                    {"truth_id": "g3", "confidence": 0.75},
                ],
            }
        )
    )
    # After RECON_RESULT_RECEIVED the auto chain goes:
    #   UPDATE_WORLD → ACTION_ALLOCATING → POSITION_SELECTING → ACTION_PLANNING → ACTION_PLAN_READY
    assert state == MissionState.ACTION_PLAN_READY, f"Expected ACTION_PLAN_READY, got {state}"

    # Verify intermediate results were stored
    assert ctx.recon_result is not None
    assert ctx.action_allocation is not None
    assert ctx.selected_positions is not None
    assert ctx.action_formation_plan is not None

    # Verify world_state targets were confirmed by recon result
    for target in ctx.world_state.get("targets", []):
        assert target.get("confirmed", False), f"Target {target['tid']} should be confirmed"

    # -- Phase 3: action execution events ----------------------------------
    state = brain.dispatch(MissionEvent.action_plan_dispatched())
    assert state == MissionState.ACTION_EXECUTING

    state = brain.dispatch(MissionEvent.action_finished())
    assert state == MissionState.MISSION_COMPLETE
    assert state.is_terminal

    # -- Audit trail -------------------------------------------------------
    assert len(ctx.history) > 0
    # Verify key events appear in history
    events_seen = {e["event"] for e in ctx.history}
    for expected in ("START", "RECON_ALLOCATED", "RECON_PLANNED", "ACTION_ALLOCATED", "ACTION_PLANNED", "TERMINAL"):
        assert expected in events_seen, f"Missing history event: {expected}"

    print("PASS: full recon-to-action flow completed successfully.")


def test_no_targets_skips_action(brain, sample_context):
    """When world_state has no alive targets, UPDATE_WORLD should go straight to MISSION_COMPLETE."""
    sample_context.world_state["targets"] = []  # no targets at all
    # Rebuild brain with the modified context
    from brain.adapters import MILPTaskAllocator, MPPIFormationPlanner, PositionSelector
    from brain.core import MissionBrain

    brain2 = MissionBrain(
        sample_context,
        MILPTaskAllocator(),
        MPPIFormationPlanner(),
        PositionSelector(),
    )
    state = brain2.start()
    # Should reach RECON_PLAN_READY
    assert state == MissionState.RECON_PLAN_READY

    # Dispatch recon events — after UPDATE_WORLD it should go to MISSION_COMPLETE
    brain2.dispatch(MissionEvent.recon_plan_dispatched())
    brain2.dispatch(MissionEvent.recon_finished())
    state = brain2.dispatch(MissionEvent.recon_result_received(data={"detections": []}))
    # UPDATE_WORLD should not find any pending action tasks → MISSION_COMPLETE
    assert state == MissionState.MISSION_COMPLETE, f"Expected MISSION_COMPLETE, got {state}"
    print("PASS: no-targets path goes to MISSION_COMPLETE.")
