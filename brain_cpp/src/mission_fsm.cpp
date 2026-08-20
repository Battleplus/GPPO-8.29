#include "brain_cpp/mission_fsm.hpp"

#include <set>
#include <sstream>

namespace brain_cpp {

MissionFSM::MissionFSM(
    MissionContext& context,
    ITaskAllocator& taskAllocator,
    IRoutePlanner& routePlanner,
    IPositionSelector& positionSelector)
    : ctx_(context),
      taskAllocator_(taskAllocator),
      routePlanner_(routePlanner),
      positionSelector_(positionSelector) {}

MissionState MissionFSM::dispatch(const MissionEvent& event) {
    if (event.type == MissionEventType::RESET) {
        ctx_.recordEvent("RESET", "Resetting mission to INIT");
        ctx_.state = MissionState::INIT;
        ctx_.retry_count = 0;
        ctx_.last_failed_state.clear();
        ctx_.last_error.clear();
        ctx_.recon_allocation.clear();
        ctx_.action_allocation.clear();
        ctx_.selected_positions.clear();
        ctx_.pending_strike_targets.clear();
        ctx_.engaged_targets.clear();
        ctx_.active_action_plans.clear();
        ctx_.target_tracks.clear();
        return ctx_.state;
    }

    MissionState next = ctx_.state;
    bool legal = true;

    switch (ctx_.state) {
    case MissionState::INIT:
        if (event.type == MissionEventType::START) {
            ctx_.recordEvent("START", "Mission initiated");
            next = MissionState::RECON_ALLOCATING_BY_MILP;
        } else {
            legal = false;
        }
        break;
    case MissionState::RECON_PLAN_READY:
        if (event.type == MissionEventType::RECON_PLAN_DISPATCHED) {
            ctx_.recordEvent("RECON_PLAN_DISPATCHED", "Recon formation plan dispatched");
            next = MissionState::RECON_EXECUTING;
        } else {
            legal = false;
        }
        break;
    case MissionState::RECON_EXECUTING:
        if (event.type == MissionEventType::RECON_FINISHED) {
            ctx_.recordEvent("RECON_FINISHED", "Reconnaissance execution completed");
            next = MissionState::WAIT_RECON_RESULT;
        } else {
            legal = false;
        }
        break;
    case MissionState::WAIT_RECON_RESULT:
        if (event.type == MissionEventType::RECON_RESULT_RECEIVED) {
            ctx_.recon_result_targets.clear();
            for (const auto& target : event.detected_targets) {
                appendUnique(ctx_.recon_result_targets, target);
            }
            ctx_.recordEvent("RECON_RESULT_RECEIVED", "Recon result received");
            next = MissionState::UPDATE_WORLD_STATE;
        } else {
            legal = false;
        }
        break;
    case MissionState::ACTION_PLAN_READY:
        if (event.type == MissionEventType::ACTION_PLAN_DISPATCHED) {
            ctx_.recordEvent("ACTION_PLAN_DISPATCHED", "Action formation plan dispatched");
            next = MissionState::ACTION_EXECUTING;
        } else {
            legal = false;
        }
        break;
    case MissionState::ACTION_EXECUTING:
        if (event.type == MissionEventType::ACTION_FINISHED) {
            ctx_.recordEvent("ACTION_FINISHED", "Action execution completed");
            next = MissionState::MISSION_COMPLETE;
        } else {
            legal = false;
        }
        break;
    default:
        legal = false;
        break;
    }

    if (!legal) {
        ctx_.recordEvent(
            "ILLEGAL_TRANSITION",
            "No handler for (" + toString(ctx_.state) + ", " + toString(event.type) + ")");
        return ctx_.state;
    }

    transitionTo(next, &event);
    return runAutoChain();
}

MissionState MissionFSM::currentState() const {
    return ctx_.state;
}

MissionState MissionFSM::runAutoChain() {
    constexpr int maxChain = 32;
    for (int index = 0; index < maxChain; ++index) {
        MissionState next = ctx_.state;
        bool handled = true;
        switch (ctx_.state) {
        case MissionState::RECON_ALLOCATING_BY_MILP:
            next = doReconAllocate();
            break;
        case MissionState::RECON_PLANNING_BY_MPPI:
            next = doReconPlan();
            break;
        case MissionState::UPDATE_WORLD_STATE:
            next = doUpdateWorld();
            break;
        case MissionState::ACTION_ALLOCATING_BY_MILP:
            next = doActionAllocate();
            break;
        case MissionState::POSITION_SELECTING:
            next = doPositionSelect();
            break;
        case MissionState::ACTION_PLANNING_BY_MPPI:
            next = doActionPlan();
            break;
        case MissionState::REPLAN:
            next = doReplan();
            break;
        default:
            handled = false;
            break;
        }

        if (!handled) {
            break;
        }
        transitionTo(next);
        if (isWaiting(next) || isTerminal(next)) {
            break;
        }
    }

    if (isTerminal(ctx_.state)) {
        if (ctx_.history.empty() || ctx_.history.back().event != "TERMINAL") {
            ctx_.recordEvent("TERMINAL", "Mission ended in state " + toString(ctx_.state));
        }
    }
    return ctx_.state;
}

void MissionFSM::transitionTo(MissionState nextState, const MissionEvent* event) {
    const auto old = ctx_.state;
    ctx_.state = nextState;
    if (old != nextState) {
        const std::string trigger = event ? toString(event->type) : "auto";
        ctx_.recordEvent("STATE_CHANGED", toString(old) + " -> " + toString(nextState) + " (" + trigger + ")");
    }
}

MissionState MissionFSM::doReconAllocate() {
    ctx_.recordEvent("RECON_ALLOCATING", "Calling ITaskAllocator::allocateRecon()");
    auto result = taskAllocator_.allocateRecon(ctx_);
    if (result.success) {
        ctx_.recon_allocation = result.data;
        ctx_.recordEvent("RECON_ALLOCATED", "Recon allocation succeeded");
        return MissionState::RECON_PLANNING_BY_MPPI;
    }

    ctx_.last_error = result.reason;
    ctx_.recordEvent("ALGORITHM_FAILED", "Recon allocation failed: " + result.reason);
    return MissionState::MISSION_FAILED;
}

MissionState MissionFSM::doReconPlan() {
    ctx_.recordEvent("RECON_PLANNING", "Calling IRoutePlanner::planReconRoute()");
    auto result = routePlanner_.planReconRoute(ctx_, ctx_.recon_allocation);
    if (result.success) {
        ctx_.recon_formation_plan = result.data;
        ctx_.recordEvent("RECON_PLANNED", "Recon route planning succeeded");
        return MissionState::RECON_PLAN_READY;
    }

    ctx_.last_error = result.reason;
    ctx_.last_failed_state = toString(MissionState::RECON_PLANNING_BY_MPPI);
    ctx_.recordEvent("ALGORITHM_FAILED", "Recon route planning failed: " + result.reason);
    return MissionState::REPLAN;
}

MissionState MissionFSM::doUpdateWorld() {
    ctx_.recordEvent("UPDATE_WORLD", "Updating world state from recon result");

    for (auto& target : ctx_.world.targets) {
        const bool detected = containsString(ctx_.recon_result_targets, target.tid);
        if (detected) {
            target.confirmed = true;
        }
        if (target.alive
            && target.confirmed
            && ctx_.engaged_targets.find(target.tid) == ctx_.engaged_targets.end()) {
            appendUnique(ctx_.pending_strike_targets, target.tid);
        }
    }

    if (ctx_.hasPendingActionTasks()) {
        ctx_.recordEvent("WORLD_UPDATED", "Pending action tasks remain");
        return MissionState::ACTION_ALLOCATING_BY_MILP;
    }

    ctx_.recordEvent("WORLD_UPDATED", "No pending action tasks");
    return MissionState::MISSION_COMPLETE;
}

MissionState MissionFSM::doActionAllocate() {
    ctx_.recordEvent("ACTION_ALLOCATING", "Calling ITaskAllocator::allocateAction()");
    auto result = taskAllocator_.allocateAction(ctx_, ctx_.pending_strike_targets, false);
    if (result.success) {
        ctx_.action_allocation = result.data;
        std::set<std::string> assignedTargets;
        for (const auto& task : ctx_.action_allocation) {
            if (!task.target.empty()) {
                assignedTargets.insert(task.target);
            }
        }
        for (const auto& target : assignedTargets) {
            ctx_.engaged_targets.insert(target);
            removeValue(ctx_.pending_strike_targets, target);
        }
        ctx_.recordEvent("ACTION_ALLOCATED", "Action allocation succeeded");
        return MissionState::POSITION_SELECTING;
    }

    ctx_.last_error = result.reason;
    ctx_.recordEvent("ALGORITHM_FAILED", "Action allocation failed: " + result.reason);
    return MissionState::MISSION_FAILED;
}

MissionState MissionFSM::doPositionSelect() {
    ctx_.recordEvent("POSITION_SELECTING", "Calling IPositionSelector::select()");
    auto result = positionSelector_.select(ctx_, ctx_.action_allocation);
    if (result.success) {
        ctx_.selected_positions = result.data;
        ctx_.recordEvent("POSITION_SELECTED", "Position selection succeeded");
        return MissionState::ACTION_PLANNING_BY_MPPI;
    }

    ctx_.last_error = result.reason;
    ctx_.last_failed_state = toString(MissionState::POSITION_SELECTING);
    ctx_.recordEvent("ALGORITHM_FAILED", "Position selection failed: " + result.reason);
    return MissionState::REPLAN;
}

MissionState MissionFSM::doActionPlan() {
    ctx_.recordEvent("ACTION_PLANNING", "Calling IRoutePlanner::planActionRoute()");
    auto result = routePlanner_.planActionRoute(
        ctx_,
        ctx_.action_allocation,
        ctx_.selected_positions);
    if (result.success) {
        ctx_.action_formation_plan = result.data;
        ctx_.recordEvent("ACTION_PLANNED", "Action route planning succeeded");
        return MissionState::ACTION_PLAN_READY;
    }

    ctx_.last_error = result.reason;
    ctx_.last_failed_state = toString(MissionState::ACTION_PLANNING_BY_MPPI);
    ctx_.recordEvent("ALGORITHM_FAILED", "Action route planning failed: " + result.reason);
    return MissionState::REPLAN;
}

MissionState MissionFSM::doReplan() {
    ctx_.retry_count += 1;
    ctx_.recordEvent(
        "REPLAN",
        "Retry " + std::to_string(ctx_.retry_count) + "/" + std::to_string(ctx_.max_retry)
            + " returning to " + ctx_.last_failed_state);

    if (ctx_.retry_count > ctx_.max_retry) {
        ctx_.recordEvent("MISSION_FAILED", "Max retries exceeded. Last error: " + ctx_.last_error);
        return MissionState::MISSION_FAILED;
    }

    const auto failedState = missionStateFromString(ctx_.last_failed_state);
    if (!failedState.has_value()) {
        ctx_.last_error = "Unknown failed state: " + ctx_.last_failed_state;
        return MissionState::MISSION_FAILED;
    }
    return *failedState;
}

}  // namespace brain_cpp
