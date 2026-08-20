#include "brain_cpp/mission_state.hpp"

#include <array>

namespace brain_cpp {

std::string toString(MissionState state) {
    switch (state) {
    case MissionState::INIT: return "INIT";
    case MissionState::RECON_ALLOCATING_BY_MILP: return "RECON_ALLOCATING_BY_MILP";
    case MissionState::RECON_PLANNING_BY_MPPI: return "RECON_PLANNING_BY_MPPI";
    case MissionState::RECON_PLAN_READY: return "RECON_PLAN_READY";
    case MissionState::RECON_EXECUTING: return "RECON_EXECUTING";
    case MissionState::WAIT_RECON_RESULT: return "WAIT_RECON_RESULT";
    case MissionState::UPDATE_WORLD_STATE: return "UPDATE_WORLD_STATE";
    case MissionState::ACTION_ALLOCATING_BY_MILP: return "ACTION_ALLOCATING_BY_MILP";
    case MissionState::POSITION_SELECTING: return "POSITION_SELECTING";
    case MissionState::ACTION_PLANNING_BY_MPPI: return "ACTION_PLANNING_BY_MPPI";
    case MissionState::ACTION_PLAN_READY: return "ACTION_PLAN_READY";
    case MissionState::ACTION_EXECUTING: return "ACTION_EXECUTING";
    case MissionState::REPLAN: return "REPLAN";
    case MissionState::MISSION_COMPLETE: return "MISSION_COMPLETE";
    case MissionState::MISSION_FAILED: return "MISSION_FAILED";
    }
    return "UNKNOWN";
}

std::optional<MissionState> missionStateFromString(const std::string& value) {
    static const std::array<MissionState, 15> states = {
        MissionState::INIT,
        MissionState::RECON_ALLOCATING_BY_MILP,
        MissionState::RECON_PLANNING_BY_MPPI,
        MissionState::RECON_PLAN_READY,
        MissionState::RECON_EXECUTING,
        MissionState::WAIT_RECON_RESULT,
        MissionState::UPDATE_WORLD_STATE,
        MissionState::ACTION_ALLOCATING_BY_MILP,
        MissionState::POSITION_SELECTING,
        MissionState::ACTION_PLANNING_BY_MPPI,
        MissionState::ACTION_PLAN_READY,
        MissionState::ACTION_EXECUTING,
        MissionState::REPLAN,
        MissionState::MISSION_COMPLETE,
        MissionState::MISSION_FAILED,
    };
    for (const auto state : states) {
        if (toString(state) == value) {
            return state;
        }
    }
    return std::nullopt;
}

bool isWaiting(MissionState state) {
    return state == MissionState::RECON_PLAN_READY
        || state == MissionState::RECON_EXECUTING
        || state == MissionState::WAIT_RECON_RESULT
        || state == MissionState::ACTION_PLAN_READY
        || state == MissionState::ACTION_EXECUTING;
}

bool isTerminal(MissionState state) {
    return state == MissionState::MISSION_COMPLETE
        || state == MissionState::MISSION_FAILED;
}

}  // namespace brain_cpp
