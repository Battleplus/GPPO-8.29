#pragma once

#include <optional>
#include <string>

namespace brain_cpp {

enum class MissionState {
    INIT,
    RECON_ALLOCATING_BY_MILP,
    RECON_PLANNING_BY_MPPI,
    RECON_PLAN_READY,
    RECON_EXECUTING,
    WAIT_RECON_RESULT,
    UPDATE_WORLD_STATE,
    ACTION_ALLOCATING_BY_MILP,
    POSITION_SELECTING,
    ACTION_PLANNING_BY_MPPI,
    ACTION_PLAN_READY,
    ACTION_EXECUTING,
    REPLAN,
    MISSION_COMPLETE,
    MISSION_FAILED,
};

std::string toString(MissionState state);
std::optional<MissionState> missionStateFromString(const std::string& value);
bool isWaiting(MissionState state);
bool isTerminal(MissionState state);

}  // namespace brain_cpp
