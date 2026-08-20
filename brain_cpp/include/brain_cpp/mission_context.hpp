#pragma once

#include "brain_cpp/domain.hpp"
#include "brain_cpp/mission_state.hpp"
#include "brain_cpp/target_tracking_fsm.hpp"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace brain_cpp {

struct MissionContext {
    std::string mission_id = "MISSION_001";
    MissionState state = MissionState::INIT;

    std::vector<AgentSpec> agents;
    WorldState world;

    std::vector<ReconTask> recon_allocation;
    std::vector<StrikeTask> action_allocation;

    FormationPlan recon_formation_plan;
    FormationPlan action_formation_plan;
    std::vector<PatrolPlan> recon_patrol_plans;
    std::vector<Position> selected_positions;

    std::vector<std::string> recon_result_targets;
    std::vector<std::string> pending_strike_targets;
    std::set<std::string> engaged_targets;
    std::map<std::string, FormationPlan> active_action_plans;
    std::map<std::string, TargetTrackingStatus> target_tracks;

    bool environment_initialized = false;
    std::string environment_name;
    double environment_time_s = 0.0;

    int retry_count = 0;
    int max_retry = 3;
    std::string last_failed_state;
    std::string last_error;

    std::vector<HistoryEntry> history;

    void recordEvent(const std::string& event, const std::string& detail = "");
    bool hasPendingActionTasks() const;
};

std::string timestampNow();
bool containsString(const std::vector<std::string>& values, const std::string& item);
void appendUnique(std::vector<std::string>& values, const std::string& item);
void removeValue(std::vector<std::string>& values, const std::string& item);

}  // namespace brain_cpp
