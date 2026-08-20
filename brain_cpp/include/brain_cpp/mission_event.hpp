#pragma once

#include <string>
#include <vector>

namespace brain_cpp {

enum class MissionEventType {
    START,
    RECON_PLAN_DISPATCHED,
    RECON_FINISHED,
    RECON_RESULT_RECEIVED,
    ACTION_PLAN_DISPATCHED,
    ACTION_FINISHED,
    TARGET_DETECTED,
    TARGET_CONFIRMED,
    PLATFORM_LOST,
    ATTACK_FINISHED,
    RESET,
};

struct MissionEvent {
    MissionEventType type = MissionEventType::START;
    std::string platform_id;
    std::string target_id;
    std::vector<std::string> detected_targets;
    bool destroyed = true;

    static MissionEvent start();
    static MissionEvent reconPlanDispatched();
    static MissionEvent reconFinished();
    static MissionEvent reconResultReceived(std::vector<std::string> target_ids);
    static MissionEvent actionPlanDispatched();
    static MissionEvent actionFinished();
    static MissionEvent targetDetected(
        std::string target_id,
        std::string platform_id = "");
    static MissionEvent targetConfirmed(
        std::string target_id,
        std::string platform_id = "");
    static MissionEvent platformLost(std::string platform_id);
    static MissionEvent attackFinished(
        std::string target_id,
        bool destroyed = true);
    static MissionEvent reset();
};

std::string toString(MissionEventType type);

}  // namespace brain_cpp
