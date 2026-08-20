#pragma once

#include <string>

namespace brain_cpp {

enum class TargetTrackingState {
    UNTRACKED,
    DETECTED,
    TRACKING,
    APPROACH_CONFIRMING,
    CONFIRMED,
    ATTACK_PLANNING,
    ATTACK_PLAN_READY,
    FAILED,
};

enum class TargetTrackingEvent {
    SENSOR_CONTACT,
    START_TRACKING,
    REQUEST_APPROACH_CONFIRMATION,
    TARGET_CONFIRMED,
    START_ATTACK_PLANNING,
    ATTACK_PLAN_SUCCEEDED,
    ATTACK_PLAN_FAILED,
};

struct TargetTrackingStatus {
    std::string target_id;
    std::string platform_id;
    std::string sensor;
    TargetTrackingState state = TargetTrackingState::UNTRACKED;
    std::string platform_command = "NONE";
    std::string failure_reason;
};

class TargetTrackingFSM {
public:
    explicit TargetTrackingFSM(TargetTrackingStatus& status);

    bool dispatch(TargetTrackingEvent event, const std::string& detail = "");

private:
    TargetTrackingStatus& status_;
};

std::string toString(TargetTrackingState state);

}  // namespace brain_cpp
