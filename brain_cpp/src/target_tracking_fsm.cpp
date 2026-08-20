#include "brain_cpp/target_tracking_fsm.hpp"

namespace brain_cpp {

TargetTrackingFSM::TargetTrackingFSM(TargetTrackingStatus& status)
    : status_(status) {}

bool TargetTrackingFSM::dispatch(TargetTrackingEvent event, const std::string& detail) {
    TargetTrackingState next = status_.state;
    switch (status_.state) {
    case TargetTrackingState::UNTRACKED:
        if (event != TargetTrackingEvent::SENSOR_CONTACT) return false;
        next = TargetTrackingState::DETECTED;
        break;
    case TargetTrackingState::DETECTED:
        if (event == TargetTrackingEvent::SENSOR_CONTACT) return true;
        if (event != TargetTrackingEvent::START_TRACKING) return false;
        next = TargetTrackingState::TRACKING;
        break;
    case TargetTrackingState::TRACKING:
        if (event == TargetTrackingEvent::SENSOR_CONTACT) return true;
        if (event != TargetTrackingEvent::REQUEST_APPROACH_CONFIRMATION) return false;
        next = TargetTrackingState::APPROACH_CONFIRMING;
        status_.platform_command = "HOVER";
        break;
    case TargetTrackingState::APPROACH_CONFIRMING:
        if (event == TargetTrackingEvent::SENSOR_CONTACT) return true;
        if (event != TargetTrackingEvent::TARGET_CONFIRMED) return false;
        next = TargetTrackingState::CONFIRMED;
        status_.platform_command = "NONE";
        break;
    case TargetTrackingState::CONFIRMED:
        if (event != TargetTrackingEvent::START_ATTACK_PLANNING) return false;
        next = TargetTrackingState::ATTACK_PLANNING;
        break;
    case TargetTrackingState::ATTACK_PLANNING:
        if (event == TargetTrackingEvent::ATTACK_PLAN_SUCCEEDED) {
            next = TargetTrackingState::ATTACK_PLAN_READY;
        } else if (event == TargetTrackingEvent::ATTACK_PLAN_FAILED) {
            next = TargetTrackingState::FAILED;
            status_.failure_reason = detail;
        } else {
            return false;
        }
        break;
    case TargetTrackingState::ATTACK_PLAN_READY:
        return event == TargetTrackingEvent::SENSOR_CONTACT;
    case TargetTrackingState::FAILED:
        return false;
    }
    status_.state = next;
    return true;
}

std::string toString(TargetTrackingState state) {
    switch (state) {
    case TargetTrackingState::UNTRACKED: return "UNTRACKED";
    case TargetTrackingState::DETECTED: return "DETECTED";
    case TargetTrackingState::TRACKING: return "TRACKING";
    case TargetTrackingState::APPROACH_CONFIRMING: return "APPROACH_CONFIRMING";
    case TargetTrackingState::CONFIRMED: return "CONFIRMED";
    case TargetTrackingState::ATTACK_PLANNING: return "ATTACK_PLANNING";
    case TargetTrackingState::ATTACK_PLAN_READY: return "ATTACK_PLAN_READY";
    case TargetTrackingState::FAILED: return "FAILED";
    }
    return "UNKNOWN";
}

}  // namespace brain_cpp
