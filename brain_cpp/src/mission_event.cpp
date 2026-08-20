#include "brain_cpp/mission_event.hpp"

#include <utility>

namespace brain_cpp {

MissionEvent MissionEvent::start() {
    MissionEvent event;
    event.type = MissionEventType::START;
    return event;
}

MissionEvent MissionEvent::reconPlanDispatched() {
    MissionEvent event;
    event.type = MissionEventType::RECON_PLAN_DISPATCHED;
    return event;
}

MissionEvent MissionEvent::reconFinished() {
    MissionEvent event;
    event.type = MissionEventType::RECON_FINISHED;
    return event;
}

MissionEvent MissionEvent::reconResultReceived(std::vector<std::string> target_ids) {
    MissionEvent event;
    event.type = MissionEventType::RECON_RESULT_RECEIVED;
    event.detected_targets = std::move(target_ids);
    return event;
}

MissionEvent MissionEvent::actionPlanDispatched() {
    MissionEvent event;
    event.type = MissionEventType::ACTION_PLAN_DISPATCHED;
    return event;
}

MissionEvent MissionEvent::actionFinished() {
    MissionEvent event;
    event.type = MissionEventType::ACTION_FINISHED;
    return event;
}

MissionEvent MissionEvent::targetDetected(
    std::string target_id,
    std::string platform_id) {
    MissionEvent event;
    event.type = MissionEventType::TARGET_DETECTED;
    event.target_id = std::move(target_id);
    event.platform_id = std::move(platform_id);
    return event;
}

MissionEvent MissionEvent::targetConfirmed(
    std::string target_id,
    std::string platform_id) {
    MissionEvent event;
    event.type = MissionEventType::TARGET_CONFIRMED;
    event.target_id = std::move(target_id);
    event.platform_id = std::move(platform_id);
    return event;
}

MissionEvent MissionEvent::platformLost(std::string platform_id) {
    MissionEvent event;
    event.type = MissionEventType::PLATFORM_LOST;
    event.platform_id = std::move(platform_id);
    return event;
}

MissionEvent MissionEvent::attackFinished(
    std::string target_id,
    bool destroyed) {
    MissionEvent event;
    event.type = MissionEventType::ATTACK_FINISHED;
    event.target_id = std::move(target_id);
    event.destroyed = destroyed;
    return event;
}

MissionEvent MissionEvent::reset() {
    MissionEvent event;
    event.type = MissionEventType::RESET;
    return event;
}

std::string toString(MissionEventType type) {
    switch (type) {
    case MissionEventType::START: return "START";
    case MissionEventType::RECON_PLAN_DISPATCHED: return "RECON_PLAN_DISPATCHED";
    case MissionEventType::RECON_FINISHED: return "RECON_FINISHED";
    case MissionEventType::RECON_RESULT_RECEIVED: return "RECON_RESULT_RECEIVED";
    case MissionEventType::ACTION_PLAN_DISPATCHED: return "ACTION_PLAN_DISPATCHED";
    case MissionEventType::ACTION_FINISHED: return "ACTION_FINISHED";
    case MissionEventType::TARGET_DETECTED: return "TARGET_DETECTED";
    case MissionEventType::TARGET_CONFIRMED: return "TARGET_CONFIRMED";
    case MissionEventType::PLATFORM_LOST: return "PLATFORM_LOST";
    case MissionEventType::ATTACK_FINISHED: return "ATTACK_FINISHED";
    case MissionEventType::RESET: return "RESET";
    }
    return "UNKNOWN";
}

}  // namespace brain_cpp
