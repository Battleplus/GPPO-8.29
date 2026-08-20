#include "brain_cpp/parallel_platform_controller.hpp"

#include <algorithm>
#include <cmath>

namespace brain_cpp {
namespace {

double distance(const Point3& lhs, const Point3& rhs) {
    const double dx = rhs[0] - lhs[0];
    const double dy = rhs[1] - lhs[1];
    const double dz = rhs[2] - lhs[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

bool samePoint(const Point3& lhs, const Point3& rhs) {
    return distance(lhs, rhs) < 1e-9;
}

}  // namespace

ParallelPlatformController::ParallelPlatformController(double speed_units_per_s)
    : speed_(speed_units_per_s) {}

AlgorithmResult<ParallelExecutionSnapshot> ParallelPlatformController::load(
    const std::vector<AgentSpec>& agents,
    const std::vector<Route>& routes) {
    if (speed_ <= 0.0) {
        return AlgorithmResult<ParallelExecutionSnapshot>::fail("Platform speed must be positive");
    }
    executions_.clear();
    elapsed_s_ = 0.0;
    tick_ = 0;

    std::map<std::string, Point3> starts;
    for (const auto& agent : agents) {
        starts[agent.pid] = {agent.position[0], agent.position[1], agent.altitude_km};
    }
    for (const auto& route : routes) {
        if (route.platform.empty() || route.waypoints.empty()) {
            continue;
        }
        const auto start = starts.find(route.platform);
        if (start == starts.end()) {
            return AlgorithmResult<ParallelExecutionSnapshot>::fail(
                "Route references unknown platform: " + route.platform);
        }
        auto& execution = executions_[route.platform];
        if (execution.status.platform.empty()) {
            execution.status.platform = route.platform;
            execution.status.position = start->second;
            execution.status.state = PlatformExecutionState::READY;
        }
        for (const auto& waypoint : route.waypoints) {
            if (execution.waypoints.empty() || !samePoint(execution.waypoints.back(), waypoint)) {
                execution.waypoints.push_back(waypoint);
            }
        }
    }
    if (executions_.empty()) {
        return AlgorithmResult<ParallelExecutionSnapshot>::fail("No executable platform routes");
    }
    for (auto& item : executions_) {
        auto& execution = item.second;
        while (execution.status.waypoint_index < execution.waypoints.size()
               && samePoint(execution.status.position,
                            execution.waypoints[execution.status.waypoint_index])) {
            ++execution.status.waypoint_index;
        }
        execution.status.waypoint_count = execution.waypoints.size();
        execution.status.state = execution.status.waypoint_index == execution.waypoints.size()
            ? PlatformExecutionState::COMPLETED
            : PlatformExecutionState::READY;
    }
    return AlgorithmResult<ParallelExecutionSnapshot>::ok(snapshot());
}

AlgorithmResult<ParallelExecutionSnapshot> ParallelPlatformController::step(double dt) {
    if (dt <= 0.0) {
        return AlgorithmResult<ParallelExecutionSnapshot>::fail("Execution dt must be positive");
    }
    if (executions_.empty()) {
        return AlgorithmResult<ParallelExecutionSnapshot>::fail("No routes loaded");
    }
    ++tick_;
    elapsed_s_ += dt;
    for (auto& item : executions_) {
        auto& execution = item.second;
        if (execution.status.state == PlatformExecutionState::COMPLETED
            || execution.status.state == PlatformExecutionState::FAILED) {
            continue;
        }
        execution.status.state = PlatformExecutionState::MOVING;
        double remaining = speed_ * dt;
        while (remaining > 0.0
               && execution.status.waypoint_index < execution.waypoints.size()) {
            const auto& target = execution.waypoints[execution.status.waypoint_index];
            const double segment = distance(execution.status.position, target);
            if (segment <= remaining || segment < 1e-9) {
                execution.status.position = target;
                execution.status.distance_travelled += segment;
                remaining -= segment;
                ++execution.status.waypoint_index;
                continue;
            }
            const double ratio = remaining / segment;
            for (std::size_t axis = 0; axis < 3; ++axis) {
                execution.status.position[axis] +=
                    (target[axis] - execution.status.position[axis]) * ratio;
            }
            execution.status.distance_travelled += remaining;
            remaining = 0.0;
        }
        if (execution.status.waypoint_index == execution.waypoints.size()) {
            execution.status.state = PlatformExecutionState::COMPLETED;
        }
    }
    return AlgorithmResult<ParallelExecutionSnapshot>::ok(snapshot());
}

ParallelExecutionSnapshot ParallelPlatformController::snapshot() const {
    ParallelExecutionSnapshot result;
    result.elapsed_s = elapsed_s_;
    result.tick = tick_;
    result.all_completed = allCompleted();
    for (const auto& item : executions_) {
        result.platforms.push_back(item.second.status);
    }
    return result;
}

bool ParallelPlatformController::allCompleted() const {
    return !executions_.empty()
        && std::all_of(executions_.begin(), executions_.end(), [](const auto& item) {
               return item.second.status.state == PlatformExecutionState::COMPLETED;
           });
}

std::string toString(PlatformExecutionState state) {
    switch (state) {
        case PlatformExecutionState::READY: return "READY";
        case PlatformExecutionState::MOVING: return "MOVING";
        case PlatformExecutionState::COMPLETED: return "COMPLETED";
        case PlatformExecutionState::FAILED: return "FAILED";
    }
    return "UNKNOWN";
}

}  // namespace brain_cpp
