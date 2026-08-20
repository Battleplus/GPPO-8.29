#pragma once

#include "brain_cpp/domain.hpp"

#include <map>
#include <string>
#include <vector>

namespace brain_cpp {

enum class PlatformExecutionState { READY, MOVING, COMPLETED, FAILED };

struct PlatformExecutionStatus {
    std::string platform;
    PlatformExecutionState state = PlatformExecutionState::READY;
    Point3 position{0.0, 0.0, 0.0};
    std::size_t waypoint_index = 0;
    std::size_t waypoint_count = 0;
    double distance_travelled = 0.0;
    std::string error;
};

struct ParallelExecutionSnapshot {
    double elapsed_s = 0.0;
    std::size_t tick = 0;
    bool all_completed = false;
    std::vector<PlatformExecutionStatus> platforms;
};

// Advances every assigned platform once per simulation tick. Routes belonging to
// the same platform are joined into one queue so repeated MILP sensor tasks do
// not create duplicate aircraft.
class ParallelPlatformController {
public:
    explicit ParallelPlatformController(double speed_units_per_s = 20.0);

    AlgorithmResult<ParallelExecutionSnapshot>
    load(const std::vector<AgentSpec>& agents, const std::vector<Route>& routes);

    AlgorithmResult<ParallelExecutionSnapshot> step(double dt);
    ParallelExecutionSnapshot snapshot() const;
    bool allCompleted() const;

private:
    struct Execution {
        PlatformExecutionStatus status;
        std::vector<Point3> waypoints;
    };

    double speed_ = 20.0;
    double elapsed_s_ = 0.0;
    std::size_t tick_ = 0;
    std::map<std::string, Execution> executions_;
};

std::string toString(PlatformExecutionState state);

}  // namespace brain_cpp
