#pragma once

#include "brain_cpp/domain.hpp"

#include <map>
#include <string>
#include <vector>

namespace brain_cpp {

struct MissionContext;

struct SensorContact {
    std::string platform_id;
    std::string target_id;
    std::string sensor;
    std::string channel;
    double distance_km = 0.0;
    int priority = 0;
};

struct EnvironmentSnapshot {
    bool initialized = false;
    std::string name;
    double tactical_time_s = 0.0;

    std::vector<AgentSpec> agents;
    std::vector<TargetInfo> targets;
    std::vector<SensorContact> sensor_contacts;
    std::vector<Aoi> aois;
    Point2 staging_position{150.0, -50.0};
    std::map<std::string, double> weather;
    std::map<std::string, int> terrain;
};

class IEnvironmentRuntime {
public:
    virtual AlgorithmResult<EnvironmentSnapshot>
    initialize(const MissionContext& context) = 0;

    virtual AlgorithmResult<EnvironmentSnapshot>
    reset(const MissionContext& context) = 0;

    virtual AlgorithmResult<EnvironmentSnapshot>
    step(const MissionContext& context, double dt) = 0;

    virtual void close() {}

    virtual ~IEnvironmentRuntime() = default;
};

}  // namespace brain_cpp
