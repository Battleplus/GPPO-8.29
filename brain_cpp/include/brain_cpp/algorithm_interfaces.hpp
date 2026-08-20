#pragma once

#include "brain_cpp/domain.hpp"
#include "brain_cpp/mission_context.hpp"
#include "ppo_avoidance.hpp"

#include <string>
#include <vector>

namespace brain_cpp {

class ITaskAllocator {
public:
    virtual AlgorithmResult<std::vector<ReconTask>>
    allocateRecon(const MissionContext& context) = 0;

    virtual AlgorithmResult<std::vector<StrikeTask>>
    allocateAction(
        const MissionContext& context,
        const std::vector<std::string>& targetIds,
        bool includeEngaged) = 0;

    virtual ~ITaskAllocator() = default;
};

class IRoutePlanner {
public:
    virtual AlgorithmResult<FormationPlan>
    planReconRoute(
        const MissionContext& context,
        const std::vector<ReconTask>& allocation) = 0;

    virtual AlgorithmResult<FormationPlan>
    planActionRoute(
        const MissionContext& context,
        const std::vector<StrikeTask>& allocation,
        const std::vector<Position>& selectedPositions) = 0;

    virtual ~IRoutePlanner() = default;
};

class IPositionSelector {
public:
    virtual AlgorithmResult<std::vector<Position>>
    select(
        const MissionContext& context,
        const std::vector<StrikeTask>& allocation) = 0;

    virtual ~IPositionSelector() = default;
};

class IPatrolPlanner {
public:
    virtual AlgorithmResult<std::vector<PatrolPlan>>
    planPatrols(
        const MissionContext& context,
        const std::vector<ReconTask>& allocation,
        const FormationPlan& transitPlan) = 0;

    virtual ~IPatrolPlanner() = default;
};

class IPpoAvoidanceController {
public:
    virtual ppo_avoidance::Vec3 compute(
        const ppo_avoidance::Vec3& position,
        const ppo_avoidance::Vec3& velocity,
        const ppo_avoidance::Vec3& waypoint,
        const std::vector<ppo_avoidance::Obstacle>& obstacles) const = 0;

    virtual ~IPpoAvoidanceController() = default;
};

class IPpoReallocator {
public:
    virtual AlgorithmResult<std::vector<ReconTask>>
    handlePlatformLoss(const MissionContext& context, const std::string& platformId) = 0;

    virtual AlgorithmResult<std::vector<ReconTask>>
    handleTargetDiscovered(
        const MissionContext& context,
        const std::string& platformId,
        const std::string& targetId) = 0;

    virtual AlgorithmResult<std::vector<ReconTask>>
    handleTargetDestroyed(const MissionContext& context, const std::string& targetId) = 0;

    virtual ~IPpoReallocator() = default;
};

}  // namespace brain_cpp
