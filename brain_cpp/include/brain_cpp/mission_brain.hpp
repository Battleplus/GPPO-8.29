#pragma once

#include "brain_cpp/algorithm_interfaces.hpp"
#include "brain_cpp/domain.hpp"
#include "brain_cpp/environment_runtime.hpp"
#include "brain_cpp/mission_context.hpp"
#include "brain_cpp/mission_event.hpp"
#include "brain_cpp/mission_fsm.hpp"

#include <memory>
#include <string>
#include <vector>

namespace brain_cpp {

struct BrainRuntimeResult {
    std::vector<ReconTask> recon_allocation;
    std::vector<StrikeTask> action_allocation;
    std::vector<Position> selected_positions;
    FormationPlan recon_formation_plan;
    FormationPlan action_formation_plan;
    std::vector<PatrolPlan> recon_patrol_plans;
    std::vector<TargetTrackingStatus> target_tracks;
};

class MissionBrain {
public:
    MissionBrain(
        MissionContext& context,
        ITaskAllocator& taskAllocator,
        IRoutePlanner& routePlanner,
        IPositionSelector& positionSelector,
        IPpoReallocator* ppoReallocator = nullptr,
        IPatrolPlanner* patrolPlanner = nullptr);

    MissionBrain(
        MissionContext& context,
        ITaskAllocator& taskAllocator,
        IRoutePlanner& routePlanner,
        IPositionSelector& positionSelector,
        IEnvironmentRuntime* environmentRuntime,
        IPpoReallocator* ppoReallocator = nullptr,
        IPatrolPlanner* patrolPlanner = nullptr);

    MissionState start();
    MissionState dispatch(const MissionEvent& event);

    AlgorithmResult<BrainRuntimeResult> handleTargetDetected(const MissionEvent& event);
    AlgorithmResult<BrainRuntimeResult> handleTargetConfirmed(const MissionEvent& event);
    AlgorithmResult<BrainRuntimeResult>
    ingestSensorContacts(const std::vector<SensorContact>& contacts);
    AlgorithmResult<BrainRuntimeResult> stepEnvironment(double dt);
    AlgorithmResult<BrainRuntimeResult> handlePlatformLoss(const MissionEvent& event);
    AlgorithmResult<BrainRuntimeResult> handleAttackFinished(const MissionEvent& event);
    AlgorithmResult<std::vector<PatrolPlan>> buildReconPatrolPlans();

    MissionState currentState() const;

    const MissionContext& context() const;

private:
    MissionContext& ctx_;
    ITaskAllocator& taskAllocator_;
    IRoutePlanner& routePlanner_;
    IPositionSelector& positionSelector_;
    IEnvironmentRuntime* environmentRuntime_ = nullptr;
    IPpoReallocator* ppoReallocator_ = nullptr;
    IPatrolPlanner* patrolPlanner_ = nullptr;
    MissionFSM fsm_;

    AlgorithmResult<EnvironmentSnapshot> initializeEnvironment();
    void syncEnvironmentSnapshot(const EnvironmentSnapshot& snapshot);
    void refreshReconPatrolPlans(const std::string& source);

    AlgorithmResult<BrainRuntimeResult>
    buildActionPlan(const std::vector<std::string>& targetIds, const std::string& source);

    TargetInfo& ensureTarget(const std::string& targetId);
    AgentSpec* findAgent(const std::string& platformId);
    void discardTargetFromQueues(const std::string& targetId);
    BrainRuntimeResult snapshot() const;
};

}  // namespace brain_cpp
