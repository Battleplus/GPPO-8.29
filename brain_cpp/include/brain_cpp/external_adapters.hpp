#pragma once

#include "brain_cpp/algorithm_interfaces.hpp"

#include <string>
#include <memory>

namespace brain_cpp {

struct MilpTaskAllocatorOptions {
    std::string milp_dir = "milp";
    std::string solver = "cbc";
    double time_limit_s = 3.0;
    int verbose = 0;
};

class MilpTaskAllocator : public ITaskAllocator {
public:
    explicit MilpTaskAllocator(MilpTaskAllocatorOptions options = {});
    ~MilpTaskAllocator() override;
    AlgorithmResult<std::vector<ReconTask>> allocateRecon(const MissionContext& context) override;
    AlgorithmResult<std::vector<StrikeTask>> allocateAction(
        const MissionContext& context, const std::vector<std::string>& targetIds,
        bool includeEngaged) override;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

struct MppiRoutePlannerOptions {
    std::string project_root = "brain_cpp/mppi_compat";
    double map_size_units = 3000.0;
    double meters_per_unit = 100.0;
    double terrain_vertical_exaggeration = 10.0;
    double recon_altitude_km = 8.0;
    double action_cruise_altitude_km = 3.0;
    double formation_spacing_units = 40.0;
    int num_samples = 512;
    int num_iterations = 5;
    int horizon = 50;
    bool verbose = false;
};

class MppiRoutePlanner : public IRoutePlanner {
public:
    explicit MppiRoutePlanner(MppiRoutePlannerOptions options = {});
    ~MppiRoutePlanner() override;

    AlgorithmResult<FormationPlan>
    planReconRoute(
        const MissionContext& context,
        const std::vector<ReconTask>& allocation) override;

    AlgorithmResult<FormationPlan>
    planActionRoute(
        const MissionContext& context,
        const std::vector<StrikeTask>& allocation,
        const std::vector<Position>& selectedPositions) override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

struct PerchPositionSelectorOptions {
    std::string python_executable = "python";
    std::string bridge_script = "perch/cpp_bridge.py";
    std::string attack_region_mode = "llm";
    std::string preference = "balanced";
    std::string terrain_mode = "scene";
    int top_k = 3;
    bool use_pymoo = false;
    bool attack_region_strict = true;
};

class PerchPositionSelector : public IPositionSelector {
public:
    explicit PerchPositionSelector(PerchPositionSelectorOptions options = {});

    AlgorithmResult<std::vector<Position>>
    select(
        const MissionContext& context,
        const std::vector<StrikeTask>& allocation) override;

private:
    PerchPositionSelectorOptions options_;
};

struct SarSearchPatrolOptions {
    std::string python_executable = "python3";
    std::string bridge_script = "search_planner/cpp_plan_bridge.py";
    // Non-empty value overrides sensor-based selection for every task.
    std::string pattern;
    double subarea_width_km = 25.0;
    double subarea_height_km = 25.0;
    double aoi_width_km = 50.0;
    double aoi_height_km = 50.0;
    double altitude_agl_m = 5000.0;
};

// Integration policy owned by brain_cpp. search_planner only receives the
// resolved pattern and remains independent of MILP sensor semantics.
std::string patrolPatternForSensor(const std::string& sensor);

class SarSearchPatrolPlanner : public IPatrolPlanner {
public:
    explicit SarSearchPatrolPlanner(SarSearchPatrolOptions options = {});

    AlgorithmResult<std::vector<PatrolPlan>>
    planPatrols(
        const MissionContext& context,
        const std::vector<ReconTask>& allocation,
        const FormationPlan& transitPlan) override;

private:
    SarSearchPatrolOptions options_;
};

class PpoAvoidanceController final : public IPpoAvoidanceController {
public:
    explicit PpoAvoidanceController(
        const std::string& modelPath,
        ppo_avoidance::Config config = {});

    ppo_avoidance::Vec3 compute(
        const ppo_avoidance::Vec3& position,
        const ppo_avoidance::Vec3& velocity,
        const ppo_avoidance::Vec3& waypoint,
        const std::vector<ppo_avoidance::Obstacle>& obstacles) const override;

private:
    ppo_avoidance::Controller controller_;
};

struct PpoBridgeOptions {
    std::string python_executable = "python3";
    std::string bridge_script = "ppo_allocation/cpp_bridge.py";
    std::string model_path =
        "ppo_allocation/results/models/run_20260605_210049/maskable_ppo_uav_task_allocation.zip";
    bool deterministic = true;
};

class PpoBridgeReallocator : public IPpoReallocator {
public:
    explicit PpoBridgeReallocator(PpoBridgeOptions options = {});

    AlgorithmResult<std::vector<ReconTask>>
    handlePlatformLoss(const MissionContext& context, const std::string& platformId) override;

    AlgorithmResult<std::vector<ReconTask>>
    handleTargetDiscovered(
        const MissionContext& context,
        const std::string& platformId,
        const std::string& targetId) override;

    AlgorithmResult<std::vector<ReconTask>>
    handleTargetDestroyed(const MissionContext& context, const std::string& targetId) override;

private:
    PpoBridgeOptions options_;

    AlgorithmResult<std::vector<ReconTask>>
    runEvent(
        const MissionContext& context,
        const std::string& eventJson,
        const std::string& removedPlatform = "") const;
};

}  // namespace brain_cpp
