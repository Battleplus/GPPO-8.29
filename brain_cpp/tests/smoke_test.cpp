#include "brain_cpp/external_adapters.hpp"
#include "brain_cpp/mission_brain.hpp"
#include "brain_cpp/parallel_platform_controller.hpp"
#include "brain_cpp/scenario_initializer.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

class TestEnvironmentRuntime : public brain_cpp::IEnvironmentRuntime {
public:
    brain_cpp::AlgorithmResult<brain_cpp::EnvironmentSnapshot>
    initialize(const brain_cpp::MissionContext& context) override {
        brain_cpp::EnvironmentSnapshot snapshot;
        snapshot.initialized = true;
        snapshot.name = "TestEnvironmentRuntime";
        snapshot.agents = context.agents.empty()
            ? brain_cpp::ScenarioInitializer::buildDefaultAgents()
            : context.agents;
        const auto defaultWorld = brain_cpp::ScenarioInitializer::buildDefaultWorld();
        snapshot.targets = context.world.targets.empty()
            ? defaultWorld.targets
            : context.world.targets;
        snapshot.aois = {context.world.aoi};
        snapshot.staging_position = context.world.staging_position;
        return brain_cpp::AlgorithmResult<brain_cpp::EnvironmentSnapshot>::ok(snapshot);
    }

    brain_cpp::AlgorithmResult<brain_cpp::EnvironmentSnapshot>
    reset(const brain_cpp::MissionContext& context) override {
        return initialize(context);
    }

    brain_cpp::AlgorithmResult<brain_cpp::EnvironmentSnapshot>
    step(const brain_cpp::MissionContext& context, double) override {
        return initialize(context);
    }
};

class TestTaskAllocator : public brain_cpp::ITaskAllocator {
public:
    brain_cpp::AlgorithmResult<std::vector<brain_cpp::ReconTask>>
    allocateRecon(const brain_cpp::MissionContext& context) override {
        static const std::vector<std::string> cells = {"c0", "c1", "c2", "c3", "c4"};
        std::vector<brain_cpp::ReconTask> tasks;
        for (const auto& agent : context.agents) {
            if (agent.type != "UAV" || agent.lost) continue;
            const auto index = tasks.size();
            tasks.push_back({
                agent.pid,
                cells[index % cells.size()],
                agent.sensors.empty() ? "SAR" : agent.sensors.front(),
                index == 0 ? "area_scan" : "subarea_search",
                context.world.aoi.id,
            });
        }
        if (tasks.empty()) {
            return brain_cpp::AlgorithmResult<std::vector<brain_cpp::ReconTask>>::fail(
                "No test UAV platforms");
        }
        return brain_cpp::AlgorithmResult<std::vector<brain_cpp::ReconTask>>::ok(tasks);
    }

    brain_cpp::AlgorithmResult<std::vector<brain_cpp::StrikeTask>>
    allocateAction(
        const brain_cpp::MissionContext& context,
        const std::vector<std::string>& targetIds,
        bool) override {
        std::vector<brain_cpp::StrikeTask> tasks;
        std::size_t targetIndex = 0;
        for (const auto& agent : context.agents) {
            if (agent.type != "HELI" || agent.lost || targetIndex >= targetIds.size()) continue;
            brain_cpp::StrikeTask task;
            task.platform = agent.pid;
            task.target = targetIds[targetIndex++];
            task.munition = "HF";
            task.qty = 1;
            task.role = "lead";
            task.aoi = context.world.aoi.id;
            task.assigned_munitions = {{"HF", 1}};
            tasks.push_back(task);
        }
        if (tasks.empty()) {
            return brain_cpp::AlgorithmResult<std::vector<brain_cpp::StrikeTask>>::fail(
                "No test strike assignment");
        }
        return brain_cpp::AlgorithmResult<std::vector<brain_cpp::StrikeTask>>::ok(tasks);
    }
};

class TestRoutePlanner : public brain_cpp::IRoutePlanner {
public:
    explicit TestRoutePlanner(int failReconAttempts = 0)
        : failReconAttempts_(failReconAttempts) {}

    brain_cpp::AlgorithmResult<brain_cpp::FormationPlan>
    planReconRoute(
        const brain_cpp::MissionContext&,
        const std::vector<brain_cpp::ReconTask>& allocation) override {
        if (failReconAttempts_ > 0) {
            --failReconAttempts_;
            return brain_cpp::AlgorithmResult<brain_cpp::FormationPlan>::fail(
                "Test recon planner forced failure");
        }
        brain_cpp::FormationPlan plan;
        plan.success = true;
        for (const auto& task : allocation) {
            brain_cpp::Route route;
            route.platform = task.platform;
            route.role = task.role;
            route.waypoints = {{{0.0, 0.0, 2.0}}, {{1.0, 1.0, 2.0}}};
            plan.routes.push_back(route);
        }
        return brain_cpp::AlgorithmResult<brain_cpp::FormationPlan>::ok(plan);
    }

    brain_cpp::AlgorithmResult<brain_cpp::FormationPlan>
    planActionRoute(
        const brain_cpp::MissionContext&,
        const std::vector<brain_cpp::StrikeTask>& allocation,
        const std::vector<brain_cpp::Position>& positions) override {
        if (allocation.empty() || positions.size() < allocation.size()) {
            return brain_cpp::AlgorithmResult<brain_cpp::FormationPlan>::fail(
                "Missing test strike positions");
        }
        brain_cpp::FormationPlan plan;
        plan.success = true;
        for (std::size_t index = 0; index < allocation.size(); ++index) {
            brain_cpp::Route route;
            route.platform = allocation[index].platform;
            route.target_id = allocation[index].target;
            route.position_id = positions[index].pos_id;
            route.waypoints = {{{positions[index].x, positions[index].y, positions[index].z}}};
            plan.routes.push_back(route);
        }
        return brain_cpp::AlgorithmResult<brain_cpp::FormationPlan>::ok(plan);
    }

private:
    int failReconAttempts_ = 0;
};

class TestPositionSelector : public brain_cpp::IPositionSelector {
public:
    explicit TestPositionSelector(int failAttempts = 0)
        : failAttempts_(failAttempts) {}

    brain_cpp::AlgorithmResult<std::vector<brain_cpp::Position>>
    select(
        const brain_cpp::MissionContext& context,
        const std::vector<brain_cpp::StrikeTask>& allocation) override {
        if (failAttempts_ > 0) {
            --failAttempts_;
            return brain_cpp::AlgorithmResult<std::vector<brain_cpp::Position>>::fail(
                "Test position selector forced failure");
        }
        std::map<std::string, brain_cpp::TargetInfo> targets;
        for (const auto& target : context.world.targets) targets[target.tid] = target;
        std::vector<brain_cpp::Position> result;
        for (const auto& task : allocation) {
            const auto target = targets.find(task.target);
            if (target == targets.end()) {
                return brain_cpp::AlgorithmResult<std::vector<brain_cpp::Position>>::fail(
                    "Unknown test target");
            }
            brain_cpp::Position position;
            position.pos_id = task.platform + "_" + task.target + "_TEST";
            position.x = target->second.pos[0] - 20.0;
            position.y = target->second.pos[1];
            position.z = 3.0;
            position.metadata = {
                {"platform_id", task.platform},
                {"target_id", task.target},
            };
            result.push_back(position);
        }
        return brain_cpp::AlgorithmResult<std::vector<brain_cpp::Position>>::ok(result);
    }

private:
    int failAttempts_ = 0;
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << "\n";
        std::exit(1);
    }
}

brain_cpp::MissionContext makeContext(int maxRetry = 3) {
    brain_cpp::MissionContext context;
    context.max_retry = maxRetry;
    brain_cpp::StartupOptions options;
    options.mission_id = "CPP_SMOKE";
    options.aoi = "A_3_4";
    brain_cpp::ScenarioInitializer().normalize(context, options);
    return context;
}

std::vector<std::string> targetIds(const brain_cpp::MissionContext& context) {
    std::vector<std::string> ids;
    for (const auto& target : context.world.targets) {
        ids.push_back(target.tid);
    }
    return ids;
}

bool historyHas(const brain_cpp::MissionContext& context, const std::string& event) {
    for (const auto& item : context.history) {
        if (item.event == event) {
            return true;
        }
    }
    return false;
}

void testStartInitializesEnvironmentBeforeMilp() {
    brain_cpp::MissionContext context;
    context.mission_id = "CPP_ENV_SMOKE";
    TestTaskAllocator taskAllocator;
    TestRoutePlanner routePlanner;
    TestPositionSelector positionSelector;
    TestEnvironmentRuntime environment;
    brain_cpp::MissionBrain brain(
        context,
        taskAllocator,
        routePlanner,
        positionSelector,
        &environment);

    const auto state = brain.start();
    require(state == brain_cpp::MissionState::RECON_PLAN_READY, "environment start reaches RECON_PLAN_READY");
    require(context.environment_initialized, "environment initialized flag is set");
    require(context.environment_name == "TestEnvironmentRuntime", "environment name synced");
    require(context.agents.size() == 7, "environment supplies 5 UAV + 2 HELI");
    require(context.world.targets.size() == 3, "environment supplies default targets");
    require(historyHas(context, "ENVIRONMENT_INITIALIZED"), "environment event recorded");
    require(historyHas(context, "RECON_ALLOCATING"), "MILP allocation happens after environment init");
    require(brain.stepEnvironment(0.5).success, "configured environment can step");
}

void testSensorPatternMappingLivesInBrainCpp() {
    require(brain_cpp::patrolPatternForSensor("SAR") == "sar_polygon", "SAR pattern mapping");
    require(brain_cpp::patrolPatternForSensor("sar") == "sar_polygon", "sensor mapping is case-insensitive");
    require(brain_cpp::patrolPatternForSensor("EO") == "racetrack", "EO pattern mapping");
    require(brain_cpp::patrolPatternForSensor("EOIR") == "racetrack", "EOIR pattern mapping");
    require(brain_cpp::patrolPatternForSensor("ESM") == "figure_eight", "ESM pattern mapping");
    require(brain_cpp::patrolPatternForSensor("MMW") == "sar_rounded", "MMW pattern mapping");
}

void testFullFlow() {
    auto context = makeContext();
    TestTaskAllocator taskAllocator;
    TestRoutePlanner routePlanner;
    TestPositionSelector positionSelector;
    brain_cpp::MissionBrain brain(context, taskAllocator, routePlanner, positionSelector);

    auto state = brain.start();
    require(state == brain_cpp::MissionState::RECON_PLAN_READY, "start reaches RECON_PLAN_READY");
    require(!context.recon_allocation.empty(), "recon allocation exists");
    require(context.recon_formation_plan.success, "recon formation plan exists");

    state = brain.dispatch(brain_cpp::MissionEvent::reconPlanDispatched());
    require(state == brain_cpp::MissionState::RECON_EXECUTING, "dispatch recon plan");

    state = brain.dispatch(brain_cpp::MissionEvent::reconFinished());
    require(state == brain_cpp::MissionState::WAIT_RECON_RESULT, "dispatch recon finished");

    state = brain.dispatch(brain_cpp::MissionEvent::reconResultReceived(targetIds(context)));
    require(state == brain_cpp::MissionState::ACTION_PLAN_READY, "recon result reaches ACTION_PLAN_READY");
    require(!context.action_allocation.empty(), "action allocation exists");
    require(!context.selected_positions.empty(), "positions exist");
    require(context.action_formation_plan.success, "action formation plan exists");

    state = brain.dispatch(brain_cpp::MissionEvent::actionPlanDispatched());
    require(state == brain_cpp::MissionState::ACTION_EXECUTING, "dispatch action plan");

    state = brain.dispatch(brain_cpp::MissionEvent::actionFinished());
    require(state == brain_cpp::MissionState::MISSION_COMPLETE, "action finished completes mission");
    require(historyHas(context, "TERMINAL"), "terminal event recorded");
}

void testTransientMppiFailureRetries() {
    auto context = makeContext();
    TestTaskAllocator taskAllocator;
    TestRoutePlanner routePlanner(1);
    TestPositionSelector positionSelector;
    brain_cpp::MissionBrain brain(context, taskAllocator, routePlanner, positionSelector);

    const auto state = brain.start();
    require(state == brain_cpp::MissionState::RECON_PLAN_READY, "transient recon MPPI failure recovers");
    require(context.retry_count == 1, "retry count increments once");
    require(historyHas(context, "REPLAN"), "replan event recorded");
}

void testPersistentMppiFailureFails() {
    auto context = makeContext(2);
    TestTaskAllocator taskAllocator;
    TestRoutePlanner routePlanner(10);
    TestPositionSelector positionSelector;
    brain_cpp::MissionBrain brain(context, taskAllocator, routePlanner, positionSelector);

    const auto state = brain.start();
    require(state == brain_cpp::MissionState::MISSION_FAILED, "persistent MPPI failure fails mission");
    require(context.retry_count == 3, "retry exceeds max_retry before failure");
}

void testTransientPositionFailureRetries() {
    auto context = makeContext();
    TestTaskAllocator taskAllocator;
    TestRoutePlanner routePlanner;
    TestPositionSelector positionSelector(1);
    brain_cpp::MissionBrain brain(context, taskAllocator, routePlanner, positionSelector);

    auto state = brain.start();
    require(state == brain_cpp::MissionState::RECON_PLAN_READY, "start for position retry test");
    state = brain.dispatch(brain_cpp::MissionEvent::reconPlanDispatched());
    state = brain.dispatch(brain_cpp::MissionEvent::reconFinished());
    state = brain.dispatch(brain_cpp::MissionEvent::reconResultReceived({"g1"}));

    require(state == brain_cpp::MissionState::ACTION_PLAN_READY, "transient position failure recovers");
    require(context.retry_count == 1, "position retry increments once");
    require(historyHas(context, "REPLAN"), "position retry records replan");
}

void testSensorContactWaitsForConfirmationBeforeAttackPlanning() {
    auto context = makeContext();
    TestTaskAllocator taskAllocator;
    TestRoutePlanner routePlanner;
    TestPositionSelector positionSelector;
    brain_cpp::MissionBrain brain(context, taskAllocator, routePlanner, positionSelector);

    require(brain.start() == brain_cpp::MissionState::RECON_PLAN_READY, "tracking test starts recon");
    brain_cpp::SensorContact contact;
    contact.platform_id = "Blue_CH4_Recon";
    contact.target_id = "g1";
    contact.sensor = "EO_IR_Gimbal";
    contact.channel = "eo_ir";
    contact.distance_km = 12.5;
    contact.priority = 90;

    auto detected = brain.ingestSensorContacts({contact});
    require(detected.success, "sensor contact ingestion succeeds");
    require(context.action_allocation.empty(), "detection does not call attack MILP");
    require(context.selected_positions.empty(), "detection does not call position selector");
    require(!context.world.targets.front().confirmed, "detected target is not yet confirmed");
    const auto tracking = context.target_tracks.find("g1");
    require(tracking != context.target_tracks.end(), "target tracking status exists");
    require(
        tracking->second.state == brain_cpp::TargetTrackingState::APPROACH_CONFIRMING,
        "target waits in approach confirmation");
    require(tracking->second.platform_command == "HOVER", "tracking platform receives hover command");

    auto confirmed = brain.handleTargetConfirmed(
        brain_cpp::MissionEvent::targetConfirmed("g1", "Blue_CH4_Recon"));
    require(confirmed.success, "target confirmation builds attack plan");
    require(context.world.targets.front().confirmed, "confirmed target updates world state");
    require(!context.action_allocation.empty(), "confirmation calls attack MILP");
    require(!context.selected_positions.empty(), "confirmation calls position selector");
    require(context.action_formation_plan.success, "confirmation calls action route planner");
    require(
        context.target_tracks.at("g1").state == brain_cpp::TargetTrackingState::ATTACK_PLAN_READY,
        "tracking state reaches attack plan ready");
    require(brain.currentState() == brain_cpp::MissionState::ACTION_PLAN_READY, "mission exposes action plan");
}

void testRuntimeLossAndAttackEvents() {
    auto context = makeContext();
    TestTaskAllocator taskAllocator;
    TestRoutePlanner routePlanner;
    TestPositionSelector positionSelector;
    brain_cpp::MissionBrain brain(context, taskAllocator, routePlanner, positionSelector);

    require(brain.start() == brain_cpp::MissionState::RECON_PLAN_READY, "runtime event test starts");
    auto lost = brain.handlePlatformLoss(brain_cpp::MissionEvent::platformLost("U1"));
    require(lost.success, "platform loss replans recon routes");
    require(context.agents.front().lost, "platform loss updates agent state");

    require(
        brain.handleTargetDetected(brain_cpp::MissionEvent::targetDetected("g1", "U2")).success,
        "runtime target detection succeeds");
    require(
        brain.handleTargetConfirmed(brain_cpp::MissionEvent::targetConfirmed("g1", "U2")).success,
        "runtime target confirmation succeeds");
    auto finished = brain.handleAttackFinished(
        brain_cpp::MissionEvent::attackFinished("g1", true));
    require(finished.success, "attack-finished event succeeds");
    require(!context.world.targets.front().alive, "destroyed target is removed from active world state");
}

void testParallelPlatformControllerMovesAllPlatformsPerTick() {
    std::vector<brain_cpp::AgentSpec> agents(2);
    agents[0].pid = "U0";
    agents[0].position = {0.0, 0.0};
    agents[0].altitude_km = 0.0;
    agents[1].pid = "U1";
    agents[1].position = {0.0, 10.0};
    agents[1].altitude_km = 0.0;

    std::vector<brain_cpp::Route> routes = {
        {"U0", {{10.0, 0.0, 0.0}}},
        {"U1", {{0.0, 20.0, 0.0}}},
        {"U0", {{10.0, 0.0, 0.0}, {20.0, 0.0, 0.0}}},
    };
    brain_cpp::ParallelPlatformController controller(5.0);
    auto loaded = controller.load(agents, routes);
    require(loaded.success, "parallel routes load");
    require(loaded.data.platforms.size() == 2, "duplicate tasks produce two physical platforms");

    auto tick = controller.step(1.0);
    require(tick.success, "parallel tick succeeds");
    require(tick.data.tick == 1, "one shared simulation tick");
    require(tick.data.platforms[0].distance_travelled == 5.0, "first platform moved in tick");
    require(tick.data.platforms[1].distance_travelled == 5.0, "second platform moved in same tick");

    controller.step(1.0);
    require(!controller.allCompleted(), "longer route remains active independently");
    controller.step(2.0);
    require(controller.allCompleted(), "all platforms eventually complete");
}

void testPpoAvoidanceInterface(const std::string& modelPath) {
    brain_cpp::PpoAvoidanceController controller(modelPath);
    const brain_cpp::IPpoAvoidanceController& interface = controller;
    const ppo_avoidance::Vec3 position{10.0, -5.0, 8.0};
    const ppo_avoidance::Vec3 velocity{7.0, 2.0, -0.5};
    const ppo_avoidance::Vec3 waypoint{180.0, 95.0, 15.0};
    const std::vector<ppo_avoidance::Obstacle> obstacles = {
        {{42.0, 16.0, 0.0}, 8.0},
    };

    const auto acceleration = interface.compute(position, velocity, waypoint, obstacles);
    require(std::isfinite(acceleration.x) && std::isfinite(acceleration.y)
                && std::isfinite(acceleration.z),
            "PPO acceleration interface");
}

}  // namespace

int main(int argc, char** argv) {
    require(argc == 2, "PPO model path is required");
    testSensorPatternMappingLivesInBrainCpp();
    testStartInitializesEnvironmentBeforeMilp();
    testFullFlow();
    testTransientMppiFailureRetries();
    testPersistentMppiFailureFails();
    testTransientPositionFailureRetries();
    testSensorContactWaitsForConfirmationBeforeAttackPlanning();
    testRuntimeLossAndAttackEvents();
    testParallelPlatformControllerMovesAllPlatformsPerTick();
    testPpoAvoidanceInterface(argv[1]);
    std::cout << "brain_cpp smoke tests passed\n";
    return 0;
}
