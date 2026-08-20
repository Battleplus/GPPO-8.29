#include "brain_cpp/mission_brain.hpp"

#include <algorithm>

namespace brain_cpp {

MissionBrain::MissionBrain(
    MissionContext& context,
    ITaskAllocator& taskAllocator,
    IRoutePlanner& routePlanner,
    IPositionSelector& positionSelector,
    IPpoReallocator* ppoReallocator,
    IPatrolPlanner* patrolPlanner)
    : MissionBrain(
          context,
          taskAllocator,
          routePlanner,
          positionSelector,
          nullptr,
          ppoReallocator,
          patrolPlanner) {}

MissionBrain::MissionBrain(
    MissionContext& context,
    ITaskAllocator& taskAllocator,
    IRoutePlanner& routePlanner,
    IPositionSelector& positionSelector,
    IEnvironmentRuntime* environmentRuntime,
    IPpoReallocator* ppoReallocator,
    IPatrolPlanner* patrolPlanner)
    : ctx_(context),
      taskAllocator_(taskAllocator),
      routePlanner_(routePlanner),
      positionSelector_(positionSelector),
      environmentRuntime_(environmentRuntime),
      ppoReallocator_(ppoReallocator),
      patrolPlanner_(patrolPlanner),
      fsm_(context, taskAllocator, routePlanner, positionSelector) {}

MissionState MissionBrain::start() {
    if (ctx_.state == MissionState::INIT) {
        auto initialized = initializeEnvironment();
        if (!initialized.success) {
            ctx_.last_error = initialized.reason;
            ctx_.state = MissionState::MISSION_FAILED;
            ctx_.recordEvent("ENVIRONMENT_INIT_FAILED", initialized.reason);
            return ctx_.state;
        }
    }
    const auto state = fsm_.dispatch(MissionEvent::start());
    if (state == MissionState::RECON_PLAN_READY) {
        refreshReconPatrolPlans("START");
    }
    return state;
}

MissionState MissionBrain::dispatch(const MissionEvent& event) {
    if (event.type == MissionEventType::TARGET_DETECTED) {
        (void)handleTargetDetected(event);
        return ctx_.state;
    }
    if (event.type == MissionEventType::TARGET_CONFIRMED) {
        (void)handleTargetConfirmed(event);
        return ctx_.state;
    }
    if (event.type == MissionEventType::PLATFORM_LOST) {
        (void)handlePlatformLoss(event);
        return ctx_.state;
    }
    if (event.type == MissionEventType::ATTACK_FINISHED) {
        (void)handleAttackFinished(event);
        return ctx_.state;
    }
    return fsm_.dispatch(event);
}

AlgorithmResult<BrainRuntimeResult> MissionBrain::handleTargetDetected(const MissionEvent& event) {
    if (event.target_id.empty()) {
        return AlgorithmResult<BrainRuntimeResult>::fail("TARGET_DETECTED missing target_id");
    }

    auto& tracking = ctx_.target_tracks[event.target_id];
    if (tracking.target_id.empty()) tracking.target_id = event.target_id;
    if (!event.platform_id.empty()) tracking.platform_id = event.platform_id;
    if (tracking.state != TargetTrackingState::UNTRACKED) {
        ctx_.recordEvent(
            "TARGET_CONTACT_REFRESHED",
            event.target_id + " remains in " + toString(tracking.state));
        return AlgorithmResult<BrainRuntimeResult>::ok(snapshot());
    }

    TargetTrackingFSM trackingFsm(tracking);
    if (!trackingFsm.dispatch(TargetTrackingEvent::SENSOR_CONTACT)
        || !trackingFsm.dispatch(TargetTrackingEvent::START_TRACKING)
        || !trackingFsm.dispatch(TargetTrackingEvent::REQUEST_APPROACH_CONFIRMATION)) {
        return AlgorithmResult<BrainRuntimeResult>::fail(
            "Failed to enter approach confirmation for " + event.target_id);
    }

    auto& target = ensureTarget(event.target_id);
    target.alive = true;
    target.confirmed = false;

    if (ppoReallocator_ != nullptr && !event.platform_id.empty()) {
        auto ppo = ppoReallocator_->handleTargetDiscovered(ctx_, event.platform_id, event.target_id);
        if (ppo.success) {
            ctx_.recon_allocation = ppo.data;
            auto route = routePlanner_.planReconRoute(ctx_, ctx_.recon_allocation);
            if (route.success) {
                ctx_.recon_formation_plan = route.data;
                refreshReconPatrolPlans("TARGET_DETECTED");
            } else {
                ctx_.recordEvent("PPO_RECON_ROUTE_FAILED", route.reason);
            }
        } else {
            ctx_.recordEvent("PPO_REALLOCATION_FAILED", ppo.reason);
        }
    }

    ctx_.recordEvent(
        "TARGET_DETECTED",
        event.target_id + " tracking started; approach confirmation command=HOVER");
    return AlgorithmResult<BrainRuntimeResult>::ok(snapshot());
}

AlgorithmResult<BrainRuntimeResult> MissionBrain::handleTargetConfirmed(
    const MissionEvent& event) {
    if (event.target_id.empty()) {
        return AlgorithmResult<BrainRuntimeResult>::fail("TARGET_CONFIRMED missing target_id");
    }
    const auto found = ctx_.target_tracks.find(event.target_id);
    if (found == ctx_.target_tracks.end()) {
        return AlgorithmResult<BrainRuntimeResult>::fail(
            "TARGET_CONFIRMED received before target tracking: " + event.target_id);
    }

    auto& tracking = found->second;
    if (!event.platform_id.empty()) tracking.platform_id = event.platform_id;
    if (tracking.state == TargetTrackingState::ATTACK_PLAN_READY) {
        return AlgorithmResult<BrainRuntimeResult>::ok(snapshot());
    }
    TargetTrackingFSM trackingFsm(tracking);
    if (!trackingFsm.dispatch(TargetTrackingEvent::TARGET_CONFIRMED)) {
        return AlgorithmResult<BrainRuntimeResult>::fail(
            "Illegal target confirmation from " + toString(tracking.state));
    }

    auto& target = ensureTarget(event.target_id);
    target.confirmed = true;
    target.alive = true;
    if (ctx_.engaged_targets.find(target.tid) == ctx_.engaged_targets.end()) {
        appendUnique(ctx_.pending_strike_targets, target.tid);
    }
    if (!trackingFsm.dispatch(TargetTrackingEvent::START_ATTACK_PLANNING)) {
        return AlgorithmResult<BrainRuntimeResult>::fail(
            "Failed to start attack planning for " + event.target_id);
    }

    auto result = buildActionPlan({event.target_id}, "TARGET_CONFIRMED");
    if (!result.success) {
        (void)trackingFsm.dispatch(TargetTrackingEvent::ATTACK_PLAN_FAILED, result.reason);
        ctx_.recordEvent("TARGET_ATTACK_PLAN_FAILED", event.target_id + ": " + result.reason);
        return result;
    }
    if (!trackingFsm.dispatch(TargetTrackingEvent::ATTACK_PLAN_SUCCEEDED)) {
        return AlgorithmResult<BrainRuntimeResult>::fail(
            "Failed to finalize attack plan state for " + event.target_id);
    }

    const auto previousState = ctx_.state;
    ctx_.state = MissionState::ACTION_PLAN_READY;
    if (previousState != ctx_.state) {
        ctx_.recordEvent(
            "STATE_CHANGED",
            toString(previousState) + " -> ACTION_PLAN_READY (TARGET_CONFIRMED)");
    }
    ctx_.recordEvent(
        "TARGET_CONFIRMED",
        event.target_id + " confirmed; MILP allocation, Perch selection, and action route ready");
    return AlgorithmResult<BrainRuntimeResult>::ok(snapshot());
}

AlgorithmResult<BrainRuntimeResult> MissionBrain::ingestSensorContacts(
    const std::vector<SensorContact>& contacts) {
    for (const auto& contact : contacts) {
        if (contact.platform_id.empty() || contact.target_id.empty()) continue;
        auto& tracking = ctx_.target_tracks[contact.target_id];
        if (tracking.sensor.empty()) tracking.sensor = contact.sensor;
        auto result = handleTargetDetected(MissionEvent::targetDetected(
            contact.target_id, contact.platform_id));
        if (!result.success) return result;
    }
    return AlgorithmResult<BrainRuntimeResult>::ok(snapshot());
}

AlgorithmResult<BrainRuntimeResult> MissionBrain::stepEnvironment(double dt) {
    if (environmentRuntime_ == nullptr) {
        return AlgorithmResult<BrainRuntimeResult>::fail("Environment runtime not configured");
    }
    auto stepped = environmentRuntime_->step(ctx_, dt);
    if (!stepped.success) {
        return AlgorithmResult<BrainRuntimeResult>::fail(stepped.reason);
    }
    syncEnvironmentSnapshot(stepped.data);
    return ingestSensorContacts(stepped.data.sensor_contacts);
}

AlgorithmResult<BrainRuntimeResult> MissionBrain::handlePlatformLoss(const MissionEvent& event) {
    if (event.platform_id.empty()) {
        return AlgorithmResult<BrainRuntimeResult>::fail("PLATFORM_LOST missing platform_id");
    }

    auto* agent = findAgent(event.platform_id);
    if (agent != nullptr) {
        agent->lost = true;
    }

    const std::string platformType = agent != nullptr ? agent->type : "UAV";
    if (platformType == "HELI") {
        std::vector<std::string> affectedTargets;
        for (const auto& task : ctx_.action_allocation) {
            if (task.platform == event.platform_id && !task.target.empty()) {
                affectedTargets.push_back(task.target);
            }
        }

        for (const auto& targetId : affectedTargets) {
            ctx_.engaged_targets.erase(targetId);
            appendUnique(ctx_.pending_strike_targets, targetId);
        }

        if (affectedTargets.empty()) {
            ctx_.recordEvent("PLATFORM_LOST", event.platform_id + " lost; no active strike target affected");
            return AlgorithmResult<BrainRuntimeResult>::ok(snapshot());
        }
        return buildActionPlan(affectedTargets, "HELI_LOST");
    }

    AlgorithmResult<std::vector<ReconTask>> realloc =
        AlgorithmResult<std::vector<ReconTask>>::fail("PPO reallocator not configured");
    if (ppoReallocator_ != nullptr) {
        realloc = ppoReallocator_->handlePlatformLoss(ctx_, event.platform_id);
    }

    if (realloc.success) {
        ctx_.recon_allocation = realloc.data;
    } else {
        ctx_.recordEvent("PPO_REALLOCATION_FAILED", realloc.reason);
        ctx_.recon_allocation.erase(
            std::remove_if(
                ctx_.recon_allocation.begin(),
                ctx_.recon_allocation.end(),
                [&](const ReconTask& task) { return task.platform == event.platform_id; }),
            ctx_.recon_allocation.end());
    }

    auto route = routePlanner_.planReconRoute(ctx_, ctx_.recon_allocation);
    if (!route.success) {
        return AlgorithmResult<BrainRuntimeResult>::fail(
            "Recon reallocation succeeded but route planning failed: " + route.reason);
    }
    ctx_.recon_formation_plan = route.data;
    refreshReconPatrolPlans("PLATFORM_LOST");
    ctx_.state = MissionState::RECON_PLAN_READY;
    ctx_.recordEvent("PLATFORM_LOST", event.platform_id + " lost; recon routes regenerated");
    return AlgorithmResult<BrainRuntimeResult>::ok(snapshot());
}

AlgorithmResult<BrainRuntimeResult> MissionBrain::handleAttackFinished(const MissionEvent& event) {
    if (event.target_id.empty()) {
        return AlgorithmResult<BrainRuntimeResult>::fail("ATTACK_FINISHED missing target_id");
    }

    auto& target = ensureTarget(event.target_id);
    target.confirmed = true;

    if (event.destroyed) {
        target.alive = false;
        discardTargetFromQueues(event.target_id);
        if (ppoReallocator_ != nullptr) {
            auto ppo = ppoReallocator_->handleTargetDestroyed(ctx_, event.target_id);
            if (ppo.success) {
                ctx_.recon_allocation = ppo.data;
                auto route = routePlanner_.planReconRoute(ctx_, ctx_.recon_allocation);
                if (route.success) {
                    ctx_.recon_formation_plan = route.data;
                    refreshReconPatrolPlans("TARGET_DESTROYED");
                }
            }
        }
        ctx_.recordEvent("ATTACK_FINISHED", event.target_id + " destroyed");
        return AlgorithmResult<BrainRuntimeResult>::ok(snapshot());
    }

    target.alive = true;
    ctx_.engaged_targets.erase(event.target_id);
    appendUnique(ctx_.pending_strike_targets, event.target_id);
    ctx_.recordEvent("ATTACK_FAILED", event.target_id + " remains alive; rebuilding strike plan");
    return buildActionPlan({event.target_id}, "ATTACK_FAILED");
}

MissionState MissionBrain::currentState() const {
    return fsm_.currentState();
}

const MissionContext& MissionBrain::context() const {
    return ctx_;
}

AlgorithmResult<std::vector<PatrolPlan>> MissionBrain::buildReconPatrolPlans() {
    if (patrolPlanner_ == nullptr) {
        return AlgorithmResult<std::vector<PatrolPlan>>::fail("Patrol planner not configured");
    }

    auto result = patrolPlanner_->planPatrols(ctx_, ctx_.recon_allocation, ctx_.recon_formation_plan);
    if (!result.success) {
        ctx_.recordEvent("PATROL_PLANNING_FAILED", result.reason);
        return result;
    }

    ctx_.recon_patrol_plans = result.data;
    ctx_.recordEvent(
        "PATROL_PLANNED",
        "Recon patrol waypoint generation succeeded: "
            + std::to_string(ctx_.recon_patrol_plans.size()) + " plans");
    return result;
}

AlgorithmResult<EnvironmentSnapshot> MissionBrain::initializeEnvironment() {
    if (environmentRuntime_ == nullptr) {
        ctx_.recordEvent("ENVIRONMENT_SKIPPED", "No environment runtime configured");
        return AlgorithmResult<EnvironmentSnapshot>::ok(EnvironmentSnapshot{});
    }

    auto result = environmentRuntime_->initialize(ctx_);
    if (!result.success) {
        return result;
    }

    syncEnvironmentSnapshot(result.data);
    ctx_.recordEvent(
        "ENVIRONMENT_INITIALIZED",
        "Environment initialized: " + ctx_.environment_name
            + ", platforms=" + std::to_string(ctx_.agents.size())
            + ", targets=" + std::to_string(ctx_.world.targets.size()));
    return result;
}

void MissionBrain::syncEnvironmentSnapshot(const EnvironmentSnapshot& snapshot) {
    ctx_.environment_initialized = snapshot.initialized;
    ctx_.environment_name = snapshot.name;
    ctx_.environment_time_s = snapshot.tactical_time_s;

    if (!snapshot.agents.empty()) {
        ctx_.agents = snapshot.agents;
    }
    if (!snapshot.targets.empty()) {
        ctx_.world.targets = snapshot.targets;
    }
    if (!snapshot.aois.empty()) {
        ctx_.world.aois = snapshot.aois;
        ctx_.world.aoi = snapshot.aois.front();
        ctx_.world.commander_aoi.clear();
        for (const auto& aoi : snapshot.aois) {
            ctx_.world.commander_aoi.push_back(aoi.id);
        }
    }
    ctx_.world.staging_position = snapshot.staging_position;
    if (!snapshot.weather.empty()) {
        ctx_.world.weather = snapshot.weather;
    }
    if (!snapshot.terrain.empty()) {
        ctx_.world.terrain = snapshot.terrain;
    }
}

void MissionBrain::refreshReconPatrolPlans(const std::string& source) {
    if (patrolPlanner_ == nullptr || ctx_.recon_allocation.empty()) {
        return;
    }
    auto result = buildReconPatrolPlans();
    if (!result.success) {
        ctx_.recordEvent("PATROL_REFRESH_SKIPPED", source + ": " + result.reason);
    }
}

AlgorithmResult<BrainRuntimeResult> MissionBrain::buildActionPlan(
    const std::vector<std::string>& targetIds,
    const std::string& source) {
    auto allocation = taskAllocator_.allocateAction(ctx_, targetIds, false);
    if (!allocation.success) {
        ctx_.recordEvent("ACTION_REALLOCATION_FAILED", allocation.reason);
        return AlgorithmResult<BrainRuntimeResult>::fail(allocation.reason);
    }
    ctx_.action_allocation = allocation.data;

    auto selected = positionSelector_.select(ctx_, ctx_.action_allocation);
    if (!selected.success) {
        ctx_.recordEvent("ACTION_POSITION_FAILED", selected.reason);
        return AlgorithmResult<BrainRuntimeResult>::fail(
            "Position selection failed during " + source + ": " + selected.reason);
    }
    ctx_.selected_positions = selected.data;

    auto route = routePlanner_.planActionRoute(
        ctx_,
        ctx_.action_allocation,
        ctx_.selected_positions);
    if (!route.success) {
        ctx_.recordEvent("ACTION_ROUTE_FAILED", route.reason);
        return AlgorithmResult<BrainRuntimeResult>::fail(
            "Action route planning failed during " + source + ": " + route.reason);
    }
    ctx_.action_formation_plan = route.data;

    for (const auto& task : ctx_.action_allocation) {
        if (!task.target.empty()) {
            ctx_.engaged_targets.insert(task.target);
            removeValue(ctx_.pending_strike_targets, task.target);
            ctx_.active_action_plans[task.target] = ctx_.action_formation_plan;
        }
    }

    ctx_.recordEvent("ACTION_PLAN_UPDATED", source + ": strike plan ready");
    return AlgorithmResult<BrainRuntimeResult>::ok(snapshot());
}

TargetInfo& MissionBrain::ensureTarget(const std::string& targetId) {
    for (auto& target : ctx_.world.targets) {
        if (target.tid == targetId) {
            return target;
        }
    }
    TargetInfo target;
    target.tid = targetId;
    target.type = "AV";
    target.pos = {0.0, 0.0};
    ctx_.world.targets.push_back(target);
    return ctx_.world.targets.back();
}

AgentSpec* MissionBrain::findAgent(const std::string& platformId) {
    for (auto& agent : ctx_.agents) {
        if (agent.pid == platformId) {
            return &agent;
        }
    }
    return nullptr;
}

void MissionBrain::discardTargetFromQueues(const std::string& targetId) {
    ctx_.engaged_targets.erase(targetId);
    removeValue(ctx_.pending_strike_targets, targetId);
    ctx_.active_action_plans.erase(targetId);
}

BrainRuntimeResult MissionBrain::snapshot() const {
    BrainRuntimeResult result;
    result.recon_allocation = ctx_.recon_allocation;
    result.action_allocation = ctx_.action_allocation;
    result.selected_positions = ctx_.selected_positions;
    result.recon_formation_plan = ctx_.recon_formation_plan;
    result.action_formation_plan = ctx_.action_formation_plan;
    result.recon_patrol_plans = ctx_.recon_patrol_plans;
    for (const auto& item : ctx_.target_tracks) {
        result.target_tracks.push_back(item.second);
    }
    return result;
}

}  // namespace brain_cpp
