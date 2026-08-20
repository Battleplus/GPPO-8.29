#include "brain_cpp/external_adapters.hpp"

#include "brain_cpp/mppi_planner_client.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace brain_cpp {
namespace {

Point2 cellCenterKm(const Aoi& aoi, const std::string& cell) {
    const double x0 = static_cast<double>(aoi.col - 1) * 50.0;
    const double y0 = static_cast<double>(aoi.row - 1) * 50.0;
    const std::map<std::string, Point2> centers = {
        {"c0", {x0 + 25.0, y0 + 25.0}},
        {"c1", {x0 + 12.5, y0 + 12.5}},
        {"c2", {x0 + 37.5, y0 + 12.5}},
        {"c3", {x0 + 12.5, y0 + 37.5}},
        {"c4", {x0 + 37.5, y0 + 37.5}},
    };
    const auto found = centers.find(cell);
    return found == centers.end() ? centers.at("c0") : found->second;
}

const AgentSpec* findAgent(const MissionContext& context, const std::string& platform) {
    for (const auto& agent : context.agents) {
        if (agent.pid == platform) return &agent;
    }
    return nullptr;
}

int positionRank(const Position& position) {
    const auto found = position.metadata.find("rank");
    if (found == position.metadata.end()) return 999;
    try {
        return std::stoi(found->second);
    } catch (const std::exception&) {
        return 999;
    }
}

std::string bindingKey(const std::string& platform, const std::string& target) {
    return platform + "\n" + target;
}

int munitionPriority(const std::string& name) {
    if (name == "HF") return 0;
    if (name == "RKT") return 1;
    if (name == "GUN") return 2;
    return 99;
}

std::vector<StrikeTask> oneToOneStrikeTasks(const std::vector<StrikeTask>& allocation) {
    std::map<std::string, std::vector<const StrikeTask*>> rowsByPair;
    std::vector<std::string> platforms;
    std::map<std::string, std::vector<std::string>> adjacency;
    for (const auto& task : allocation) {
        if (task.platform.empty() || task.target.empty()) continue;
        rowsByPair[bindingKey(task.platform, task.target)].push_back(&task);
        if (adjacency.find(task.platform) == adjacency.end()) platforms.push_back(task.platform);
        auto& targets = adjacency[task.platform];
        if (std::find(targets.begin(), targets.end(), task.target) == targets.end()) {
            targets.push_back(task.target);
        }
    }

    std::map<std::string, std::string> targetToPlatform;
    std::function<bool(const std::string&, std::set<std::string>&)> augment;
    augment = [&](const std::string& platform, std::set<std::string>& seen) {
        for (const auto& target : adjacency[platform]) {
            if (!seen.insert(target).second) continue;
            const auto previous = targetToPlatform.find(target);
            if (previous == targetToPlatform.end() || augment(previous->second, seen)) {
                targetToPlatform[target] = platform;
                return true;
            }
        }
        return false;
    };
    for (const auto& platform : platforms) {
        std::set<std::string> seen;
        augment(platform, seen);
    }

    std::map<std::string, std::string> platformToTarget;
    for (const auto& item : targetToPlatform) platformToTarget[item.second] = item.first;
    std::vector<StrikeTask> result;
    for (const auto& platform : platforms) {
        const auto assigned = platformToTarget.find(platform);
        if (assigned == platformToTarget.end()) continue;
        const auto& rows = rowsByPair.at(bindingKey(platform, assigned->second));
        StrikeTask task = *rows.front();
        task.assigned_munitions.clear();
        for (const auto* row : rows) {
            if (row->assigned_munitions.empty()) {
                task.assigned_munitions[row->munition] += row->qty;
            } else {
                for (const auto& load : row->assigned_munitions) {
                    task.assigned_munitions[load.first] += load.second;
                }
            }
        }
        if (!task.assigned_munitions.empty()) {
            const auto primary = std::min_element(
                task.assigned_munitions.begin(), task.assigned_munitions.end(),
                [](const auto& left, const auto& right) {
                    const int leftPriority = munitionPriority(left.first);
                    const int rightPriority = munitionPriority(right.first);
                    return leftPriority == rightPriority
                        ? left.first < right.first : leftPriority < rightPriority;
                });
            task.munition = primary->first;
            task.qty = primary->second;
        }
        result.push_back(std::move(task));
    }
    return result;
}

}  // namespace

class MppiRoutePlanner::Impl {
public:
    explicit Impl(MppiRoutePlannerOptions plannerOptions)
        : options(std::move(plannerOptions)), client(options.project_root) {
        if (options.map_size_units <= 0.0 || options.meters_per_unit <= 0.0) {
            throw std::invalid_argument("MPPI map_size_units and meters_per_unit must be positive");
        }
    }

    Point3 missionToScene(const Point3& point) const {
        const double unitsPerKm = 1000.0 / options.meters_per_unit;
        const double mapSizeKm = options.map_size_units / unitsPerKm;
        return {
            (point[0] - mapSizeKm * 0.5) * unitsPerKm,
            (point[1] - mapSizeKm * 0.5) * unitsPerKm,
            point[2] * unitsPerKm,
        };
    }

    Point3 sceneToMission(const mppi_client::Point3& point) const {
        const double unitsPerKm = 1000.0 / options.meters_per_unit;
        const double mapSizeKm = options.map_size_units / unitsPerKm;
        return {
            point[0] / unitsPerKm + mapSizeKm * 0.5,
            point[1] / unitsPerKm + mapSizeKm * 0.5,
            point[2] / unitsPerKm,
        };
    }

    mppi_client::PlanRequest request(
        int teamCount, const Point3& start, const Point3& goal,
        const std::string& formation) const {
        mppi_client::PlanRequest result;
        result.team_count = teamCount;
        result.start = missionToScene(start);
        result.goal = missionToScene(goal);
        result.formation = formation;
        result.spacing = options.formation_spacing_units;
        result.map_size_units = options.map_size_units;
        result.meters_per_unit = options.meters_per_unit;
        result.terrain_vertical_exaggeration = options.terrain_vertical_exaggeration;
        result.verbose = options.verbose;
        result.planner_config.num_samples = options.num_samples;
        result.planner_config.num_iterations = options.num_iterations;
        result.planner_config.horizon = options.horizon;
        return result;
    }

    std::vector<Point3> pathToMission(const std::vector<mppi_client::Point3>& path) const {
        std::vector<Point3> result;
        result.reserve(path.size());
        for (const auto& point : path) result.push_back(sceneToMission(point));
        return result;
    }

    MppiRoutePlannerOptions options;
    mppi_client::PlannerClient client;
};

MppiRoutePlanner::MppiRoutePlanner(MppiRoutePlannerOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

MppiRoutePlanner::~MppiRoutePlanner() = default;

AlgorithmResult<FormationPlan>
MppiRoutePlanner::planReconRoute(
    const MissionContext& context,
    const std::vector<ReconTask>& allocation) {
    if (allocation.empty()) {
        return AlgorithmResult<FormationPlan>::fail("No recon allocation for MPPI");
    }

    std::map<std::string, std::vector<const ReconTask*>> groups;
    for (const auto& task : allocation) {
        if (!task.platform.empty() && task.role != "track") groups[task.cell].push_back(&task);
    }
    if (groups.empty()) {
        return AlgorithmResult<FormationPlan>::fail("No eligible recon platforms for MPPI");
    }

    FormationPlan plan;
    plan.formation_type = groups.size() == 1 ? "v_shape" : "per_cell_recon";
    plan.success = true;
    try {
        for (const auto& group : groups) {
            const auto& tasks = group.second;
            Point3 start{0.0, 0.0, impl_->options.recon_altitude_km};
            for (const auto* task : tasks) {
                const AgentSpec* agent = findAgent(context, task->platform);
                if (agent == nullptr) {
                    return AlgorithmResult<FormationPlan>::fail(
                        "Unknown recon platform for MPPI: " + task->platform);
                }
                start[0] += agent->position[0];
                start[1] += agent->position[1];
            }
            start[0] /= static_cast<double>(tasks.size());
            start[1] /= static_cast<double>(tasks.size());
            const Point2 center = cellCenterKm(context.world.aoi, group.first);
            const Point3 goal{center[0], center[1], impl_->options.recon_altitude_km};
            const std::string formation = tasks.size() == 1 ? "column" : "v_shape";
            const auto result = impl_->client.plan(
                impl_->request(static_cast<int>(tasks.size()), start, goal, formation));
            if (!result.success || result.team_paths.size() != tasks.size()) {
                return AlgorithmResult<FormationPlan>::fail(
                    "MPPI recon planning failed for cell=" + group.first);
            }

            if (plan.center_path.empty()) plan.center_path = impl_->pathToMission(result.center_path);
            for (std::size_t i = 0; i < tasks.size(); ++i) {
                if (result.team_paths[i].empty()) {
                    return AlgorithmResult<FormationPlan>::fail(
                        "MPPI returned an empty recon path for " + tasks[i]->platform);
                }
                Route route;
                route.platform = tasks[i]->platform;
                route.role = tasks[i]->role;
                route.target_id = group.first;
                route.position_id = "cell_" + group.first;
                route.waypoints = impl_->pathToMission(result.team_paths[i]);
                route.metadata = {
                    {"planner", "MPPI"},
                    {"adapter", "mppi_cpp_interface"},
                    {"coordinate_frame", "mission_km"},
                    {"cell", group.first},
                    {"sensor", tasks[i]->sensor},
                    {"aoi", tasks[i]->aoi},
                };
                plan.routes.push_back(route);
                plan.team_paths.push_back(route.waypoints);
                plan.formation_roles.push_back(
                    i < result.formation_roles.size() ? result.formation_roles[i] : tasks[i]->role);
                plan.assignment_map.push_back({
                    {"platform_id", tasks[i]->platform},
                    {"cell", group.first},
                    {"route_index", std::to_string(plan.routes.size() - 1)},
                });
            }
        }
    } catch (const std::exception& exc) {
        return AlgorithmResult<FormationPlan>::fail(std::string("MPPI C++ recon call failed: ") + exc.what());
    }
    plan.team_count = static_cast<int>(plan.routes.size());
    plan.planner_stats = {
        {"algorithm", "MPPI"},
        {"adapter", "mppi_cpp_interface"},
        {"coordinate_frame", "mission_km"},
        {"group_count", std::to_string(groups.size())},
        {"route_count", std::to_string(plan.routes.size())},
        {"num_samples", std::to_string(impl_->options.num_samples)},
        {"num_iterations", std::to_string(impl_->options.num_iterations)},
        {"horizon", std::to_string(impl_->options.horizon)},
    };
    return AlgorithmResult<FormationPlan>::ok(std::move(plan));
}

AlgorithmResult<FormationPlan>
MppiRoutePlanner::planActionRoute(
    const MissionContext& context,
    const std::vector<StrikeTask>& allocation,
    const std::vector<Position>& selectedPositions) {
    const auto matchedTasks = oneToOneStrikeTasks(allocation);
    if (matchedTasks.empty()) {
        return AlgorithmResult<FormationPlan>::fail("No strike allocation for MPPI");
    }
    std::map<std::string, const Position*> positions;
    for (const auto& position : selectedPositions) {
        const auto platform = position.metadata.find("platform_id");
        const auto target = position.metadata.find("target_id");
        if (platform == position.metadata.end() || target == position.metadata.end()) continue;
        const std::string key = bindingKey(platform->second, target->second);
        const auto current = positions.find(key);
        if (current == positions.end() || positionRank(position) < positionRank(*current->second)) {
            positions[key] = &position;
        }
    }

    FormationPlan plan;
    plan.formation_type = "individual_strike";
    plan.success = true;
    try {
        for (const auto& task : matchedTasks) {
            const AgentSpec* agent = findAgent(context, task.platform);
            if (agent == nullptr) {
                return AlgorithmResult<FormationPlan>::fail(
                    "Unknown strike platform for MPPI: " + task.platform);
            }
            const auto found = positions.find(bindingKey(task.platform, task.target));
            if (found == positions.end()) {
                return AlgorithmResult<FormationPlan>::fail(
                    "Missing selected position for platform=" + task.platform
                    + ", target=" + task.target);
            }
            const Position& position = *found->second;
            const Point3 start{agent->position[0], agent->position[1], agent->altitude_km};
            const Point3 goal{position.x, position.y, position.z};
            const Point3 approach{
                position.x,
                position.y,
                std::max(position.z, impl_->options.action_cruise_altitude_km),
            };
            const auto result = impl_->client.plan(impl_->request(1, start, approach, "column"));
            if (!result.success || result.team_paths.size() != 1 || result.team_paths[0].empty()) {
                return AlgorithmResult<FormationPlan>::fail(
                    "MPPI action planning failed for platform=" + task.platform);
            }

            Route route;
            route.platform = task.platform;
            route.target_id = task.target;
            route.position_id = position.pos_id;
            route.role = task.role;
            route.waypoints = impl_->pathToMission(result.team_paths[0]);
            if (route.waypoints.empty()
                || std::abs(route.waypoints.back()[0] - goal[0]) > 1e-9
                || std::abs(route.waypoints.back()[1] - goal[1]) > 1e-9
                || std::abs(route.waypoints.back()[2] - goal[2]) > 1e-9) {
                route.waypoints.push_back(goal);
            }
            route.metadata = {
                {"planner", "MPPI"},
                {"adapter", "mppi_cpp_interface"},
                {"coordinate_frame", "mission_km"},
                {"munition", task.munition},
                {"approach_altitude_km", std::to_string(approach[2])},
            };
            plan.routes.push_back(route);
            plan.team_paths.push_back(route.waypoints);
            plan.formation_roles.push_back(task.role);
            plan.assignment_map.push_back({
                {"platform_id", task.platform},
                {"target_id", task.target},
                {"position_id", position.pos_id},
                {"munition", task.munition},
                {"route_index", std::to_string(plan.routes.size() - 1)},
            });
        }
    } catch (const std::exception& exc) {
        return AlgorithmResult<FormationPlan>::fail(std::string("MPPI C++ action call failed: ") + exc.what());
    }
    plan.team_count = static_cast<int>(plan.routes.size());
    if (!plan.routes.empty()) plan.center_path = plan.routes.front().waypoints;
    plan.planner_stats = {
        {"algorithm", "MPPI"},
        {"adapter", "mppi_cpp_interface"},
        {"coordinate_frame", "mission_km"},
        {"mode", "one_aircraft_one_target"},
        {"route_count", std::to_string(plan.routes.size())},
        {"num_samples", std::to_string(impl_->options.num_samples)},
        {"num_iterations", std::to_string(impl_->options.num_iterations)},
        {"horizon", std::to_string(impl_->options.horizon)},
    };
    return AlgorithmResult<FormationPlan>::ok(std::move(plan));
}

}  // namespace brain_cpp
