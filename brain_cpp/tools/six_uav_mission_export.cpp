#include "brain_cpp/external_adapters.hpp"
#include "brain_cpp/scenario_initializer.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string output_dir = "brain_cpp/outputs/six_uav_milp_mppi_ppo_20260714";
    std::string aoi = "A_3_4";
    int mppi_samples = 64;
    int mppi_iterations = 2;
    int mppi_horizon = 30;
};

Options parseArgs(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        auto value = [&]() -> std::string {
            if (index + 1 >= argc) throw std::runtime_error(key + " requires a value");
            return argv[++index];
        };
        if (key == "--output-dir") options.output_dir = value();
        else if (key == "--aoi") options.aoi = value();
        else if (key == "--mppi-samples") options.mppi_samples = std::stoi(value());
        else if (key == "--mppi-iterations") options.mppi_iterations = std::stoi(value());
        else if (key == "--mppi-horizon") options.mppi_horizon = std::stoi(value());
        else throw std::runtime_error("Unknown option: " + key);
    }
    return options;
}

brain_cpp::AgentSpec aircraft(
    const std::string& id, brain_cpp::Point2 position,
    std::vector<std::string> sensors, double altitudeKm) {
    brain_cpp::AgentSpec result;
    result.pid = id;
    result.type = "UAV";
    result.position = position;
    result.sensors = std::move(sensors);
    result.altitude_km = altitudeKm;
    result.munitions = {{"HF", 0}, {"RKT", 0}, {"GUN", 0}};
    return result;
}

std::vector<brain_cpp::AgentSpec> sixAircraft() {
    return {
        aircraft("Blue_CH4_Recon", {0.0, 92.0}, {"SAR", "EO", "ESM"}, 4.6),
        aircraft("Blue_CH4_StrikeRecon", {0.0, 64.0}, {"SAR", "EO", "ESM"}, 3.8),
        aircraft("Blue_Quad_Recon_1", {8.0, 178.0}, {"EO", "SAR"}, 1.25),
        aircraft("Blue_Quad_Recon_2", {8.0, 126.0}, {"EO", "SAR"}, 1.15),
        aircraft("Blue_Quad_Strike_1", {10.0, 158.0}, {"EO", "SAR"}, 0.95),
        aircraft("Blue_Quad_Strike_2", {10.0, 106.0}, {"EO", "SAR"}, 0.90),
    };
}

brain_cpp::Point3 missionToScene(const brain_cpp::Point3& point) {
    return {(point[0] - 150.0) * 10.0, (point[1] - 150.0) * 10.0, point[2] * 10.0};
}

std::string jsonEscape(const std::string& value) {
    std::string result;
    for (const char ch : value) {
        if (ch == '\\' || ch == '"') result.push_back('\\');
        result.push_back(ch);
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseArgs(argc, argv);
        const std::filesystem::path outputDir = std::filesystem::absolute(options.output_dir);
        std::filesystem::create_directories(outputDir);

        brain_cpp::MissionContext context;
        brain_cpp::StartupOptions startup;
        startup.mission_id = "SIX_UAV_MILP_MPPI_PPO";
        startup.aoi = options.aoi;
        brain_cpp::ScenarioInitializer().normalize(context, startup);
        context.agents = sixAircraft();

        brain_cpp::MilpTaskAllocator allocator;
        const auto allocation = allocator.allocateRecon(context);
        if (!allocation.success) throw std::runtime_error("MILP failed: " + allocation.reason);
        context.recon_allocation = allocation.data;

        std::set<std::string> allocatedPlatforms;
        std::map<std::string, const brain_cpp::ReconTask*> primaryTask;
        for (const auto& task : allocation.data) {
            allocatedPlatforms.insert(task.platform);
            if (primaryTask.find(task.platform) == primaryTask.end()) primaryTask[task.platform] = &task;
            std::cout << "MILP_ASSIGN platform=" << task.platform
                      << " cell=" << task.cell << " sensor=" << task.sensor
                      << " role=" << task.role << '\n';
        }
        std::vector<std::string> reservePlatforms;
        for (const auto& agent : context.agents) {
            if (allocatedPlatforms.count(agent.pid) == 0) reservePlatforms.push_back(agent.pid);
        }

        brain_cpp::MppiRoutePlannerOptions mppiOptions;
        mppiOptions.num_samples = options.mppi_samples;
        mppiOptions.num_iterations = options.mppi_iterations;
        mppiOptions.horizon = options.mppi_horizon;
        brain_cpp::MppiRoutePlanner mppi(mppiOptions);
        const auto transit = mppi.planReconRoute(context, allocation.data);
        if (!transit.success) throw std::runtime_error("MPPI failed: " + transit.reason);

        brain_cpp::SarSearchPatrolOptions searchOptions;
        searchOptions.python_executable = "python";
        brain_cpp::SarSearchPatrolPlanner search(searchOptions);
        const auto patrols = search.planPatrols(context, allocation.data, transit.data);
        if (!patrols.success) throw std::runtime_error("Search planning failed: " + patrols.reason);

        std::map<std::string, const brain_cpp::Route*> transitByPlatform;
        for (const auto& route : transit.data.routes) {
            const auto task = primaryTask.find(route.platform);
            if (task != primaryTask.end() && route.target_id == task->second->cell) {
                transitByPlatform[route.platform] = &route;
            }
        }
        std::map<std::string, const brain_cpp::PatrolPlan*> patrolByPlatform;
        for (const auto& patrol : patrols.data) patrolByPlatform[patrol.platform] = &patrol;

        const auto allocationPath = outputDir / "milp_allocation.csv";
        std::ofstream allocationFile(allocationPath);
        allocationFile << "platform,cell,sensor,role,aoi\n";
        for (const auto& task : allocation.data) {
            allocationFile << task.platform << ',' << task.cell << ',' << task.sensor << ','
                           << task.role << ',' << task.aoi << '\n';
        }

        const auto planPath = outputDir / "mission_plan.csv";
        std::ofstream planFile(planPath);
        planFile << "platform,phase,controller,task_id,sensor,point_index,x,y,z\n";
        for (const auto& agent : context.agents) {
            if (allocatedPlatforms.count(agent.pid) == 0) continue;
            const auto transitFound = transitByPlatform.find(agent.pid);
            const auto patrolFound = patrolByPlatform.find(agent.pid);
            if (transitFound == transitByPlatform.end() || patrolFound == patrolByPlatform.end()) {
                throw std::runtime_error("Missing MPPI or search plan for " + agent.pid);
            }
            const auto* task = primaryTask.at(agent.pid);
            std::size_t pointIndex = 0;
            for (const auto& waypointKm : transitFound->second->waypoints) {
                const auto waypoint = missionToScene(waypointKm);
                planFile << agent.pid << ",transit,mppi_follow," << task->cell << ','
                         << task->sensor << ',' << pointIndex++ << ',' << waypoint[0] << ','
                         << waypoint[1] << ',' << waypoint[2] << '\n';
            }
            pointIndex = 0;
            for (const auto& waypoint : patrolFound->second->waypoints) {
                planFile << agent.pid << ",search,ppo_local," << task->cell << ','
                         << patrolFound->second->sensor << ',' << pointIndex++ << ','
                         << waypoint[0] << ',' << waypoint[1] << ',' << waypoint[2] << '\n';
            }
        }

        const auto summaryPath = outputDir / "planning_summary.json";
        std::ofstream summary(summaryPath);
        summary << "{\n"
                << "  \"success\": true,\n"
                << "  \"input_aircraft_count\": " << context.agents.size() << ",\n"
                << "  \"milp_task_count\": " << allocation.data.size() << ",\n"
                << "  \"milp_allocated_aircraft\": " << allocatedPlatforms.size() << ",\n"
                << "  \"all_six_allocated\": "
                << (allocatedPlatforms.size() == context.agents.size() ? "true" : "false") << ",\n"
                << "  \"reserve_aircraft\": [";
        for (std::size_t index = 0; index < reservePlatforms.size(); ++index) {
            if (index) summary << ',';
            summary << "\"" << jsonEscape(reservePlatforms[index]) << "\"";
        }
        summary << "],\n"
                << "  \"mppi_route_count\": " << transit.data.routes.size() << ",\n"
                << "  \"search_plan_count\": " << patrols.data.size() << ",\n"
                << "  \"transit_controller\": \"mppi_follow\",\n"
                << "  \"search_controller\": \"ppo_local\",\n"
                << "  \"ppo_enabled_during_transit\": false,\n"
                << "  \"allocation_csv\": \"" << jsonEscape(allocationPath.string()) << "\",\n"
                << "  \"mission_plan_csv\": \"" << jsonEscape(planPath.string()) << "\"\n"
                << "}\n";

        std::cout << "SIX_UAV_PLAN_OK input_aircraft=6 active_aircraft="
                  << allocatedPlatforms.size() << " reserve_aircraft=" << reservePlatforms.size()
                  << " milp_tasks=" << allocation.data.size()
                  << " mppi_routes=" << transit.data.routes.size()
                  << " search_plans=" << patrols.data.size()
                  << " output=" << outputDir.string() << '\n';
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "brain_cpp_six_uav_mission_export: " << exc.what() << '\n';
        return 1;
    }
}
