#include "brain_cpp/external_adapters.hpp"
#include "brain_cpp/scenario_initializer.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::string aoi = "A_3_6";
    std::string cell = "c3";
    std::string sensor = "SAR";
    std::string platform = "Blue_CH4_Recon";
    std::string output = "drl_env/outputs/brain_cpp_ppo_isaac/global_path.csv";
    std::string python = "python";
    std::string bridge = "search_planner/cpp_plan_bridge.py";
};

Options parseArgs(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        auto value = [&]() -> std::string {
            if (index + 1 >= argc) {
                throw std::runtime_error(key + " requires a value");
            }
            return argv[++index];
        };
        if (key == "--aoi") {
            options.aoi = value();
        } else if (key == "--cell") {
            options.cell = value();
        } else if (key == "--sensor") {
            options.sensor = value();
        } else if (key == "--platform") {
            options.platform = value();
        } else if (key == "--output") {
            options.output = value();
        } else if (key == "--python") {
            options.python = value();
        } else if (key == "--bridge") {
            options.bridge = value();
        } else if (key == "--help" || key == "-h") {
            std::cout
                << "Usage: brain_cpp_patrol_export [--aoi A_3_6] [--cell c3] "
                << "[--sensor SAR] [--platform ID] [--output path]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + key);
        }
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseArgs(argc, argv);
        brain_cpp::MissionContext context;
        brain_cpp::StartupOptions startup;
        startup.mission_id = "BRAIN_CPP_PATROL_EXPORT";
        startup.aoi = options.aoi;
        brain_cpp::ScenarioInitializer().normalize(context, startup);

        brain_cpp::ReconTask task;
        task.platform = options.platform;
        task.cell = options.cell;
        task.sensor = options.sensor;
        task.role = options.cell == "c0" ? "area_scan" : "subarea_search";
        task.aoi = context.world.aoi.id;

        brain_cpp::SarSearchPatrolOptions plannerOptions;
        plannerOptions.python_executable = options.python;
        plannerOptions.bridge_script = options.bridge;
        brain_cpp::SarSearchPatrolPlanner planner(plannerOptions);
        brain_cpp::FormationPlan transit;
        transit.success = true;

        const auto result = planner.planPatrols(context, {task}, transit);
        if (!result.success || result.data.empty()) {
            throw std::runtime_error(
                result.reason.empty() ? "patrol planner returned no plan" : result.reason);
        }
        const auto& patrol = result.data.front();
        const std::filesystem::path outputPath(options.output);
        if (outputPath.has_parent_path()) {
            std::filesystem::create_directories(outputPath.parent_path());
        }
        std::ofstream output(outputPath);
        if (!output) {
            throw std::runtime_error("failed to open output: " + outputPath.string());
        }
        output << "platform,sensor,pattern,cell,point_index,x,y,z\n";
        for (std::size_t index = 0; index < patrol.waypoints.size(); ++index) {
            const auto& point = patrol.waypoints[index];
            output << patrol.platform << ','
                   << patrol.sensor << ','
                   << patrol.pattern << ','
                   << options.cell << ','
                   << index << ','
                   << point[0] << ','
                   << point[1] << ','
                   << point[2] << '\n';
        }
        std::cout << "BRAIN_CPP_PATROL_OK"
                  << " sensor=" << patrol.sensor
                  << " pattern=" << patrol.pattern
                  << " waypoints=" << patrol.waypoints.size()
                  << " output=" << std::filesystem::absolute(outputPath).string()
                  << '\n';
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "brain_cpp_patrol_export: " << exc.what() << '\n';
        return 1;
    }
}
