#include "brain_cpp/external_adapters.hpp"
#include "brain_cpp/isaac_python_environment.hpp"
#include "brain_cpp/mission_brain.hpp"
#include "brain_cpp/parallel_platform_controller.hpp"
#include "brain_cpp/scenario_initializer.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliArgs {
    brain_cpp::StartupOptions startup;
    std::string environment = "isaac";
    std::string mppi_project_root = "brain_cpp/mppi_compat";
    double mppi_map_size_units = 3000.0;
    double mppi_meters_per_unit = 100.0;
    double mppi_recon_altitude_km = 8.0;
    double mppi_action_cruise_altitude_km = 3.0;
    double mppi_spacing_units = 40.0;
    int mppi_samples = 512;
    int mppi_iterations = 5;
    int mppi_horizon = 50;
    bool mppi_verbose = false;
    std::string milp_dir = "milp";
    std::string milp_solver = "cbc";
    double milp_time_limit = 3.0;
    int milp_verbose = 0;
    std::string isaac_python;
    std::string isaac_helper = "brain_cpp/tools/isaac_snapshot.py";
    std::string patrol_planner = "sar";
    std::string patrol_python = "python3";
    std::string patrol_bridge = "search_planner/cpp_plan_bridge.py";
    std::string ppo_adapter = "bridge";
    std::string ppo_python = "python3";
    std::string ppo_bridge = "ppo_allocation/cpp_bridge.py";
    std::string ppo_model;
    std::string perch_python = "python";
    std::string perch_bridge = "perch/cpp_bridge.py";
    std::string perch_region_mode = "demo";
    std::string perch_preference = "balanced";
    std::string perch_terrain_mode = "scene";
    int perch_top_k = 3;
    bool perch_use_pymoo = false;
    bool perch_region_strict = true;
    bool headless = true;
    bool execute_parallel = false;
    double execution_dt = 1.0;
    double platform_speed = 20.0;
    int execution_max_ticks = 10000;
    bool help = false;
};

void printUsage() {
    std::cout
        << "Usage: brain_cpp_demo [--mission-id ID] [--aoi A_3_4] [--aois A_3_4,A_3_5]\n"
        << "                      [--mission-input path] [--environment isaac|none]\n"
        << "                      [--milp-dir path]\n"
        << "                      [--milp-solver cbc] [--milp-time-limit seconds]\n"
        << "                      [--milp-verbose 0|1]\n"
        << "                      [--mppi-project-root path]\n"
        << "                      [--mppi-samples N] [--mppi-iterations N] [--mppi-horizon N]\n"
        << "                      [--mppi-recon-altitude-km km] [--mppi-action-cruise-altitude-km km]\n"
        << "                      [--mppi-verbose 0|1]\n"
        << "                      [--patrol-planner sar|none] [--patrol-python python3]\n"
        << "                      [--ppo-adapter bridge|none] [--ppo-model path]\n"
        << "                      [--perch-python python]\n"
        << "                      [--perch-region-mode demo|llm|disabled]"
        << " [--perch-preference balanced|survival_first|aggressive]\n"
        << "                      [--perch-terrain-mode scene|flat] [--perch-top-k N]"
        << " [--perch-use-pymoo 0|1] [--perch-region-strict 0|1]\n"
        << "                      [--isaac-python /home/isaac/isaacsim/python.sh]\n\n"
        << "                      [--execute-parallel] [--execution-dt seconds]\n"
        << "                      [--platform-speed units_per_second] [--execution-max-ticks N]\n\n"
        << "Default environment is the repository Isaac air-combat scene.\n";
}

CliArgs parseArgs(int argc, char** argv) {
    CliArgs args;
    args.startup.mission_id = "CPP_BRAIN";
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto requireValue = [&](const std::string& option) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(option + " requires a value");
            }
            return argv[++i];
        };

        if (key == "--help" || key == "-h") {
            args.help = true;
        } else if (key == "--mission-id") {
            args.startup.mission_id = requireValue(key);
        } else if (key == "--aoi") {
            args.startup.aoi = requireValue(key);
        } else if (key == "--aois") {
            args.startup.aois = requireValue(key);
        } else if (key == "--mission-input") {
            args.startup.mission_input = requireValue(key);
        } else if (key == "--environment") {
            args.environment = requireValue(key);
        } else if (key == "--mppi-project-root") {
            args.mppi_project_root = requireValue(key);
        } else if (key == "--mppi-map-size-units") {
            args.mppi_map_size_units = std::stod(requireValue(key));
        } else if (key == "--mppi-meters-per-unit") {
            args.mppi_meters_per_unit = std::stod(requireValue(key));
        } else if (key == "--mppi-recon-altitude-km") {
            args.mppi_recon_altitude_km = std::stod(requireValue(key));
        } else if (key == "--mppi-action-cruise-altitude-km") {
            args.mppi_action_cruise_altitude_km = std::stod(requireValue(key));
        } else if (key == "--mppi-spacing-units") {
            args.mppi_spacing_units = std::stod(requireValue(key));
        } else if (key == "--mppi-samples") {
            args.mppi_samples = std::stoi(requireValue(key));
        } else if (key == "--mppi-iterations") {
            args.mppi_iterations = std::stoi(requireValue(key));
        } else if (key == "--mppi-horizon") {
            args.mppi_horizon = std::stoi(requireValue(key));
        } else if (key == "--mppi-verbose") {
            args.mppi_verbose = std::stoi(requireValue(key)) != 0;
        } else if (key == "--milp-dir") {
            args.milp_dir = requireValue(key);
        } else if (key == "--milp-solver") {
            args.milp_solver = requireValue(key);
        } else if (key == "--milp-time-limit") {
            args.milp_time_limit = std::stod(requireValue(key));
        } else if (key == "--milp-verbose") {
            args.milp_verbose = std::stoi(requireValue(key));
        } else if (key == "--isaac-python") {
            args.isaac_python = requireValue(key);
        } else if (key == "--isaac-helper") {
            args.isaac_helper = requireValue(key);
        } else if (key == "--patrol-planner") {
            args.patrol_planner = requireValue(key);
        } else if (key == "--patrol-python") {
            args.patrol_python = requireValue(key);
        } else if (key == "--patrol-bridge") {
            args.patrol_bridge = requireValue(key);
        } else if (key == "--ppo-adapter") {
            args.ppo_adapter = requireValue(key);
        } else if (key == "--ppo-python") {
            args.ppo_python = requireValue(key);
        } else if (key == "--ppo-bridge") {
            args.ppo_bridge = requireValue(key);
        } else if (key == "--ppo-model") {
            args.ppo_model = requireValue(key);
        } else if (key == "--perch-python") {
            args.perch_python = requireValue(key);
        } else if (key == "--perch-bridge") {
            args.perch_bridge = requireValue(key);
        } else if (key == "--perch-region-mode") {
            args.perch_region_mode = requireValue(key);
        } else if (key == "--perch-preference") {
            args.perch_preference = requireValue(key);
        } else if (key == "--perch-terrain-mode") {
            args.perch_terrain_mode = requireValue(key);
        } else if (key == "--perch-top-k") {
            args.perch_top_k = std::stoi(requireValue(key));
        } else if (key == "--perch-use-pymoo") {
            args.perch_use_pymoo = std::stoi(requireValue(key)) != 0;
        } else if (key == "--perch-region-strict") {
            args.perch_region_strict = std::stoi(requireValue(key)) != 0;
        } else if (key == "--headless") {
            args.headless = true;
        } else if (key == "--no-headless") {
            args.headless = false;
        } else if (key == "--execute-parallel") {
            args.execute_parallel = true;
        } else if (key == "--execution-dt") {
            args.execution_dt = std::stod(requireValue(key));
        } else if (key == "--platform-speed") {
            args.platform_speed = std::stod(requireValue(key));
        } else if (key == "--execution-max-ticks") {
            args.execution_max_ticks = std::stoi(requireValue(key));
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }
    return args;
}

bool executeReconRoutesInParallel(
    const brain_cpp::MissionContext& context,
    const CliArgs& args) {
    brain_cpp::ParallelPlatformController controller(args.platform_speed);
    auto loaded = controller.load(context.agents, context.recon_formation_plan.routes);
    if (!loaded.success) {
        std::cerr << "Parallel recon execution failed to load: " << loaded.reason << "\n";
        return false;
    }

    std::cout << "\nParallel recon execution (brain_cpp control layer)\n";
    std::cout << "  physical_platforms: " << loaded.data.platforms.size() << "\n";
    std::cout << "  shared_dt_s: " << args.execution_dt << "\n";
    std::cout << "  speed_units_per_s: " << args.platform_speed << "\n";
    for (const auto& platform : loaded.data.platforms) {
        std::cout << "  START platform=" << platform.platform
                  << " pos=[" << platform.position[0] << "," << platform.position[1]
                  << "," << platform.position[2] << "]"
                  << " waypoints=" << platform.waypoint_count << "\n";
    }

    brain_cpp::ParallelExecutionSnapshot finalSnapshot = loaded.data;
    for (int count = 0; count < args.execution_max_ticks && !controller.allCompleted(); ++count) {
        auto stepped = controller.step(args.execution_dt);
        if (!stepped.success) {
            std::cerr << "Parallel recon execution step failed: " << stepped.reason << "\n";
            return false;
        }
        finalSnapshot = stepped.data;
    }
    for (const auto& platform : finalSnapshot.platforms) {
        std::cout << "  END platform=" << platform.platform
                  << " state=" << brain_cpp::toString(platform.state)
                  << " pos=[" << platform.position[0] << "," << platform.position[1]
                  << "," << platform.position[2] << "]"
                  << " waypoint=" << platform.waypoint_index << "/" << platform.waypoint_count
                  << " distance=" << platform.distance_travelled << "\n";
    }
    std::cout << "  shared_ticks: " << finalSnapshot.tick
              << " elapsed_s: " << finalSnapshot.elapsed_s
              << " all_completed: " << (finalSnapshot.all_completed ? "true" : "false") << "\n";
    return finalSnapshot.all_completed;
}

std::vector<std::string> allTargetIds(const brain_cpp::MissionContext& context) {
    std::vector<std::string> ids;
    for (const auto& target : context.world.targets) {
        ids.push_back(target.tid);
    }
    return ids;
}

void printSummary(const brain_cpp::MissionBrain& brain) {
    const auto& ctx = brain.context();
    std::cout << "\nMission summary\n";
    std::cout << "  mission_id: " << ctx.mission_id << "\n";
    std::cout << "  state: " << brain_cpp::toString(ctx.state) << "\n";
    std::cout << "  environment: "
              << (ctx.environment_initialized ? ctx.environment_name : "not initialized")
              << "\n";
    std::cout << "  platforms: " << ctx.agents.size() << "\n";
    std::cout << "  targets: " << ctx.world.targets.size() << "\n";
    std::cout << "  task_areas:";
    for (const auto& item : ctx.world.commander_aoi) {
        std::cout << " " << item;
    }
    std::cout << "\n";
    std::cout << "  recon_allocation: " << ctx.recon_allocation.size() << " tasks\n";
    std::cout << "  action_allocation: " << ctx.action_allocation.size() << " tasks\n";
    std::cout << "  selected_positions: " << ctx.selected_positions.size() << "\n";
    for (const auto& position : ctx.selected_positions) {
        const auto source = position.metadata.find("source");
        const auto adapter = position.metadata.find("adapter");
        std::cout << "    " << position.pos_id
                  << " xyz=[" << position.x << ',' << position.y << ',' << position.z << ']'
                  << " source=" << (source == position.metadata.end() ? "unknown" : source->second)
                  << " adapter=" << (adapter == position.metadata.end() ? "unknown" : adapter->second)
                  << "\n";
    }
    std::cout << "  recon_routes: " << ctx.recon_formation_plan.routes.size() << "\n";
    for (const auto& route : ctx.recon_formation_plan.routes) {
        const auto planner = route.metadata.find("planner");
        const auto adapter = route.metadata.find("adapter");
        std::cout << "    " << route.platform << " -> " << route.target_id
                  << " waypoints=" << route.waypoints.size()
                  << " planner=" << (planner == route.metadata.end() ? "unknown" : planner->second)
                  << " adapter=" << (adapter == route.metadata.end() ? "unknown" : adapter->second)
                  << "\n";
    }
    std::cout << "  recon_patrol_plans: " << ctx.recon_patrol_plans.size() << "\n";
    std::cout << "  action_routes: " << ctx.action_formation_plan.routes.size() << "\n";
    for (const auto& route : ctx.action_formation_plan.routes) {
        const auto planner = route.metadata.find("planner");
        const auto adapter = route.metadata.find("adapter");
        std::cout << "    " << route.platform << " -> " << route.target_id
                  << " waypoints=" << route.waypoints.size()
                  << " planner=" << (planner == route.metadata.end() ? "unknown" : planner->second)
                  << " adapter=" << (adapter == route.metadata.end() ? "unknown" : adapter->second)
                  << "\n";
    }
    std::cout << "  history_entries: " << ctx.history.size() << "\n";
    if (!ctx.last_error.empty()) {
        std::cout << "  last_error: " << ctx.last_error << "\n";
    }
    if (!ctx.last_failed_state.empty()) {
        std::cout << "  last_failed_state: " << ctx.last_failed_state << "\n";
    }
    if (!ctx.world.mission_input_path.empty()) {
        std::cout << "  mission_input: " << ctx.world.mission_input_path << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parseArgs(argc, argv);
        if (args.help) {
            printUsage();
            return 0;
        }

        brain_cpp::MissionContext context;
        brain_cpp::ScenarioInitializer initializer;
        initializer.normalize(context, args.startup);

        brain_cpp::MilpTaskAllocatorOptions milpOptions;
        milpOptions.milp_dir = args.milp_dir;
        milpOptions.solver = args.milp_solver;
        milpOptions.time_limit_s = args.milp_time_limit;
        milpOptions.verbose = args.milp_verbose;
        brain_cpp::MilpTaskAllocator allocator(milpOptions);
        brain_cpp::MppiRoutePlannerOptions mppiOptions;
        mppiOptions.project_root = args.mppi_project_root;
        mppiOptions.map_size_units = args.mppi_map_size_units;
        mppiOptions.meters_per_unit = args.mppi_meters_per_unit;
        mppiOptions.recon_altitude_km = args.mppi_recon_altitude_km;
        mppiOptions.action_cruise_altitude_km = args.mppi_action_cruise_altitude_km;
        mppiOptions.formation_spacing_units = args.mppi_spacing_units;
        mppiOptions.num_samples = args.mppi_samples;
        mppiOptions.num_iterations = args.mppi_iterations;
        mppiOptions.horizon = args.mppi_horizon;
        mppiOptions.verbose = args.mppi_verbose;
        brain_cpp::MppiRoutePlanner routePlanner(mppiOptions);
        brain_cpp::PerchPositionSelectorOptions perchOptions;
        perchOptions.python_executable = args.perch_python;
        perchOptions.bridge_script = args.perch_bridge;
        perchOptions.attack_region_mode = args.perch_region_mode;
        perchOptions.preference = args.perch_preference;
        perchOptions.terrain_mode = args.perch_terrain_mode;
        perchOptions.top_k = args.perch_top_k;
        perchOptions.use_pymoo = args.perch_use_pymoo;
        perchOptions.attack_region_strict = args.perch_region_strict;
        brain_cpp::PerchPositionSelector perchPositionSelector(perchOptions);
        brain_cpp::PpoBridgeOptions ppoBridgeOptions;
        ppoBridgeOptions.python_executable = args.ppo_python;
        ppoBridgeOptions.bridge_script = args.ppo_bridge;
        if (!args.ppo_model.empty()) {
            ppoBridgeOptions.model_path = args.ppo_model;
        }
        brain_cpp::PpoBridgeReallocator ppoBridgeReallocator(ppoBridgeOptions);
        brain_cpp::SarSearchPatrolOptions sarOptions;
        sarOptions.python_executable = args.patrol_python;
        sarOptions.bridge_script = args.patrol_bridge;
        brain_cpp::SarSearchPatrolPlanner sarPatrolPlanner(sarOptions);
        brain_cpp::IsaacPythonEnvironment isaacEnvironment(
            args.isaac_python,
            args.isaac_helper,
            args.headless);
        brain_cpp::IEnvironmentRuntime* environment = nullptr;
        if (args.environment == "isaac") {
            environment = &isaacEnvironment;
        } else if (args.environment == "none") {
            environment = nullptr;
        } else {
            throw std::runtime_error("--environment must be isaac or none");
        }
        brain_cpp::IPpoReallocator* ppo = nullptr;
        if (args.ppo_adapter == "bridge") {
            ppo = &ppoBridgeReallocator;
        } else if (args.ppo_adapter == "none") {
            ppo = nullptr;
        } else {
            throw std::runtime_error("--ppo-adapter must be bridge or none");
        }

        brain_cpp::IPatrolPlanner* patrol = nullptr;
        if (args.patrol_planner == "sar") {
            patrol = &sarPatrolPlanner;
        } else if (args.patrol_planner == "none") {
            patrol = nullptr;
        } else {
            throw std::runtime_error("--patrol-planner must be sar or none");
        }
        brain_cpp::MissionBrain brain(
            context,
            allocator,
            routePlanner,
            perchPositionSelector,
            environment,
            ppo,
            patrol);

        auto state = brain.start();
        std::cout << "After start: " << brain_cpp::toString(state) << "\n";

        if (state == brain_cpp::MissionState::RECON_PLAN_READY) {
            state = brain.dispatch(brain_cpp::MissionEvent::reconPlanDispatched());
            if (args.execute_parallel && !executeReconRoutesInParallel(context, args)) {
                std::cerr << "Recon execution did not complete; RECON_FINISHED will not be sent\n";
                printSummary(brain);
                return 3;
            }
            state = brain.dispatch(brain_cpp::MissionEvent::reconFinished());
            state = brain.dispatch(brain_cpp::MissionEvent::reconResultReceived(allTargetIds(context)));
            std::cout << "After recon result: " << brain_cpp::toString(state) << "\n";
        }

        if (state == brain_cpp::MissionState::ACTION_PLAN_READY) {
            state = brain.dispatch(brain_cpp::MissionEvent::actionPlanDispatched());
            state = brain.dispatch(brain_cpp::MissionEvent::actionFinished());
            std::cout << "After action execution: " << brain_cpp::toString(state) << "\n";
        }

        printSummary(brain);
        return brain_cpp::isTerminal(state) ? 0 : 2;
    } catch (const std::exception& exc) {
        std::cerr << "brain_cpp_demo: " << exc.what() << "\n";
        return 1;
    }
}
