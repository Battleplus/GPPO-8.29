#pragma once

#include <string>
#include <vector>

namespace sar_search_planner {

struct SearchTask {
    std::string platform_id;
    double center_x_km = 0.0;
    double center_y_km = 0.0;
    double width_km = 25.0;
    double height_km = 25.0;
    std::string pattern = "racetrack";
    double altitude_agl_m = 5000.0;
};

struct Waypoint {
    std::string platform_id;
    int point_index = 0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double terrain_z = 0.0;
    double yaw_deg = 0.0;
    double total_km = 0.0;
};

struct PlannerClientOptions {
    // Run from the project root by default. Override these if your main
    // program starts from another working directory.
    std::string python_executable = "python";
    std::string bridge_script = "search_planner/cpp_plan_bridge.py";
};

// Calls the Python SAR planner bridge and returns all generated waypoints.
// If a platform_id appears in multiple SearchTask objects, the planner builds
// a multi-region cycle path for that platform.
std::vector<Waypoint> PlanSarSearchPath(
    const std::vector<SearchTask>& tasks,
    const PlannerClientOptions& options = PlannerClientOptions()
);

}  // namespace sar_search_planner
