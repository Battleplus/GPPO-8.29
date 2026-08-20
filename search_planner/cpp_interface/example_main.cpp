#include "SarSearchPlannerClient.hpp"

#include <iostream>
#include <vector>

int main() {
    using namespace sar_search_planner;

    std::vector<SearchTask> tasks = {
        {"Blue_CH4_Recon", 37.5, 42.5, 25.0, 25.0, "racetrack", 5000.0},
        {"Blue_CH4_Recon_2", 62.5, 42.5, 25.0, 25.0, "racetrack", 5000.0},
        {"Blue_CH4_StrikeRecon", 37.5, 17.5, 25.0, 25.0, "figure_eight", 5000.0},
        {"Blue_CH4_StrikeRecon", 62.5, 17.5, 25.0, 25.0, "figure_eight", 5000.0},
    };

    try {
        std::vector<Waypoint> waypoints = PlanSarSearchPath(tasks);
        std::cout << "waypoints: " << waypoints.size() << "\n";
        for (std::size_t i = 0; i < waypoints.size() && i < 5; ++i) {
            const Waypoint& wp = waypoints[i];
            std::cout << wp.platform_id << " #" << wp.point_index
                      << " x=" << wp.x
                      << " y=" << wp.y
                      << " z=" << wp.z
                      << " yaw=" << wp.yaw_deg << "\n";
        }
    } catch (const std::exception& exc) {
        std::cerr << "planner failed: " << exc.what() << "\n";
        return 1;
    }

    return 0;
}
