#include "ql_path_planner.h"

#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: ql_path_planner_example <project-root>\n";
        return 2;
    }
    try {
        ql::PathPlanner planner(argv[1]);
        std::cout << planner.plan(R"({
            "team_count": 2,
            "start": [-800, -600, 80],
            "goal": [800, 600, 80],
            "formation": "v_shape",
            "spacing": 40,
            "planner_config": {"num_samples": 64, "num_iterations": 2, "horizon": 30}
        })") << '\n';
    } catch (const std::exception& error) {
        std::cerr << "planning failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
