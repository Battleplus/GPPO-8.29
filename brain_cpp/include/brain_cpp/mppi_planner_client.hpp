#pragma once

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace brain_cpp::mppi_client {

using Point3 = std::array<double, 3>;

struct PlannerConfig {
    int num_samples = 512;
    int num_iterations = 5;
    int horizon = 50;
};

struct PlanRequest {
    int team_count = 1;
    Point3 start{};
    Point3 goal{};
    std::string formation = "column";
    double spacing = 40.0;
    double map_size_units = 3000.0;
    double meters_per_unit = 100.0;
    double terrain_vertical_exaggeration = 10.0;
    bool verbose = false;
    PlannerConfig planner_config;
    std::map<std::string, int> member_assignments;
};

struct PlanResult {
    int team_count = 0;
    std::string formation_type;
    bool success = false;
    std::vector<Point3> center_path;
    std::vector<std::vector<Point3>> team_paths;
    std::vector<std::string> formation_roles;
    std::map<std::string, std::string> planner_stats;
};

class PlannerClient {
public:
    explicit PlannerClient(const std::string& project_root);
    ~PlannerClient();

    PlannerClient(const PlannerClient&) = delete;
    PlannerClient& operator=(const PlannerClient&) = delete;

    PlanResult plan(const PlanRequest& request) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brain_cpp::mppi_client
