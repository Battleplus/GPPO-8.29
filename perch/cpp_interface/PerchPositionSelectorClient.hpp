#pragma once

#include <map>
#include <string>
#include <vector>

namespace perch_cpp {

struct Agent {
    std::string pid;
    std::string type;
    double x_km = 0.0;
    double y_km = 0.0;
    double altitude_km = 0.3;
    std::vector<std::string> sensors;
    std::map<std::string, int> munitions;
    bool lost = false;
};

struct Target {
    std::string tid;
    std::string type;
    double x_km = 0.0;
    double y_km = 0.0;
    double value = 0.5;
    double threat = 0.5;
    bool confirmed = true;
    bool alive = true;
};

struct StrikeTask {
    std::string platform;
    std::string target;
    std::string munition = "HF";
    int qty = 1;
    std::string role = "lead";
    std::string aoi;
    std::map<std::string, int> assigned_munitions;
};

struct Situation {
    std::string mission_id;
    double staging_x_km = 150.0;
    double staging_y_km = -50.0;
    std::map<std::string, double> weather;
    std::map<std::string, int> terrain;
    std::vector<Agent> agents;
    std::vector<Target> targets;
};

struct SelectedPosition {
    std::string pos_id;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    std::string kind = "attack";
    std::map<std::string, std::string> metadata;
};

struct ClientOptions {
    std::string python_executable = "python";
    std::string bridge_script = "perch/cpp_bridge.py";
    std::string attack_region_mode = "llm";
    std::string preference = "balanced";
    std::string terrain_mode = "scene";
    int top_k = 3;
    bool use_pymoo = false;
    bool attack_region_strict = true;
};

std::vector<SelectedPosition> SelectAttackPositions(
    const Situation& situation,
    const std::vector<StrikeTask>& tasks,
    const ClientOptions& options = ClientOptions()
);

}  // namespace perch_cpp
