#pragma once

#include <array>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace brain_cpp {

using Point2 = std::array<double, 2>;
using Point3 = std::array<double, 3>;

template <typename T>
struct AlgorithmResult {
    bool success = false;
    T data{};
    std::string reason;

    static AlgorithmResult<T> ok(T value) {
        AlgorithmResult<T> result;
        result.success = true;
        result.data = std::move(value);
        return result;
    }

    static AlgorithmResult<T> fail(std::string why) {
        AlgorithmResult<T> result;
        result.success = false;
        result.reason = std::move(why);
        return result;
    }
};

struct Aoi {
    std::string id = "A_3_4";
    int row = 3;
    int col = 4;
    double priority = 1.0;
    double target_prior = 0.25;
    double target_value = 0.5;
    double target_threat = 0.5;
    int index = 0;
};

struct AgentSpec {
    std::string pid;
    std::string type;
    Point2 position{150.0, -50.0};
    std::vector<std::string> sensors;
    std::map<std::string, int> munitions;
    double altitude_km = 2.0;
    bool lost = false;
};

struct TargetInfo {
    std::string tid;
    std::string type = "AV";
    Point2 pos{0.0, 0.0};
    double value = 0.5;
    double threat = 0.5;
    bool confirmed = false;
    bool alive = true;
};

struct WorldState {
    Aoi aoi;
    std::vector<Aoi> aois;
    std::vector<std::string> commander_aoi;
    Point2 staging_position{150.0, -50.0};
    std::map<std::string, double> weather;
    std::map<std::string, int> terrain;
    std::vector<TargetInfo> targets;
    std::string mission_input_path;
};

struct ReconTask {
    std::string platform;
    std::string cell;
    std::string sensor;
    std::string role = "area_scan";
    std::string aoi;
};

struct StrikeTask {
    std::string platform;
    std::string target;
    std::string munition;
    int qty = 0;
    std::string role = "lead";
    std::string aoi;
    std::map<std::string, int> assigned_munitions;
};

struct Position {
    std::string pos_id;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    std::string kind = "attack";
    std::map<std::string, std::string> metadata;
};

struct Route {
    std::string platform;
    std::vector<Point3> waypoints;
    std::string role;
    std::string target_id;
    std::string position_id;
    std::map<std::string, std::string> metadata;
};

struct PatrolPlan {
    std::string platform;
    std::vector<std::string> cells;
    std::string sensor;
    std::string pattern = "racetrack";
    std::vector<Point3> waypoints;
    std::map<std::string, std::string> metadata;
};

struct FormationPlan {
    std::string formation_type;
    int team_count = 0;
    bool success = false;
    std::vector<Point3> center_path;
    std::vector<std::vector<Point3>> team_paths;
    std::vector<std::string> formation_roles;
    std::vector<Route> routes;
    std::vector<std::map<std::string, std::string>> assignment_map;
    std::map<std::string, std::string> planner_stats;
};

struct HistoryEntry {
    std::string timestamp;
    std::string mission_id;
    std::string state;
    std::string event;
    std::string detail;
};

}  // namespace brain_cpp
