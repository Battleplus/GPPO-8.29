#include "brain_cpp/scenario_initializer.hpp"

#include <algorithm>
#include <cctype>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace brain_cpp {
namespace {

std::string trim(std::string value) {
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), value.end());
    return value;
}

bool looksLikeRowCol(const std::string& raw) {
    static const std::regex pattern("^\\s*\\d+\\s*,\\s*\\d+\\s*$");
    return std::regex_match(raw, pattern);
}

std::vector<std::string> splitComma(const std::string& raw) {
    std::vector<std::string> result;
    std::stringstream stream(raw);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item = trim(item);
        if (!item.empty()) {
            result.push_back(item);
        }
    }
    return result;
}

}  // namespace

void ScenarioInitializer::normalize(MissionContext& context, const StartupOptions& options) const {
    if (!options.mission_id.empty()) {
        context.mission_id = options.mission_id;
    }

    if (context.agents.empty()) {
        context.agents = buildDefaultAgents();
    }

    if (context.world.targets.empty()) {
        auto defaults = buildDefaultWorld();
        context.world.targets = defaults.targets;
        context.world.staging_position = defaults.staging_position;
    }

    if (context.world.weather.empty()) {
        context.world.weather = {
            {"c0", 0.20},
            {"c1", 0.15},
            {"c2", 0.40},
            {"c3", 0.55},
            {"c4", 0.70},
        };
    }

    if (context.world.terrain.empty()) {
        context.world.terrain = {
            {"c0", 0},
            {"c1", 0},
            {"c2", 1},
            {"c3", 0},
            {"c4", 2},
        };
    }

    if (!options.aois.empty()) {
        context.world.aois = parseAoiList(options.aois);
        context.world.aoi = context.world.aois.front();
    } else if (!options.aoi.empty()) {
        context.world.aoi = parseAoi(options.aoi);
        context.world.aois = {context.world.aoi};
    } else if (context.world.aois.empty()) {
        context.world.aois = {context.world.aoi};
    } else {
        context.world.aoi = context.world.aois.front();
    }

    context.world.commander_aoi.clear();
    for (const auto& aoi : context.world.aois) {
        context.world.commander_aoi.push_back(aoi.id);
    }

    context.world.mission_input_path = options.mission_input;
}

Aoi ScenarioInitializer::parseAoi(const std::string& raw) {
    const auto value = trim(raw);
    std::smatch match;
    static const std::regex aoiPattern("^A[_-]?(\\d+)[_-](\\d+)$", std::regex_constants::icase);
    static const std::regex rowColPattern("^(\\d+)\\s*[,;:]\\s*(\\d+)$");
    if (std::regex_match(value, match, aoiPattern) || std::regex_match(value, match, rowColPattern)) {
        const int row = std::stoi(match[1].str());
        const int col = std::stoi(match[2].str());
        Aoi aoi;
        aoi.row = row;
        aoi.col = col;
        aoi.id = "A_" + std::to_string(row) + "_" + std::to_string(col);
        return aoi;
    }
    throw std::runtime_error("AOI must be like A_3_4 or 3,4: " + raw);
}

std::vector<Aoi> ScenarioInitializer::parseAoiList(const std::string& raw) {
    const auto value = trim(raw);
    if (value.empty()) {
        return {};
    }
    std::vector<Aoi> aois;
    if (value.find(',') != std::string::npos && !looksLikeRowCol(value)) {
        for (const auto& item : splitComma(value)) {
            aois.push_back(parseAoi(item));
        }
    } else {
        aois.push_back(parseAoi(value));
    }
    for (std::size_t index = 0; index < aois.size(); ++index) {
        aois[index].index = static_cast<int>(index);
    }
    return aois;
}

std::vector<AgentSpec> ScenarioInitializer::buildDefaultAgents() {
    std::vector<AgentSpec> agents;
    for (int i = 1; i <= 5; ++i) {
        AgentSpec agent;
        agent.pid = "U" + std::to_string(i);
        agent.type = "UAV";
        agent.sensors = {"EO", "SAR", "ESM"};
        agent.munitions = {{"HF", 0}, {"RKT", 0}, {"GUN", 0}};
        agent.altitude_km = 2.0;
        agents.push_back(agent);
    }
    for (int i = 1; i <= 2; ++i) {
        AgentSpec agent;
        agent.pid = "H" + std::to_string(i);
        agent.type = "HELI";
        agent.sensors = {"MMW", "EOIR"};
        agent.munitions = {{"HF", 16}, {"RKT", 76}, {"GUN", 1200}};
        agent.altitude_km = 3.0;
        agents.push_back(agent);
    }
    return agents;
}

WorldState ScenarioInitializer::buildDefaultWorld() {
    WorldState world;
    world.targets = {
        {"g1", "RADAR", {270.0, 260.0}, 1.0, 0.9, false, true},
        {"g2", "CP", {310.0, 180.0}, 0.95, 0.85, false, true},
        {"g3", "AV", {220.0, 310.0}, 0.7, 0.65, false, true},
    };
    world.aois = {world.aoi};
    world.commander_aoi = {world.aoi.id};
    return world;
}

}  // namespace brain_cpp
