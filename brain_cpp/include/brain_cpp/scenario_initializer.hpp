#pragma once

#include "brain_cpp/domain.hpp"
#include "brain_cpp/mission_context.hpp"

#include <string>
#include <vector>

namespace brain_cpp {

struct StartupOptions {
    std::string mission_id = "MISSION_001";
    std::string aoi;
    std::string aois;
    std::string mission_input;
};

class ScenarioInitializer {
public:
    void normalize(MissionContext& context, const StartupOptions& options = {}) const;

    static Aoi parseAoi(const std::string& raw);
    static std::vector<Aoi> parseAoiList(const std::string& raw);
    static std::vector<AgentSpec> buildDefaultAgents();
    static WorldState buildDefaultWorld();
};

}  // namespace brain_cpp
