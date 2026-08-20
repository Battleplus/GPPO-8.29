#include "brain_cpp/external_adapters.hpp"
#include "brain_cpp/scenario_initializer.hpp"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

}  // namespace

int main() {
    brain_cpp::MissionContext context;
    brain_cpp::StartupOptions startup;
    startup.mission_id = "CPP_PERCH_BRIDGE_TEST";
    startup.aoi = "A_3_4";
    brain_cpp::ScenarioInitializer().normalize(context, startup);
    context.world.targets.front().confirmed = true;

    brain_cpp::StrikeTask task;
    task.platform = "H1";
    task.target = context.world.targets.front().tid;
    task.munition = "HF";
    task.qty = 1;
    task.role = "lead";
    task.aoi = context.world.aoi.id;
    task.assigned_munitions = {{"HF", 1}};

    brain_cpp::PerchPositionSelectorOptions options;
    options.python_executable = "python";
    options.bridge_script = "perch/cpp_bridge.py";
    options.attack_region_mode = "demo";
    options.terrain_mode = "flat";
    options.top_k = 1;
    options.use_pymoo = false;
    brain_cpp::PerchPositionSelector selector(options);

    const auto result = selector.select(context, {task});
    require(result.success, "Perch C++ bridge succeeds: " + result.reason);
    require(result.data.size() == 1, "top_k=1 returns one selected position");
    const auto& position = result.data.front();
    require(position.metadata.at("adapter") == "perch_cpp_bridge", "brain_cpp adapter metadata");
    require(position.metadata.at("source") == "perch:local_fallback", "Perch attack-region source");
    require(position.metadata.at("g_violation") == "0.0", "FREA constraints are feasible");
    require(position.metadata.at("platform_id") == "H1", "platform identity is preserved");
    require(position.metadata.at("target_id") == task.target, "target identity is preserved");
    require(position.metadata.at("coordinate_frame") == "mission_km", "C++ receives mission-km coordinates");
    const auto& target = context.world.targets.front();
    const double distance = std::hypot(
        position.x - target.pos[0], position.y - target.pos[1]);
    require(distance >= 2.0 && distance <= 8.0, "position is inside HF range in kilometre coordinates");
    require(!position.metadata.at("situation").empty(), "situation text is returned to C++");

    brain_cpp::PerchPositionSelectorOptions badOptions = options;
    badOptions.bridge_script = "perch/missing_bridge.py";
    brain_cpp::PerchPositionSelector badSelector(badOptions);
    const auto failed = badSelector.select(context, {task});
    require(!failed.success, "bridge execution failure is returned as AlgorithmResult failure");

    std::cout << "Perch C++ bridge test passed: position=" << position.pos_id
              << " source=" << position.metadata.at("source")
              << " range_km=" << position.metadata.at("target_range_km") << '\n';
    return 0;
}
