#include "brain_cpp/external_adapters.hpp"
#include "brain_cpp/scenario_initializer.hpp"

#include <cmath>
#include <cstdlib>
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

bool near(double left, double right, double tolerance = 1e-5) {
    return std::abs(left - right) <= tolerance;
}

brain_cpp::MppiRoutePlanner makePlanner() {
    brain_cpp::MppiRoutePlannerOptions options;
    options.project_root = "brain_cpp/mppi_compat";
    options.num_samples = 32;
    options.num_iterations = 1;
    options.horizon = 12;
    options.recon_altitude_km = 8.0;
    return brain_cpp::MppiRoutePlanner(options);
}

}  // namespace

int main() {
    brain_cpp::MissionContext context;
    brain_cpp::StartupOptions startup;
    startup.mission_id = "CPP_MPPI_ROUTE_TEST";
    startup.aoi = "A_3_4";
    brain_cpp::ScenarioInitializer().normalize(context, startup);
    context.agents[0].position = {160.0, 110.0};
    context.agents[1].position = {162.0, 110.0};
    context.agents[5].position = {160.0, 110.0};
    context.agents[5].altitude_km = 3.0;

    brain_cpp::ReconTask recon1;
    recon1.platform = context.agents[0].pid;
    recon1.cell = "c0";
    recon1.sensor = "SAR";
    recon1.role = "leader";
    recon1.aoi = context.world.aoi.id;
    brain_cpp::ReconTask recon2 = recon1;
    recon2.platform = context.agents[1].pid;
    recon2.role = "wing";

    auto planner = makePlanner();
    const auto recon = planner.planReconRoute(context, {recon1, recon2});
    require(recon.success, "real MPPI C++ recon call succeeds: " + recon.reason);
    require(recon.data.success, "recon formation is successful");
    require(recon.data.routes.size() == 2, "two recon platforms receive routes");
    require(recon.data.planner_stats.at("algorithm") == "MPPI", "recon reports MPPI algorithm");
    require(recon.data.planner_stats.at("adapter") == "mppi_cpp_interface", "recon reports C++ adapter");
    for (const auto& route : recon.data.routes) {
        require(route.waypoints.size() >= 2, "MPPI recon route has waypoints");
        require(route.metadata.at("planner") == "MPPI", "route is not a demo route");
        require(route.metadata.at("adapter") == "mppi_cpp_interface", "route came through C++ API");
        require(route.metadata.at("coordinate_frame") == "mission_km", "route uses brain coordinates");
    }

    brain_cpp::StrikeTask strike;
    strike.platform = context.agents[5].pid;
    strike.target = context.world.targets[0].tid;
    strike.munition = "HF";
    strike.qty = 1;
    strike.role = "lead";
    strike.aoi = context.world.aoi.id;
    strike.assigned_munitions = {{"HF", 1}};

    brain_cpp::Position selected;
    selected.pos_id = "H1_g1_REAL_POS";
    selected.x = 165.0;
    selected.y = 115.0;
    selected.z = 3.0;
    selected.metadata = {
        {"platform_id", strike.platform},
        {"target_id", strike.target},
        {"rank", "0"},
        {"coordinate_frame", "mission_km"},
    };

    brain_cpp::StrikeTask duplicateTarget = strike;
    duplicateTarget.platform = context.agents[6].pid;
    duplicateTarget.role = "support";
    const auto action = planner.planActionRoute(
        context, {strike, duplicateTarget}, {selected});
    require(action.success, "real MPPI C++ action call succeeds: " + action.reason);
    require(action.data.routes.size() == 1, "duplicate target rows become one matched strike route");
    const auto& route = action.data.routes.front();
    require(route.metadata.at("planner") == "MPPI", "action route is not a demo route");
    require(route.metadata.at("adapter") == "mppi_cpp_interface", "action uses C++ interface");
    require(route.target_id == strike.target, "target binding survives MPPI planning");
    require(route.position_id == selected.pos_id, "position binding survives MPPI planning");
    require(route.waypoints.size() >= 2, "MPPI action route has waypoints");
    const auto& endpoint = route.waypoints.back();
    require(near(endpoint[0], selected.x) && near(endpoint[1], selected.y)
            && near(endpoint[2], selected.z), "scene route converts back to mission-km endpoint");

    std::cout << "MPPI C++ route test passed: recon_routes=" << recon.data.routes.size()
              << " action_waypoints=" << route.waypoints.size()
              << " endpoint=[" << endpoint[0] << ',' << endpoint[1] << ',' << endpoint[2] << "]\n";
    return 0;
}
