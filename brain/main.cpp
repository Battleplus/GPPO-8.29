#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct AgentSpec {
    std::string pid;
    std::string type;
    std::pair<double, double> position;
    std::vector<std::string> sensors;
    std::map<std::string, int> munitions;
    double altitude_km = 0.0;
};

struct TargetInfo {
    std::string tid;
    std::string type;
    std::pair<double, double> pos;
    double value = 0.0;
    double threat = 0.0;
    bool confirmed = false;
    bool alive = true;
};

struct ReconTask {
    std::string platform;
    std::string cell;
    std::string sensor;
};

struct StrikeTask {
    std::string platform;
    std::string target;
    std::string munition;
    int qty = 0;
};

struct Aoi {
    std::string id = "A_3_4";
    int row = 3;
    int col = 4;
};

struct WorldState {
    Aoi aoi;
    std::vector<std::string> commander_aoi;
    std::pair<double, double> staging_position{150.0, -50.0};
    std::vector<TargetInfo> targets;
};

enum class MissionState {
    INIT,
    RECON_ALLOCATING_BY_MILP,
    RECON_PLANNING_BY_MPPI,
    RECON_PLAN_READY,
    RECON_EXECUTING,
    WAIT_RECON_RESULT,
    UPDATE_WORLD_STATE,
    ACTION_ALLOCATING_BY_MILP,
    POSITION_SELECTING,
    ACTION_PLANNING_BY_MPPI,
    ACTION_PLAN_READY,
    ACTION_EXECUTING,
    REPLAN,
    MISSION_COMPLETE,
    MISSION_FAILED,
};

enum class MissionEvent {
    START,
    RECON_PLAN_DISPATCHED,
    RECON_FINISHED,
    RECON_RESULT_RECEIVED,
    ACTION_PLAN_DISPATCHED,
    ACTION_FINISHED,
};

struct HistoryEntry {
    std::string timestamp;
    std::string state;
    std::string event;
    std::string detail;
};

struct MissionContext {
    std::string mission_id = "DEMO_001";
    MissionState state = MissionState::INIT;
    std::vector<AgentSpec> agents;
    WorldState world;
    std::vector<ReconTask> recon_allocation;
    std::vector<StrikeTask> action_allocation;
    std::vector<std::string> selected_positions;
    bool recon_formation_plan = false;
    bool action_formation_plan = false;
    std::vector<std::string> pending_strike_targets;
    std::vector<HistoryEntry> history;
};

struct Args {
    std::string mission_input;
    std::string aoi;
    std::string aois;
    std::string mission_id = "DEMO_001";
};

std::string stateName(MissionState state) {
    switch (state) {
    case MissionState::INIT: return "INIT";
    case MissionState::RECON_ALLOCATING_BY_MILP: return "RECON_ALLOCATING_BY_MILP";
    case MissionState::RECON_PLANNING_BY_MPPI: return "RECON_PLANNING_BY_MPPI";
    case MissionState::RECON_PLAN_READY: return "RECON_PLAN_READY";
    case MissionState::RECON_EXECUTING: return "RECON_EXECUTING";
    case MissionState::WAIT_RECON_RESULT: return "WAIT_RECON_RESULT";
    case MissionState::UPDATE_WORLD_STATE: return "UPDATE_WORLD_STATE";
    case MissionState::ACTION_ALLOCATING_BY_MILP: return "ACTION_ALLOCATING_BY_MILP";
    case MissionState::POSITION_SELECTING: return "POSITION_SELECTING";
    case MissionState::ACTION_PLANNING_BY_MPPI: return "ACTION_PLANNING_BY_MPPI";
    case MissionState::ACTION_PLAN_READY: return "ACTION_PLAN_READY";
    case MissionState::ACTION_EXECUTING: return "ACTION_EXECUTING";
    case MissionState::REPLAN: return "REPLAN";
    case MissionState::MISSION_COMPLETE: return "MISSION_COMPLETE";
    case MissionState::MISSION_FAILED: return "MISSION_FAILED";
    }
    return "UNKNOWN";
}

std::string eventName(MissionEvent event) {
    switch (event) {
    case MissionEvent::START: return "START";
    case MissionEvent::RECON_PLAN_DISPATCHED: return "RECON_PLAN_DISPATCHED";
    case MissionEvent::RECON_FINISHED: return "RECON_FINISHED";
    case MissionEvent::RECON_RESULT_RECEIVED: return "RECON_RESULT_RECEIVED";
    case MissionEvent::ACTION_PLAN_DISPATCHED: return "ACTION_PLAN_DISPATCHED";
    case MissionEvent::ACTION_FINISHED: return "ACTION_FINISHED";
    }
    return "UNKNOWN";
}

bool isWaiting(MissionState state) {
    return state == MissionState::RECON_PLAN_READY
        || state == MissionState::RECON_EXECUTING
        || state == MissionState::WAIT_RECON_RESULT
        || state == MissionState::ACTION_PLAN_READY
        || state == MissionState::ACTION_EXECUTING;
}

bool isTerminal(MissionState state) {
    return state == MissionState::MISSION_COMPLETE
        || state == MissionState::MISSION_FAILED;
}

std::string nowIso() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

void recordEvent(MissionContext& ctx, const std::string& event, const std::string& detail = "") {
    ctx.history.push_back({nowIso(), stateName(ctx.state), event, detail});
}

std::vector<AgentSpec> buildDemoAgents() {
    std::vector<AgentSpec> agents;
    for (int i = 1; i <= 5; ++i) {
        agents.push_back({
            "U" + std::to_string(i),
            "UAV",
            {150.0, -50.0},
            {"EO", "SAR", "ESM"},
            {},
            2.0,
        });
    }
    for (int i = 1; i <= 2; ++i) {
        agents.push_back({
            "H" + std::to_string(i),
            "HELI",
            {150.0, -50.0},
            {"MMW", "EOIR"},
            {{"HF", 16}, {"RKT", 76}, {"GUN", 1200}},
            3.0,
        });
    }
    return agents;
}

WorldState buildDemoWorld() {
    WorldState world;
    world.targets = {
        {"g1", "RADAR", {270.0, 260.0}, 1.0, 0.9, false, true},
        {"g2", "CP", {310.0, 180.0}, 0.95, 0.85, false, true},
        {"g3", "AV", {220.0, 310.0}, 0.7, 0.65, false, true},
    };
    return world;
}

void separator(const std::string& title) {
    std::cout << "\n========================================================================\n";
    std::cout << "  " << title << "\n";
    std::cout << "========================================================================\n";
}

std::vector<std::string> split(const std::string& value, char delimiter) {
    std::vector<std::string> result;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, delimiter)) {
        item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
        item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), item.end());
        if (!item.empty()) {
            result.push_back(item);
        }
    }
    return result;
}

Aoi parseAoi(const std::string& raw) {
    std::smatch match;
    static const std::regex aoi_pattern("^A[_-]?(\\d+)[_-](\\d+)$", std::regex_constants::icase);
    static const std::regex row_col_pattern("^(\\d+)\\s*[,;:]\\s*(\\d+)$");
    if (std::regex_match(raw, match, aoi_pattern) || std::regex_match(raw, match, row_col_pattern)) {
        const int row = std::stoi(match[1].str());
        const int col = std::stoi(match[2].str());
        return {"A_" + std::to_string(row) + "_" + std::to_string(col), row, col};
    }
    throw std::runtime_error("AOI must be like A_3_4 or 3,4: " + raw);
}

void applyTaskAreaOverrides(WorldState& world, const std::string& aoi, const std::string& aois) {
    if (!aois.empty()) {
        const auto items = split(aois, ',');
        if (items.empty()) {
            return;
        }
        world.commander_aoi.clear();
        for (const auto& item : items) {
            const Aoi parsed = parseAoi(item);
            if (world.commander_aoi.empty()) {
                world.aoi = parsed;
            }
            world.commander_aoi.push_back(parsed.id);
        }
    } else if (!aoi.empty()) {
        world.aoi = parseAoi(aoi);
        world.commander_aoi = {world.aoi.id};
    }
}

std::string joinStrings(const std::vector<std::string>& values) {
    if (values.empty()) {
        return "None";
    }
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << "'" << values[i] << "'";
    }
    oss << "]";
    return oss.str();
}

std::string brief(std::size_t count) {
    std::ostringstream oss;
    oss << "[" << count << " items]";
    return oss.str();
}

std::string readFile(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open mission input: " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void applyMissionInputLite(WorldState& world, const std::string& path) {
    const std::string text = readFile(path);
    std::vector<TargetInfo> parsed_targets;
    const std::regex tid_pattern("\"(?:tid|target_id)\"\\s*:\\s*\"([^\"]+)\"");
    for (std::sregex_iterator it(text.begin(), text.end(), tid_pattern), end; it != end; ++it) {
        const std::string tid = (*it)[1].str();
        const auto exists = std::any_of(parsed_targets.begin(), parsed_targets.end(), [&](const TargetInfo& target) {
            return target.tid == tid;
        });
        if (!exists) {
            parsed_targets.push_back({tid, "AV", {0.0, 0.0}, 0.5, 0.5, false, true});
        }
    }
    if (!parsed_targets.empty()) {
        world.targets = parsed_targets;
    }

    const std::regex aoi_pattern("A[_-]?(\\d+)[_-](\\d+)", std::regex_constants::icase);
    std::smatch match;
    if (std::regex_search(text, match, aoi_pattern)) {
        world.aoi = parseAoi(match[0].str());
        world.commander_aoi = {world.aoi.id};
    }
}

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string opt = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(name + " requires a value");
            }
            return argv[++i];
        };
        if (opt == "--mission-input") {
            args.mission_input = require_value(opt);
        } else if (opt == "--aoi") {
            args.aoi = require_value(opt);
        } else if (opt == "--aois") {
            args.aois = require_value(opt);
        } else if (opt == "--mission-id") {
            args.mission_id = require_value(opt);
        } else if (opt == "-h" || opt == "--help") {
            std::cout
                << "Usage: brain_demo [--mission-input PATH] [--aoi A_3_4] [--aois A_3_4,A_3_5] [--mission-id ID]\n"
                << "\n"
                << "C++ demo entry for the Brain mission pipeline.\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown option: " + opt);
        }
    }
    if (!args.aoi.empty() && !args.aois.empty()) {
        throw std::runtime_error("--aoi and --aois cannot be used together");
    }
    return args;
}

class MissionBrainDemo {
public:
    explicit MissionBrainDemo(MissionContext& context) : ctx_(context) {}

    MissionState start() {
        if (ctx_.state == MissionState::INIT) {
            recordEvent(ctx_, "START", "Mission initiated");
            transitionTo(MissionState::RECON_ALLOCATING_BY_MILP);
        }
        return runAutoChain();
    }

    MissionState dispatch(MissionEvent event) {
        switch (ctx_.state) {
        case MissionState::RECON_PLAN_READY:
            if (event == MissionEvent::RECON_PLAN_DISPATCHED) {
                recordEvent(ctx_, eventName(event), "Recon formation plan dispatched to platforms");
                transitionTo(MissionState::RECON_EXECUTING);
            }
            break;
        case MissionState::RECON_EXECUTING:
            if (event == MissionEvent::RECON_FINISHED) {
                recordEvent(ctx_, eventName(event), "Reconnaissance execution completed");
                transitionTo(MissionState::WAIT_RECON_RESULT);
            }
            break;
        case MissionState::WAIT_RECON_RESULT:
            if (event == MissionEvent::RECON_RESULT_RECEIVED) {
                recordEvent(ctx_, eventName(event), "Recon result received: true");
                confirmDemoTargets();
                transitionTo(MissionState::UPDATE_WORLD_STATE);
            }
            break;
        case MissionState::ACTION_PLAN_READY:
            if (event == MissionEvent::ACTION_PLAN_DISPATCHED) {
                recordEvent(ctx_, eventName(event), "Action formation plan dispatched to platforms");
                transitionTo(MissionState::ACTION_EXECUTING);
            }
            break;
        case MissionState::ACTION_EXECUTING:
            if (event == MissionEvent::ACTION_FINISHED) {
                recordEvent(ctx_, eventName(event), "Action execution completed");
                transitionTo(MissionState::MISSION_COMPLETE);
            }
            break;
        default:
            recordEvent(ctx_, "ILLEGAL_TRANSITION", "No handler for current state");
            break;
        }
        return runAutoChain();
    }

private:
    MissionContext& ctx_;

    void transitionTo(MissionState next) {
        ctx_.state = next;
    }

    MissionState runAutoChain() {
        for (int guard = 0; guard < 20; ++guard) {
            MissionState next = ctx_.state;
            switch (ctx_.state) {
            case MissionState::RECON_ALLOCATING_BY_MILP:
                next = doReconAllocate();
                break;
            case MissionState::RECON_PLANNING_BY_MPPI:
                next = doReconPlan();
                break;
            case MissionState::UPDATE_WORLD_STATE:
                next = doUpdateWorld();
                break;
            case MissionState::ACTION_ALLOCATING_BY_MILP:
                next = doActionAllocate();
                break;
            case MissionState::POSITION_SELECTING:
                next = doPositionSelect();
                break;
            case MissionState::ACTION_PLANNING_BY_MPPI:
                next = doActionPlan();
                break;
            default:
                if (isTerminal(ctx_.state)) {
                    recordEvent(ctx_, "TERMINAL", "Mission ended in state " + stateName(ctx_.state));
                }
                return ctx_.state;
            }
            transitionTo(next);
            if (isWaiting(next) || isTerminal(next)) {
                if (isTerminal(next)) {
                    recordEvent(ctx_, "TERMINAL", "Mission ended in state " + stateName(ctx_.state));
                }
                return ctx_.state;
            }
        }
        ctx_.state = MissionState::MISSION_FAILED;
        recordEvent(ctx_, "TERMINAL", "Auto transition guard exceeded");
        return ctx_.state;
    }

    MissionState doReconAllocate() {
        recordEvent(ctx_, "RECON_ALLOCATING", "Calling C++ MILPTaskAllocator stub");
        ctx_.recon_allocation.clear();
        const std::vector<std::string> cells = {"c0", "c1", "c2", "c3", "c4"};
        for (const auto& agent : ctx_.agents) {
            if (agent.type != "UAV") {
                continue;
            }
            const auto index = ctx_.recon_allocation.size() % cells.size();
            const std::string sensor = agent.sensors.empty() ? "EO" : agent.sensors.front();
            ctx_.recon_allocation.push_back({agent.pid, cells[index], sensor});
        }
        if (ctx_.recon_allocation.empty()) {
            recordEvent(ctx_, "ALGORITHM_FAILED", "No UAV platforms for recon allocation");
            return MissionState::MISSION_FAILED;
        }
        recordEvent(ctx_, "RECON_ALLOCATED", "MILP recon allocation stub succeeded");
        return MissionState::RECON_PLANNING_BY_MPPI;
    }

    MissionState doReconPlan() {
        recordEvent(ctx_, "RECON_PLANNING", "Calling C++ MPPIFormationPlanner stub");
        ctx_.recon_formation_plan = true;
        recordEvent(ctx_, "RECON_PLANNED", "MPPI recon planning stub succeeded");
        return MissionState::RECON_PLAN_READY;
    }

    void confirmDemoTargets() {
        for (auto& target : ctx_.world.targets) {
            if (target.tid == "g1" || target.tid == "g2" || target.tid == "g3") {
                target.confirmed = true;
            }
        }
    }

    MissionState doUpdateWorld() {
        recordEvent(ctx_, "UPDATE_WORLD", "Updating world state from recon result");
        ctx_.pending_strike_targets.clear();
        for (const auto& target : ctx_.world.targets) {
            if (target.alive && target.confirmed) {
                ctx_.pending_strike_targets.push_back(target.tid);
            }
        }
        if (!ctx_.pending_strike_targets.empty()) {
            recordEvent(ctx_, "WORLD_UPDATED", "World state updated; pending action tasks remain");
            return MissionState::ACTION_ALLOCATING_BY_MILP;
        }
        recordEvent(ctx_, "WORLD_UPDATED", "World state updated; no pending tasks");
        return MissionState::MISSION_COMPLETE;
    }

    MissionState doActionAllocate() {
        recordEvent(ctx_, "ACTION_ALLOCATING", "Calling C++ MILPTaskAllocator action stub");
        ctx_.action_allocation.clear();
        std::vector<std::string> helis;
        for (const auto& agent : ctx_.agents) {
            if (agent.type == "HELI") {
                helis.push_back(agent.pid);
            }
        }
        const std::size_t count = std::min(helis.size(), ctx_.pending_strike_targets.size());
        for (std::size_t i = 0; i < count; ++i) {
            ctx_.action_allocation.push_back({helis[i], ctx_.pending_strike_targets[i], "HF", 1});
        }
        if (ctx_.action_allocation.empty()) {
            recordEvent(ctx_, "ALGORITHM_FAILED", "No HELI platforms or confirmed targets for action allocation");
            return MissionState::MISSION_FAILED;
        }
        recordEvent(ctx_, "ACTION_ALLOCATED", "MILP action allocation stub succeeded");
        return MissionState::POSITION_SELECTING;
    }

    MissionState doPositionSelect() {
        recordEvent(ctx_, "POSITION_SELECTING", "Calling C++ PositionSelector stub");
        ctx_.selected_positions.clear();
        for (const auto& task : ctx_.action_allocation) {
            ctx_.selected_positions.push_back(task.platform + "->" + task.target + "@target_point");
        }
        recordEvent(ctx_, "POSITION_SELECTED", "Position selection stub succeeded");
        return MissionState::ACTION_PLANNING_BY_MPPI;
    }

    MissionState doActionPlan() {
        recordEvent(ctx_, "ACTION_PLANNING", "Calling C++ MPPIFormationPlanner action stub");
        ctx_.action_formation_plan = true;
        recordEvent(ctx_, "ACTION_PLANNED", "MPPI action planning stub succeeded");
        return MissionState::ACTION_PLAN_READY;
    }
};

void printIdList(const std::string& label, const std::vector<std::string>& ids) {
    std::cout << "  " << std::left << std::setw(10) << label << " = " << joinStrings(ids) << "\n";
}

int run(int argc, char** argv) {
    const Args args = parseArgs(argc, argv);

    MissionContext ctx;
    ctx.mission_id = args.mission_id;
    ctx.agents = buildDemoAgents();
    ctx.world = buildDemoWorld();

    if (!args.mission_input.empty()) {
        applyMissionInputLite(ctx.world, args.mission_input);
    }
    applyTaskAreaOverrides(ctx.world, args.aoi, args.aois);

    separator("1. Build mission context");

    separator("2. Create adapters (C++ stub implementations)");
    std::cout << "  MILPTaskAllocator  : OK (C++ stub)\n";
    std::cout << "  MPPIFormationPlanner: OK (C++ stub)\n";
    std::cout << "  PositionSelector   : OK (C++ stub)\n";

    separator("3. Create MissionBrain and START");
    std::vector<std::string> agent_ids;
    for (const auto& agent : ctx.agents) {
        agent_ids.push_back(agent.pid);
    }
    std::vector<std::string> target_ids;
    for (const auto& target : ctx.world.targets) {
        target_ids.push_back(target.tid);
    }

    std::cout << "  mission_id = " << ctx.mission_id << "\n";
    std::cout << "  task_areas = " << joinStrings(ctx.world.commander_aoi) << "\n";
    printIdList("agents", agent_ids);
    printIdList("targets", target_ids);

    MissionBrainDemo brain(ctx);
    MissionState state = brain.start();
    std::cout << "  After START: " << stateName(state) << "\n";
    std::cout << "  recon_allocation: " << brief(ctx.recon_allocation.size()) << "\n";
    std::cout << "  recon_formation_plan set: " << (ctx.recon_formation_plan ? "true" : "false") << "\n";

    if (isWaiting(state)) {
        separator("4. Waiting states - dispatch external events");
        std::cout << "  Current state: " << stateName(state) << "\n";
        state = brain.dispatch(MissionEvent::RECON_PLAN_DISPATCHED);
        std::cout << "  After RECON_PLAN_DISPATCHED: " << stateName(state) << "\n";
        state = brain.dispatch(MissionEvent::RECON_FINISHED);
        std::cout << "  After RECON_FINISHED: " << stateName(state) << "\n";
        state = brain.dispatch(MissionEvent::RECON_RESULT_RECEIVED);
        std::cout << "  After RECON_RESULT_RECEIVED + auto chain: " << stateName(state) << "\n";
    }

    if (isWaiting(state)) {
        separator("5. Action pipeline - dispatch external events");
        std::cout << "  Current state: " << stateName(state) << "\n";
        std::cout << "  action_allocation: " << brief(ctx.action_allocation.size()) << "\n";
        std::cout << "  selected_positions: " << brief(ctx.selected_positions.size()) << "\n";
        state = brain.dispatch(MissionEvent::ACTION_PLAN_DISPATCHED);
        std::cout << "  After ACTION_PLAN_DISPATCHED: " << stateName(state) << "\n";
        state = brain.dispatch(MissionEvent::ACTION_FINISHED);
        std::cout << "  After ACTION_FINISHED: " << stateName(state) << "\n";
    }

    separator("6. Mission complete");
    std::cout << "  Final state : " << stateName(state) << "\n";
    std::cout << "  Terminal    : " << (isTerminal(state) ? "true" : "false") << "\n";
    std::cout << "  History entries: " << ctx.history.size() << "\n\n";
    std::cout << "  State transition trace:\n";
    for (const auto& entry : ctx.history) {
        std::cout << "    " << std::left << std::setw(19) << entry.timestamp.substr(0, 19)
                  << "  [" << std::setw(30) << entry.state << "]  "
                  << std::setw(24) << entry.event << "  "
                  << entry.detail << "\n";
    }
    return isTerminal(state) && state != MissionState::MISSION_FAILED ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& exc) {
        std::cerr << "brain_demo: " << exc.what() << "\n";
        return 2;
    }
}
