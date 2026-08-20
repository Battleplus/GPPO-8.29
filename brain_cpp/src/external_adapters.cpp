#include "brain_cpp/external_adapters.hpp"

#include "SarSearchPlannerClient.hpp"
#include "MilpClient.hpp"
#include "PerchPositionSelectorClient.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace brain_cpp {
namespace {

std::string jsonEscape(const std::string& value);

std::string contextToMilpJson(const MissionContext& context) {
    std::ostringstream out;
    out << "{\"aoi\":{\"row\":" << context.world.aoi.row
        << ",\"col\":" << context.world.aoi.col << "},\"staging_position\":["
        << context.world.staging_position[0] << ',' << context.world.staging_position[1]
        << "],\"commander_AOI\":[";
    for (std::size_t i = 0; i < context.world.commander_aoi.size(); ++i) {
        if (i) out << ',';
        out << '"' << jsonEscape(context.world.commander_aoi[i]) << '"';
    }
    out << "],\"grid_weather\":{";
    bool first = true;
    for (const auto& item : context.world.weather) {
        if (!first) out << ',';
        first = false;
        out << '"' << jsonEscape(item.first) << "\":" << item.second;
    }
    out << "},\"grid_terrain\":{";
    first = true;
    for (const auto& item : context.world.terrain) {
        if (!first) out << ',';
        first = false;
        out << '"' << jsonEscape(item.first) << "\":" << item.second;
    }
    out << "},\"platforms\":[";
    first = true;
    for (const auto& agent : context.agents) {
        if (!first) out << ',';
        first = false;
        out << "{\"pid\":\"" << jsonEscape(agent.pid) << "\",\"type\":\""
            << jsonEscape(agent.type) << "\",\"pos\":[" << agent.position[0] << ','
            << agent.position[1] << "],\"sensors_mounted\":[";
        for (std::size_t i = 0; i < agent.sensors.size(); ++i) {
            if (i) out << ',';
            out << '"' << jsonEscape(agent.sensors[i]) << '"';
        }
        out << "],\"munitions\":{";
        bool firstMunition = true;
        for (const auto& item : agent.munitions) {
            if (!firstMunition) out << ',';
            firstMunition = false;
            out << '"' << jsonEscape(item.first) << "\":" << item.second;
        }
        out << "},\"alt\":" << agent.altitude_km << ",\"lost\":"
            << (agent.lost ? "true" : "false") << '}';
    }
    out << "],\"targets\":[";
    first = true;
    for (const auto& target : context.world.targets) {
        if (!first) out << ',';
        first = false;
        out << "{\"tid\":\"" << jsonEscape(target.tid) << "\",\"type\":\""
            << jsonEscape(target.type) << "\",\"pos\":[" << target.pos[0] << ','
            << target.pos[1] << "],\"value\":" << target.value << ",\"threat\":"
            << target.threat << ",\"confirmed\":" << (target.confirmed ? "true" : "false")
            << ",\"alive\":" << (target.alive ? "true" : "false") << '}';
    }
    out << "]}";
    return out.str();
}

}  // namespace

PerchPositionSelector::PerchPositionSelector(PerchPositionSelectorOptions options)
    : options_(std::move(options)) {}

AlgorithmResult<std::vector<Position>>
PerchPositionSelector::select(
    const MissionContext& context,
    const std::vector<StrikeTask>& allocation) {
    if (allocation.empty()) {
        return AlgorithmResult<std::vector<Position>>::fail(
            "No strike allocation for Perch position selection");
    }

    perch_cpp::Situation situation;
    situation.mission_id = context.mission_id;
    situation.staging_x_km = context.world.staging_position[0];
    situation.staging_y_km = context.world.staging_position[1];
    situation.weather = context.world.weather;
    situation.terrain = context.world.terrain;
    for (const auto& agent : context.agents) {
        perch_cpp::Agent converted;
        converted.pid = agent.pid;
        converted.type = agent.type;
        converted.x_km = agent.position[0];
        converted.y_km = agent.position[1];
        converted.altitude_km = agent.altitude_km;
        converted.sensors = agent.sensors;
        converted.munitions = agent.munitions;
        converted.lost = agent.lost;
        situation.agents.push_back(std::move(converted));
    }
    for (const auto& target : context.world.targets) {
        perch_cpp::Target converted;
        converted.tid = target.tid;
        converted.type = target.type;
        converted.x_km = target.pos[0];
        converted.y_km = target.pos[1];
        converted.value = target.value;
        converted.threat = target.threat;
        converted.confirmed = target.confirmed;
        converted.alive = target.alive;
        situation.targets.push_back(std::move(converted));
    }

    std::vector<perch_cpp::StrikeTask> tasks;
    for (const auto& task : allocation) {
        perch_cpp::StrikeTask converted;
        converted.platform = task.platform;
        converted.target = task.target;
        converted.munition = task.munition;
        converted.qty = task.qty;
        converted.role = task.role;
        converted.aoi = task.aoi;
        converted.assigned_munitions = task.assigned_munitions;
        tasks.push_back(std::move(converted));
    }

    perch_cpp::ClientOptions clientOptions;
    clientOptions.python_executable = options_.python_executable;
    clientOptions.bridge_script = options_.bridge_script;
    clientOptions.attack_region_mode = options_.attack_region_mode;
    clientOptions.preference = options_.preference;
    clientOptions.terrain_mode = options_.terrain_mode;
    clientOptions.top_k = options_.top_k;
    clientOptions.use_pymoo = options_.use_pymoo;
    clientOptions.attack_region_strict = options_.attack_region_strict;

    try {
        const auto selected = perch_cpp::SelectAttackPositions(
            situation, tasks, clientOptions);
        std::vector<Position> positions;
        positions.reserve(selected.size());
        for (const auto& item : selected) {
            Position position;
            position.pos_id = item.pos_id;
            position.x = item.x;
            position.y = item.y;
            position.z = item.z;
            position.kind = item.kind;
            position.metadata = item.metadata;
            position.metadata["adapter"] = "perch_cpp_bridge";
            positions.push_back(std::move(position));
        }
        return AlgorithmResult<std::vector<Position>>::ok(std::move(positions));
    } catch (const std::exception& exc) {
        return AlgorithmResult<std::vector<Position>>::fail(exc.what());
    }
}

namespace {

std::string contextToMultiAoiMilpJson(const MissionContext& context) {
    std::ostringstream out;
    out << "{\"aois\":[";
    for (std::size_t i = 0; i < context.world.aois.size(); ++i) {
        if (i) out << ',';
        const auto& aoi = context.world.aois[i];
        out << "{\"id\":\"" << jsonEscape(aoi.id) << "\",\"row\":" << aoi.row
            << ",\"col\":" << aoi.col << ",\"priority\":" << aoi.priority
            << ",\"target_prior\":" << aoi.target_prior
            << ",\"target_value\":" << aoi.target_value
            << ",\"target_threat\":" << aoi.target_threat << '}';
    }
    out << "],\"platforms\":[";
    bool first = true;
    for (const auto& agent : context.agents) {
        if (!first) out << ',';
        first = false;
        out << "{\"pid\":\"" << jsonEscape(agent.pid) << "\",\"type\":\""
            << jsonEscape(agent.type) << "\",\"pos\":[" << agent.position[0] << ','
            << agent.position[1] << "],\"sensors_mounted\":[";
        for (std::size_t i = 0; i < agent.sensors.size(); ++i) {
            if (i) out << ',';
            out << '"' << jsonEscape(agent.sensors[i]) << '"';
        }
        out << "],\"munitions\":{";
        bool firstMunition = true;
        for (const auto& item : agent.munitions) {
            if (!firstMunition) out << ',';
            firstMunition = false;
            out << '"' << jsonEscape(item.first) << "\":" << item.second;
        }
        out << "},\"alt\":" << agent.altitude_km << ",\"lost\":"
            << (agent.lost ? "true" : "false") << '}';
    }
    out << "],\"targets\":[";
    first = true;
    for (const auto& target : context.world.targets) {
        if (!first) out << ',';
        first = false;
        out << "{\"tid\":\"" << jsonEscape(target.tid) << "\",\"type\":\""
            << jsonEscape(target.type) << "\",\"pos\":[" << target.pos[0] << ','
            << target.pos[1] << "],\"value\":" << target.value << ",\"threat\":"
            << target.threat << ",\"confirmed\":" << (target.confirmed ? "true" : "false")
            << ",\"alive\":" << (target.alive ? "true" : "false") << '}';
    }
    out << "],\"staging_position\":[" << context.world.staging_position[0] << ','
        << context.world.staging_position[1] << "],\"grid_weather\":{";
    first = true;
    for (const auto& item : context.world.weather) {
        if (!first) out << ',';
        first = false;
        out << '"' << jsonEscape(item.first) << "\":" << item.second;
    }
    out << "},\"aoi_route_state\":null,\"execution_feedback\":null,\"cycle_id\":0}";
    return out.str();
}

std::string jsonString(const std::string& object, const std::string& key) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    std::smatch match;
    return std::regex_search(object, match, pattern) ? match[1].str() : std::string{};
}

int jsonInt(const std::string& object, const std::string& key) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(-?[0-9]+)");
    std::smatch match;
    return std::regex_search(object, match, pattern) ? std::stoi(match[1].str()) : 0;
}

std::string jsonObject(const std::string& json, const std::string& key) {
    const auto keyPos = json.find("\"" + key + "\"");
    const auto begin = keyPos == std::string::npos ? std::string::npos : json.find('{', keyPos);
    if (begin == std::string::npos) return {};
    int depth = 0;
    bool quoted = false;
    for (std::size_t i = begin; i < json.size(); ++i) {
        if (json[i] == '"' && (i == 0 || json[i - 1] != '\\')) quoted = !quoted;
        if (quoted) continue;
        if (json[i] == '{') ++depth;
        else if (json[i] == '}' && --depth == 0) return json.substr(begin, i - begin + 1);
    }
    return {};
}

std::vector<std::string> taskObjects(const std::string& json) {
    const auto tasks = json.find("\"tasks\"");
    const auto begin = tasks == std::string::npos ? std::string::npos : json.find('[', tasks);
    std::vector<std::string> result;
    if (begin == std::string::npos) return result;
    int depth = 0;
    std::size_t start = 0;
    bool quoted = false;
    for (std::size_t i = begin + 1; i < json.size(); ++i) {
        if (json[i] == '"' && (i == 0 || json[i - 1] != '\\')) quoted = !quoted;
        if (quoted) continue;
        if (json[i] == '{') { if (depth++ == 0) start = i; }
        else if (json[i] == '}' && --depth == 0) result.push_back(json.substr(start, i - start + 1));
        else if (json[i] == ']' && depth == 0) break;
    }
    return result;
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream out;
    for (const char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << ch; break;
        }
    }
    return out.str();
}

std::string shellQuote(const std::string& value) {
#ifdef _WIN32
    std::string escaped;
    for (const char ch : value) {
        if (ch == '"') {
            escaped += "\\\"";
        } else {
            escaped += ch;
        }
    }
    return "\"" + escaped + "\"";
#else
    std::string escaped = "'";
    for (const char ch : value) {
        if (ch == '\'') {
            escaped += "'\\''";
        } else {
            escaped += ch;
        }
    }
    escaped += "'";
    return escaped;
#endif
}

std::filesystem::path makeTempPath(const std::string& prefix, const std::string& suffix) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path()
        / (prefix + "_" + std::to_string(stamp) + suffix);
}

std::vector<std::string> split(const std::string& raw, char delim) {
    std::vector<std::string> result;
    std::stringstream stream(raw);
    std::string item;
    while (std::getline(stream, item, delim)) {
        if (!item.empty()) {
            result.push_back(item);
        }
    }
    return result;
}

int numericSuffix(const std::string& value) {
    std::string digits;
    for (const char ch : value) {
        if (ch >= '0' && ch <= '9') {
            digits.push_back(ch);
        }
    }
    if (digits.empty()) {
        return 9999;
    }
    return std::stoi(digits);
}

std::string upperAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        if (ch >= 'a' && ch <= 'z') {
            return static_cast<char>(ch - 'a' + 'A');
        }
        return static_cast<char>(ch);
    });
    return value;
}

std::vector<std::string> sortedSubareaPlatforms(const std::vector<ReconTask>& allocation) {
    std::set<std::string> unique;
    for (const auto& task : allocation) {
        if (task.role == "subarea_search" && !task.platform.empty()) {
            unique.insert(task.platform);
        }
    }
    std::vector<std::string> result(unique.begin(), unique.end());
    std::sort(result.begin(), result.end(), [](const std::string& lhs, const std::string& rhs) {
        const int li = numericSuffix(lhs);
        const int ri = numericSuffix(rhs);
        if (li != ri) {
            return li < ri;
        }
        return lhs < rhs;
    });
    if (result.size() > 4) {
        result.resize(4);
    }
    return result;
}

std::vector<std::string> sortedTargetIds(const MissionContext& context) {
    std::vector<std::string> result;
    for (const auto& target : context.world.targets) {
        if (!target.tid.empty()) {
            result.push_back(target.tid);
        }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

Point2 cellCenterRelativeKm(const Aoi& aoi, const std::string& cell) {
    const double x0 = static_cast<double>(aoi.col - 1) * 50.0;
    const double y0 = static_cast<double>(aoi.row - 1) * 50.0;
    const double half = 12.5;
    double absX = x0 + 25.0;
    double absY = y0 + 25.0;
    if (cell == "c1") {
        absX = x0 + half;
        absY = y0 + half;
    } else if (cell == "c2") {
        absX = x0 + 3.0 * half;
        absY = y0 + half;
    } else if (cell == "c3") {
        absX = x0 + half;
        absY = y0 + 3.0 * half;
    } else if (cell == "c4") {
        absX = x0 + 3.0 * half;
        absY = y0 + 3.0 * half;
    }
    return {absX - 150.0, absY - 150.0};
}

std::string buildPreallocationJson(const MissionContext& context) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"aoi_id\": \"" << jsonEscape(context.world.aoi.id) << "\",\n";
    out << "  \"tasks\": [\n";

    bool first = true;
    auto comma = [&]() {
        if (!first) {
            out << ",\n";
        }
        first = false;
    };

    for (const auto& task : context.recon_allocation) {
        comma();
        out << "    {"
            << "\"task_type\":\"recon\","
            << "\"platform\":\"" << jsonEscape(task.platform) << "\","
            << "\"cell\":\"" << jsonEscape(task.cell) << "\","
            << "\"sensor_used\":\"" << jsonEscape(task.sensor.empty() ? "SAR" : task.sensor) << "\","
            << "\"role\":\"" << jsonEscape(task.role) << "\""
            << "}";
    }

    for (const auto& target : context.world.targets) {
        if (!target.tid.empty()) {
            comma();
            out << "    {"
                << "\"task_type\":\"strike\","
                << "\"platform\":\"H0\","
                << "\"target\":\"" << jsonEscape(target.tid) << "\","
                << "\"munition\":\"HF\","
                << "\"qty\":1,"
                << "\"role\":\"target_slot\""
                << "}";
        }
    }

    out << "\n  ]\n";
    out << "}\n";
    return out.str();
}

std::string buildRequestJson(
    const PpoBridgeOptions& options,
    const MissionContext& context,
    const std::string& eventJson,
    const std::filesystem::path& outputPath,
    const std::filesystem::path& tasksOutputPath) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"model_path\": \"" << jsonEscape(options.model_path) << "\",\n";
    out << "  \"preallocation_json\": " << buildPreallocationJson(context) << ",\n";
    out << "  \"event\": " << eventJson << ",\n";
    out << "  \"output_path\": \"" << jsonEscape(outputPath.string()) << "\",\n";
    out << "  \"tasks_output_path\": \"" << jsonEscape(tasksOutputPath.string()) << "\",\n";
    out << "  \"deterministic\": " << (options.deterministic ? "true" : "false") << "\n";
    out << "}\n";
    return out.str();
}

std::map<std::string, std::string> internalPlatformMap(const std::vector<ReconTask>& allocation) {
    const auto platforms = sortedSubareaPlatforms(allocation);
    std::map<std::string, std::string> result;
    for (std::size_t index = 0; index < platforms.size(); ++index) {
        result["U" + std::to_string(index)] = platforms[index];
    }
    return result;
}

std::vector<ReconTask> readPpoTasksTsv(
    const std::filesystem::path& path,
    const MissionContext& context,
    const std::string& removedPlatform) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open PPO task output: " + path.string());
    }

    std::vector<ReconTask> result;
    for (const auto& task : context.recon_allocation) {
        if (task.role != "subarea_search" && task.platform != removedPlatform) {
            result.push_back(task);
        }
    }

    const auto platformMap = internalPlatformMap(context.recon_allocation);
    std::string line;
    bool header = true;
    while (std::getline(in, line)) {
        if (header) {
            header = false;
            continue;
        }
        const auto fields = split(line, '\t');
        if (fields.size() < 6) {
            continue;
        }
        const std::string& internalId = fields[0];
        const bool alive = fields[1] == "1";
        const std::string& taskType = fields[2];
        const std::string& sensor = fields[3];
        const std::string& regions = fields[4];

        const auto foundPlatform = platformMap.find(internalId);
        if (foundPlatform == platformMap.end() || !alive || taskType != "SEARCH") {
            continue;
        }
        for (const auto& region : split(regions, ',')) {
            const int rid = numericSuffix(region);
            if (rid < 0 || rid > 3) {
                continue;
            }
            ReconTask task;
            task.platform = foundPlatform->second;
            task.cell = "c" + std::to_string(rid + 1);
            task.sensor = sensor.empty() ? "SAR" : sensor;
            task.role = "subarea_search";
            task.aoi = context.world.aoi.id;
            result.push_back(task);
        }
    }
    return result;
}

int platformUid(const MissionContext& context, const std::string& platformId) {
    const auto platforms = sortedSubareaPlatforms(context.recon_allocation);
    for (std::size_t index = 0; index < platforms.size(); ++index) {
        if (platforms[index] == platformId) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

int targetTid(const MissionContext& context, const std::string& targetId) {
    const auto targets = sortedTargetIds(context);
    for (std::size_t index = 0; index < targets.size(); ++index) {
        if (targets[index] == targetId) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

}  // namespace

class MilpTaskAllocator::Impl {
public:
    explicit Impl(const MilpTaskAllocatorOptions& options)
        : singleClient(options.milp_dir, options.solver, options.time_limit_s, options.verbose),
          multiClient(options.milp_dir, options.solver, options.time_limit_s, options.verbose),
          verbose(options.verbose) {}

    const std::string& solve(const MissionContext& context) {
        const bool multiAoi = context.world.aois.size() > 1;
        std::string input = multiAoi
            ? contextToMultiAoiMilpJson(context)
            : contextToMilpJson(context);
        if (multiAoi && !aoiRouteState.empty()) {
            const std::string marker = "\"aoi_route_state\":null";
            const auto pos = input.find(marker);
            if (pos != std::string::npos) {
                input.replace(pos, marker.size(), "\"aoi_route_state\":" + aoiRouteState);
            }
        }
        if (input != cachedInput) {
            if (verbose > 0) {
                std::cout << "\n===== " << (multiAoi ? "MULTI-AOI" : "SINGLE-AOI")
                          << " MILP INPUT JSON =====\n" << input
                          << "\n===== END MILP INPUT =====\n" << std::flush;
            }
            cachedOutput = multiAoi
                ? multiClient.runJson(input)
                : singleClient.solveJson(input);
            if (multiAoi) {
                const std::string state = jsonObject(cachedOutput, "aoi_route_state");
                if (!state.empty()) aoiRouteState = state;
            }
            cachedInput = input;
            if (verbose > 0) {
                std::cout << "\n===== " << (multiAoi ? "MULTI-AOI" : "SINGLE-AOI")
                          << " MILP OUTPUT JSON =====\n" << cachedOutput
                          << "\n===== END MILP OUTPUT =====\n" << std::flush;
            }
        }
        return cachedOutput;
    }

    milp::SingleAoiMilpClient singleClient;
    milp::MultiAoiMilpClient multiClient;
    std::string cachedInput;
    std::string cachedOutput;
    std::string aoiRouteState;
    int verbose = 0;
};

MilpTaskAllocator::MilpTaskAllocator(MilpTaskAllocatorOptions options)
    : impl_(std::make_unique<Impl>(options)) {}

MilpTaskAllocator::~MilpTaskAllocator() = default;

AlgorithmResult<std::vector<ReconTask>>
MilpTaskAllocator::allocateRecon(const MissionContext& context) {
    try {
        std::vector<ReconTask> tasks;
        for (const auto& object : taskObjects(impl_->solve(context))) {
            if (jsonString(object, "task_type") != "recon") continue;
            ReconTask task;
            task.platform = jsonString(object, "platform");
            task.cell = jsonString(object, "cell");
            task.sensor = jsonString(object, "sensor_used");
            if (task.sensor.empty()) task.sensor = jsonString(object, "sensor");
            task.role = jsonString(object, "role");
            task.aoi = jsonString(object, "aoi");
            tasks.push_back(std::move(task));
        }
        if (tasks.empty()) {
            const auto& output = impl_->solve(context);
            const std::string status = jsonString(output, "solve_status");
            return AlgorithmResult<std::vector<ReconTask>>::fail(
                "MILP returned no recon tasks"
                + (status.empty() ? std::string{} : " (solve_status=" + status + ")"));
        }
        return AlgorithmResult<std::vector<ReconTask>>::ok(std::move(tasks));
    } catch (const std::exception& exc) {
        return AlgorithmResult<std::vector<ReconTask>>::fail(std::string("MILP recon allocation failed: ") + exc.what());
    }
}

AlgorithmResult<std::vector<StrikeTask>>
MilpTaskAllocator::allocateAction(
    const MissionContext& context, const std::vector<std::string>& targetIds,
    bool includeEngaged) {
    try {
        std::set<std::string> requested(targetIds.begin(), targetIds.end());
        std::vector<StrikeTask> tasks;
        for (const auto& object : taskObjects(impl_->solve(context))) {
            if (jsonString(object, "task_type") != "strike") continue;
            StrikeTask task;
            task.platform = jsonString(object, "platform");
            task.target = jsonString(object, "target");
            if (!requested.empty() && requested.count(task.target) == 0) continue;
            if (!includeEngaged && context.engaged_targets.count(task.target) != 0) continue;
            task.munition = jsonString(object, "munition");
            task.qty = jsonInt(object, "qty");
            task.role = jsonString(object, "role");
            task.aoi = jsonString(object, "aoi");
            if (!task.munition.empty() && task.qty > 0) task.assigned_munitions[task.munition] = task.qty;
            tasks.push_back(std::move(task));
        }
        if (tasks.empty()) return AlgorithmResult<std::vector<StrikeTask>>::fail("MILP returned no eligible strike tasks");
        return AlgorithmResult<std::vector<StrikeTask>>::ok(std::move(tasks));
    } catch (const std::exception& exc) {
        return AlgorithmResult<std::vector<StrikeTask>>::fail(std::string("MILP action allocation failed: ") + exc.what());
    }
}

std::string patrolPatternForSensor(const std::string& sensor) {
    const std::string normalized = upperAscii(sensor);
    if (normalized == "SAR") {
        return "sar_polygon";
    }
    if (normalized == "ESM") {
        return "figure_eight";
    }
    if (normalized == "MMW") {
        return "sar_rounded";
    }
    if (normalized == "EO" || normalized == "EOIR" || normalized.empty()) {
        return "racetrack";
    }
    throw std::invalid_argument("Unsupported patrol sensor: " + sensor);
}

SarSearchPatrolPlanner::SarSearchPatrolPlanner(SarSearchPatrolOptions options)
    : options_(std::move(options)) {}

AlgorithmResult<std::vector<PatrolPlan>>
SarSearchPatrolPlanner::planPatrols(
    const MissionContext& context,
    const std::vector<ReconTask>& allocation,
    const FormationPlan& transitPlan) {
    (void)transitPlan;
    std::vector<sar_search_planner::SearchTask> tasks;
    std::map<std::string, std::string> patternsByPlatform;
    std::map<std::string, std::string> sensorsByPlatform;
    for (const auto& task : allocation) {
        if (task.platform.empty() || task.cell.empty() || task.role == "track") {
            continue;
        }
        const auto center = cellCenterRelativeKm(context.world.aoi, task.cell);
        sar_search_planner::SearchTask searchTask;
        searchTask.platform_id = task.platform;
        searchTask.center_x_km = center[0];
        searchTask.center_y_km = center[1];
        searchTask.width_km = task.cell == "c0" ? options_.aoi_width_km : options_.subarea_width_km;
        searchTask.height_km = task.cell == "c0" ? options_.aoi_height_km : options_.subarea_height_km;
        const std::string sensor = task.sensor.empty() ? "SAR" : upperAscii(task.sensor);
        try {
            searchTask.pattern = options_.pattern.empty()
                ? patrolPatternForSensor(sensor)
                : options_.pattern;
        } catch (const std::exception& exc) {
            return AlgorithmResult<std::vector<PatrolPlan>>::fail(exc.what());
        }
        searchTask.altitude_agl_m = options_.altitude_agl_m;
        patternsByPlatform[task.platform] = searchTask.pattern;
        sensorsByPlatform[task.platform] = sensor;
        tasks.push_back(searchTask);
    }
    if (tasks.empty()) {
        return AlgorithmResult<std::vector<PatrolPlan>>::fail("No recon tasks for SAR patrol planner");
    }

    sar_search_planner::PlannerClientOptions clientOptions;
    clientOptions.python_executable = options_.python_executable;
    clientOptions.bridge_script = options_.bridge_script;

    std::vector<sar_search_planner::Waypoint> points;
    try {
        points = sar_search_planner::PlanSarSearchPath(tasks, clientOptions);
    } catch (const std::exception& exc) {
        return AlgorithmResult<std::vector<PatrolPlan>>::fail(exc.what());
    }

    std::map<std::string, PatrolPlan> plans;
    std::map<std::string, std::set<std::string>> cellsByPlatform;
    for (const auto& task : allocation) {
        if (!task.platform.empty() && !task.cell.empty()) {
            cellsByPlatform[task.platform].insert(task.cell);
        }
    }
    for (const auto& point : points) {
        auto& plan = plans[point.platform_id];
        if (plan.platform.empty()) {
            plan.platform = point.platform_id;
            plan.pattern = patternsByPlatform[point.platform_id];
            plan.sensor = sensorsByPlatform[point.platform_id];
            plan.metadata = {
                {"planner", "sar_search_planner_cpp_bridge"},
                {"aoi", context.world.aoi.id},
                {"sensor", plan.sensor},
                {"pattern", plan.pattern},
            };
            const auto foundCells = cellsByPlatform.find(point.platform_id);
            if (foundCells != cellsByPlatform.end()) {
                plan.cells.assign(foundCells->second.begin(), foundCells->second.end());
            }
        }
        plan.waypoints.push_back({point.x, point.y, point.z});
        plan.metadata["total_km"] = std::to_string(point.total_km);
    }

    std::vector<PatrolPlan> result;
    for (auto& item : plans) {
        result.push_back(std::move(item.second));
    }
    if (result.empty()) {
        return AlgorithmResult<std::vector<PatrolPlan>>::fail("SAR planner produced no patrol waypoints");
    }
    return AlgorithmResult<std::vector<PatrolPlan>>::ok(result);
}

PpoAvoidanceController::PpoAvoidanceController(
    const std::string& modelPath,
    ppo_avoidance::Config config)
    : controller_(modelPath, config) {}

ppo_avoidance::Vec3 PpoAvoidanceController::compute(
    const ppo_avoidance::Vec3& position,
    const ppo_avoidance::Vec3& velocity,
    const ppo_avoidance::Vec3& waypoint,
    const std::vector<ppo_avoidance::Obstacle>& obstacles) const {
    return controller_.compute(position, velocity, waypoint, obstacles);
}

PpoBridgeReallocator::PpoBridgeReallocator(PpoBridgeOptions options)
    : options_(std::move(options)) {}

AlgorithmResult<std::vector<ReconTask>>
PpoBridgeReallocator::handlePlatformLoss(
    const MissionContext& context,
    const std::string& platformId) {
    const int uid = platformUid(context, platformId);
    if (uid < 0) {
        return AlgorithmResult<std::vector<ReconTask>>::fail(
            "Platform is not a PPO subarea UAV: " + platformId);
    }
    const std::string eventJson =
        "{\"event_type\":\"UAV_DAMAGE\",\"uav_id\":" + std::to_string(uid) + "}";
    return runEvent(context, eventJson, platformId);
}

AlgorithmResult<std::vector<ReconTask>>
PpoBridgeReallocator::handleTargetDiscovered(
    const MissionContext& context,
    const std::string& platformId,
    const std::string& targetId) {
    const int uid = platformUid(context, platformId);
    if (uid < 0) {
        return AlgorithmResult<std::vector<ReconTask>>::fail(
            "Platform is not a PPO subarea UAV: " + platformId);
    }
    const int tid = targetTid(context, targetId);
    if (tid < 0) {
        return AlgorithmResult<std::vector<ReconTask>>::fail(
            "Target is not represented in PPO request: " + targetId);
    }
    const std::string eventJson =
        "{\"event_type\":\"TARGET_DISCOVERED\",\"uav_id\":" + std::to_string(uid)
        + ",\"target_id\":" + std::to_string(tid) + "}";
    return runEvent(context, eventJson);
}

AlgorithmResult<std::vector<ReconTask>>
PpoBridgeReallocator::handleTargetDestroyed(
    const MissionContext& context,
    const std::string& targetId) {
    const int tid = targetTid(context, targetId);
    if (tid < 0) {
        return AlgorithmResult<std::vector<ReconTask>>::fail(
            "Target is not represented in PPO request: " + targetId);
    }
    const std::string eventJson =
        "{\"event_type\":\"TARGET_DESTROYED\",\"target_id\":" + std::to_string(tid) + "}";
    return runEvent(context, eventJson);
}

AlgorithmResult<std::vector<ReconTask>>
PpoBridgeReallocator::runEvent(
    const MissionContext& context,
    const std::string& eventJson,
    const std::string& removedPlatform) const {
    const auto requestPath = makeTempPath("ppo_realloc", "_request.json");
    const auto outputPath = makeTempPath("ppo_realloc", "_result.json");
    const auto tasksPath = makeTempPath("ppo_realloc", "_tasks.tsv");

    try {
        {
            std::ofstream out(requestPath);
            if (!out) {
                throw std::runtime_error("failed to create PPO request: " + requestPath.string());
            }
            out << buildRequestJson(options_, context, eventJson, outputPath, tasksPath);
        }

        const std::string command =
            shellQuote(options_.python_executable) + " " +
            shellQuote(options_.bridge_script) + " --request-file " +
            shellQuote(requestPath.string());
        const int rc = std::system(command.c_str());
        if (rc != 0) {
            throw std::runtime_error("PPO bridge failed with code " + std::to_string(rc));
        }

        auto tasks = readPpoTasksTsv(tasksPath, context, removedPlatform);
        std::filesystem::remove(requestPath);
        std::filesystem::remove(outputPath);
        std::filesystem::remove(tasksPath);
        if (tasks.empty()) {
            return AlgorithmResult<std::vector<ReconTask>>::fail("PPO bridge returned no recon tasks");
        }
        return AlgorithmResult<std::vector<ReconTask>>::ok(tasks);
    } catch (const std::exception& exc) {
        std::filesystem::remove(requestPath);
        std::filesystem::remove(outputPath);
        std::filesystem::remove(tasksPath);
        return AlgorithmResult<std::vector<ReconTask>>::fail(exc.what());
    }
}

}  // namespace brain_cpp
