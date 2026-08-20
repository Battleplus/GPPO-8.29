#include "PerchPositionSelectorClient.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace perch_cpp {
namespace {

std::string JsonEscape(const std::string& value) {
    std::ostringstream out;
    for (const char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << ch; break;
        }
    }
    return out.str();
}

std::string ShellQuote(const std::string& value) {
#ifdef _WIN32
    std::string escaped;
    for (const char ch : value) {
        escaped += ch == '"' ? "\\\"" : std::string(1, ch);
    }
    return "\"" + escaped + "\"";
#else
    std::string escaped = "'";
    for (const char ch : value) {
        escaped += ch == '\'' ? "'\\''" : std::string(1, ch);
    }
    return escaped + "'";
#endif
}

template <typename T>
void WriteNumericMap(
    std::ostringstream& out,
    const std::map<std::string, T>& values) {
    bool first = true;
    for (const auto& item : values) {
        if (!first) out << ',';
        first = false;
        out << '"' << JsonEscape(item.first) << "\":" << item.second;
    }
}

std::string BuildRequestJson(
    const Situation& situation,
    const std::vector<StrikeTask>& tasks,
    const ClientOptions& options) {
    if (tasks.empty()) {
        throw std::invalid_argument("SelectAttackPositions requires at least one task");
    }
    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n\"mission_id\":\"" << JsonEscape(situation.mission_id) << "\",";
    out << "\n\"world_state\":{";
    out << "\"meters_per_unit\":100.0,\"map_size_km\":300.0,"
        << "\"map_size_units\":3000.0,";
    out << "\"staging_position\":[" << situation.staging_x_km << ','
        << situation.staging_y_km << "],\"weather\":{";
    WriteNumericMap(out, situation.weather);
    out << "},\"terrain\":{";
    WriteNumericMap(out, situation.terrain);
    out << "},\"targets\":[";
    for (std::size_t index = 0; index < situation.targets.size(); ++index) {
        if (index) out << ',';
        const auto& target = situation.targets[index];
        out << "{\"tid\":\"" << JsonEscape(target.tid)
            << "\",\"type\":\"" << JsonEscape(target.type)
            << "\",\"pos\":[" << target.x_km << ',' << target.y_km << ']'
            << ",\"value\":" << target.value
            << ",\"threat\":" << target.threat
            << ",\"confirmed\":" << (target.confirmed ? "true" : "false")
            << ",\"alive\":" << (target.alive ? "true" : "false") << '}';
    }
    out << "]},\n\"agents\":[";
    for (std::size_t index = 0; index < situation.agents.size(); ++index) {
        if (index) out << ',';
        const auto& agent = situation.agents[index];
        out << "{\"pid\":\"" << JsonEscape(agent.pid)
            << "\",\"type\":\"" << JsonEscape(agent.type)
            << "\",\"position\":[" << agent.x_km << ',' << agent.y_km << ']'
            << ",\"altitude_km\":" << agent.altitude_km
            << ",\"lost\":" << (agent.lost ? "true" : "false")
            << ",\"sensors\":[";
        for (std::size_t sensorIndex = 0; sensorIndex < agent.sensors.size(); ++sensorIndex) {
            if (sensorIndex) out << ',';
            out << '"' << JsonEscape(agent.sensors[sensorIndex]) << '"';
        }
        out << "],\"munitions\":{";
        WriteNumericMap(out, agent.munitions);
        out << "}}";
    }
    out << "],\n\"tasks\":[";
    for (std::size_t index = 0; index < tasks.size(); ++index) {
        if (index) out << ',';
        const auto& task = tasks[index];
        out << "{\"platform\":\"" << JsonEscape(task.platform)
            << "\",\"target\":\"" << JsonEscape(task.target)
            << "\",\"munition\":\"" << JsonEscape(task.munition)
            << "\",\"qty\":" << task.qty
            << ",\"role\":\"" << JsonEscape(task.role)
            << "\",\"aoi\":\"" << JsonEscape(task.aoi)
            << "\",\"assigned_munitions\":{";
        WriteNumericMap(out, task.assigned_munitions);
        out << "}}";
    }
    out << "],\n\"options\":{";
    out << "\"attack_region_mode\":\"" << JsonEscape(options.attack_region_mode)
        << "\",\"preference\":\"" << JsonEscape(options.preference)
        << "\",\"terrain_mode\":\"" << JsonEscape(options.terrain_mode)
        << "\",\"top_k\":" << options.top_k
        << ",\"use_pymoo\":" << (options.use_pymoo ? "true" : "false")
        << ",\"attack_region_strict\":"
        << (options.attack_region_strict ? "true" : "false") << "}\n}\n";
    return out.str();
}

std::filesystem::path MakeTempPath(const std::string& suffix) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path()
        / ("perch_position_" + std::to_string(stamp) + suffix);
}

std::vector<std::string> SplitTabs(const std::string& line) {
    std::vector<std::string> result;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, '\t')) {
        result.push_back(field);
    }
    if (!line.empty() && line.back() == '\t') {
        result.emplace_back();
    }
    return result;
}

std::vector<SelectedPosition> ReadPositions(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open Perch output: " + path.string());
    }
    std::vector<SelectedPosition> positions;
    std::string line;
    bool header = true;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        if (header) {
            header = false;
            continue;
        }
        const auto fields = SplitTabs(line);
        if (fields.size() != 17) {
            throw std::runtime_error("invalid Perch TSV row with "
                + std::to_string(fields.size()) + " fields");
        }
        SelectedPosition position;
        position.pos_id = fields[0];
        position.x = std::stod(fields[1]);
        position.y = std::stod(fields[2]);
        position.z = std::stod(fields[3]);
        position.kind = fields[4];
        position.metadata = {
            {"coordinate_frame", fields[5]},
            {"platform_id", fields[6]},
            {"target_id", fields[7]},
            {"munition", fields[8]},
            {"rank", fields[9]},
            {"source", fields[10]},
            {"target_range_km", fields[11]},
            {"agl_m", fields[12]},
            {"g_violation", fields[13]},
            {"optimiser", fields[14]},
            {"situation", fields[15]},
            {"knowledge_sources", fields[16]},
        };
        positions.push_back(std::move(position));
    }
    return positions;
}

}  // namespace

std::vector<SelectedPosition> SelectAttackPositions(
    const Situation& situation,
    const std::vector<StrikeTask>& tasks,
    const ClientOptions& options) {
    const auto requestPath = MakeTempPath("_request.json");
    const auto outputPath = MakeTempPath("_positions.tsv");
    const auto auditPath = MakeTempPath("_audit.json");
    try {
        {
            std::ofstream output(requestPath);
            if (!output) {
                throw std::runtime_error("failed to create Perch request");
            }
            output << BuildRequestJson(situation, tasks, options);
        }
        const std::string command =
            ShellQuote(options.python_executable) + " "
            + ShellQuote(options.bridge_script) + " --input "
            + ShellQuote(requestPath.string()) + " --output "
            + ShellQuote(outputPath.string()) + " --audit-output "
            + ShellQuote(auditPath.string());
        const int returnCode = std::system(command.c_str());
        if (returnCode != 0) {
            throw std::runtime_error(
                "Perch bridge failed with code " + std::to_string(returnCode));
        }
        auto positions = ReadPositions(outputPath);
        if (positions.empty()) {
            throw std::runtime_error("Perch bridge returned no positions");
        }
        std::filesystem::remove(requestPath);
        std::filesystem::remove(outputPath);
        std::filesystem::remove(auditPath);
        return positions;
    } catch (...) {
        std::filesystem::remove(requestPath);
        std::filesystem::remove(outputPath);
        std::filesystem::remove(auditPath);
        throw;
    }
}

}  // namespace perch_cpp
