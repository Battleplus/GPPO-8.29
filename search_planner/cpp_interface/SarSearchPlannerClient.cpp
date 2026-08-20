#include "SarSearchPlannerClient.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace sar_search_planner {
namespace {

std::string JsonEscape(const std::string& value) {
    std::ostringstream out;
    for (char ch : value) {
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
    for (char ch : value) {
        if (ch == '"') {
            escaped += "\\\"";
        } else {
            escaped += ch;
        }
    }
    return "\"" + escaped + "\"";
#else
    std::string escaped = "'";
    for (char ch : value) {
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

std::string BuildRequestJson(const std::vector<SearchTask>& tasks) {
    if (tasks.empty()) {
        throw std::invalid_argument("PlanSarSearchPath requires at least one task");
    }

    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n  \"tasks\": [\n";
    for (std::size_t i = 0; i < tasks.size(); ++i) {
        const SearchTask& task = tasks[i];
        out << "    {\n";
        out << "      \"platform_id\": \"" << JsonEscape(task.platform_id) << "\",\n";
        out << "      \"center_km\": [" << task.center_x_km << ", " << task.center_y_km << "],\n";
        out << "      \"width_km\": " << task.width_km << ",\n";
        out << "      \"height_km\": " << task.height_km << ",\n";
        out << "      \"pattern\": \"" << JsonEscape(task.pattern) << "\",\n";
        out << "      \"altitude_agl_m\": " << task.altitude_agl_m << "\n";
        out << "    }" << (i + 1 == tasks.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    return out.str();
}

std::vector<std::string> ParseCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    bool in_quotes = false;

    for (std::size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (in_quotes) {
            if (ch == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    current += '"';
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                current += ch;
            }
        } else {
            if (ch == '"') {
                in_quotes = true;
            } else if (ch == ',') {
                fields.push_back(current);
                current.clear();
            } else {
                current += ch;
            }
        }
    }
    fields.push_back(current);
    return fields;
}

std::vector<Waypoint> ReadWaypointsCsv(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open planner output: " + path.string());
    }

    std::vector<Waypoint> waypoints;
    std::string line;
    bool header = true;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (header) {
            header = false;
            continue;
        }

        std::vector<std::string> fields = ParseCsvLine(line);
        if (fields.size() != 8) {
            throw std::runtime_error("invalid planner CSV row: " + line);
        }

        Waypoint wp;
        wp.platform_id = fields[0];
        wp.point_index = std::stoi(fields[1]);
        wp.x = std::stod(fields[2]);
        wp.y = std::stod(fields[3]);
        wp.z = std::stod(fields[4]);
        wp.terrain_z = std::stod(fields[5]);
        wp.yaw_deg = std::stod(fields[6]);
        wp.total_km = std::stod(fields[7]);
        waypoints.push_back(wp);
    }
    return waypoints;
}

std::filesystem::path MakeTempPath(const std::string& suffix) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path()
        / ("sar_search_planner_" + std::to_string(stamp) + suffix);
}

}  // namespace

std::vector<Waypoint> PlanSarSearchPath(
    const std::vector<SearchTask>& tasks,
    const PlannerClientOptions& options
) {
    const std::filesystem::path input_path = MakeTempPath("_request.json");
    const std::filesystem::path output_path = MakeTempPath("_waypoints.csv");

    {
        std::ofstream out(input_path);
        if (!out) {
            throw std::runtime_error("failed to create planner input: " + input_path.string());
        }
        out << BuildRequestJson(tasks);
    }

    const std::string command =
        ShellQuote(options.python_executable) + " " +
        ShellQuote(options.bridge_script) + " --input " +
        ShellQuote(input_path.string()) + " --output " +
        ShellQuote(output_path.string()) + " --format csv";

    const int rc = std::system(command.c_str());
    if (rc != 0) {
        std::filesystem::remove(input_path);
        std::filesystem::remove(output_path);
        throw std::runtime_error("SAR planner bridge failed with code " + std::to_string(rc));
    }

    std::vector<Waypoint> waypoints = ReadWaypointsCsv(output_path);
    std::filesystem::remove(input_path);
    std::filesystem::remove(output_path);
    return waypoints;
}

}  // namespace sar_search_planner
