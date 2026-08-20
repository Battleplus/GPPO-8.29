#include "brain_cpp/isaac_python_environment.hpp"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace brain_cpp {
namespace {

std::string shellQuote(const std::string& value) {
    std::string quoted = "'";
    for (const char ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

std::vector<std::string> split(const std::string& value, char delimiter) {
    std::vector<std::string> parts;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, delimiter)) {
        parts.push_back(item);
    }
    return parts;
}

std::map<std::string, int> parseMunitions(const std::string& value) {
    std::map<std::string, int> result;
    for (const auto& item : split(value, ';')) {
        if (item.empty()) {
            continue;
        }
        const auto pos = item.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        result[item.substr(0, pos)] = std::stoi(item.substr(pos + 1));
    }
    return result;
}

std::map<std::string, double> parseWeather(const std::string& value) {
    std::map<std::string, double> result;
    for (const auto& item : split(value, ';')) {
        if (item.empty()) {
            continue;
        }
        const auto pos = item.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        result[item.substr(0, pos)] = std::stod(item.substr(pos + 1));
    }
    return result;
}

std::map<std::string, int> parseTerrain(const std::string& value) {
    std::map<std::string, int> result;
    for (const auto& item : split(value, ';')) {
        if (item.empty()) {
            continue;
        }
        const auto pos = item.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        result[item.substr(0, pos)] = std::stoi(item.substr(pos + 1));
    }
    return result;
}

std::vector<std::string> parseList(const std::string& value) {
    if (value.empty()) {
        return {};
    }
    return split(value, ';');
}

std::string joinAois(const std::vector<Aoi>& aois) {
    std::ostringstream out;
    for (std::size_t index = 0; index < aois.size(); ++index) {
        if (index != 0) {
            out << ",";
        }
        out << aois[index].id;
    }
    return out.str();
}

}  // namespace

IsaacPythonEnvironment::IsaacPythonEnvironment(
    std::string pythonExecutable,
    std::string helperScript,
    bool headless)
    : pythonExecutable_(std::move(pythonExecutable)),
      helperScript_(std::move(helperScript)),
      headless_(headless) {
    if (pythonExecutable_.empty()) {
        const char* env = std::getenv("ISAAC_PYTHON");
        pythonExecutable_ = env != nullptr ? std::string(env) : "/home/isaac/isaacsim/python.sh";
    }
}

AlgorithmResult<EnvironmentSnapshot>
IsaacPythonEnvironment::initialize(const MissionContext& context) {
    return runHelper(context, "initialize");
}

AlgorithmResult<EnvironmentSnapshot>
IsaacPythonEnvironment::reset(const MissionContext& context) {
    return runHelper(context, "reset");
}

AlgorithmResult<EnvironmentSnapshot>
IsaacPythonEnvironment::step(const MissionContext& context, double dt) {
    return runHelper(context, "step", dt);
}

AlgorithmResult<EnvironmentSnapshot>
IsaacPythonEnvironment::runHelper(
    const MissionContext& context,
    const std::string& command,
    double dt) {
    std::string shellCommand =
        shellQuote(pythonExecutable_) + " " + shellQuote(helperScript_)
        + " --command " + shellQuote(command)
        + " --dt " + shellQuote(std::to_string(dt));
    if (headless_) {
        shellCommand += " --headless";
    }
    if (!context.world.aois.empty()) {
        shellCommand += " --aois " + shellQuote(joinAois(context.world.aois));
    } else if (!context.world.aoi.id.empty()) {
        shellCommand += " --aoi " + shellQuote(context.world.aoi.id);
    }
    if (!context.world.mission_input_path.empty()) {
        shellCommand += " --mission-input " + shellQuote(context.world.mission_input_path);
    }

    FILE* pipe = popen(shellCommand.c_str(), "r");
    if (pipe == nullptr) {
        return AlgorithmResult<EnvironmentSnapshot>::fail(
            "Failed to run Isaac helper: " + shellCommand);
    }

    EnvironmentSnapshot snapshot;
    char buffer[4096];
    bool sawBegin = false;
    bool sawEnd = false;
    std::ostringstream rawOutput;
    try {
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            std::string line(buffer);
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
                line.pop_back();
            }
            rawOutput << line << "\n";
            const auto fields = split(line, '|');
            if (fields.empty()) {
                continue;
            }
            if (fields[0] == "snapshot" && fields.size() > 1 && fields[1] == "begin") {
                sawBegin = true;
            } else if (fields[0] == "snapshot" && fields.size() > 1 && fields[1] == "end") {
                sawEnd = true;
            } else if (fields[0] == "name" && fields.size() > 1) {
                snapshot.name = fields[1];
            } else if (fields[0] == "initialized" && fields.size() > 1) {
                snapshot.initialized = fields[1] == "1";
            } else if (fields[0] == "time" && fields.size() > 1) {
                snapshot.tactical_time_s = std::stod(fields[1]);
            } else if (fields[0] == "staging" && fields.size() > 2) {
                snapshot.staging_position = {std::stod(fields[1]), std::stod(fields[2])};
            } else if (fields[0] == "aoi" && fields.size() > 3) {
                Aoi aoi;
                aoi.id = fields[1];
                aoi.row = std::stoi(fields[2]);
                aoi.col = std::stoi(fields[3]);
                aoi.index = static_cast<int>(snapshot.aois.size());
                snapshot.aois.push_back(aoi);
            } else if (fields[0] == "agent" && fields.size() > 8) {
                AgentSpec agent;
                agent.pid = fields[1];
                agent.type = fields[2];
                agent.position = {std::stod(fields[3]), std::stod(fields[4])};
                agent.altitude_km = std::stod(fields[5]);
                agent.lost = fields[6] == "1";
                agent.sensors = parseList(fields[7]);
                agent.munitions = parseMunitions(fields[8]);
                snapshot.agents.push_back(agent);
            } else if (fields[0] == "target" && fields.size() > 8) {
                TargetInfo target;
                target.tid = fields[1];
                target.type = fields[2];
                target.pos = {std::stod(fields[3]), std::stod(fields[4])};
                target.value = std::stod(fields[5]);
                target.threat = std::stod(fields[6]);
                target.confirmed = fields[7] == "1";
                target.alive = fields[8] == "1";
                snapshot.targets.push_back(target);
            } else if (fields[0] == "contact" && fields.size() > 6) {
                SensorContact contact;
                contact.platform_id = fields[1];
                contact.target_id = fields[2];
                contact.sensor = fields[3];
                contact.channel = fields[4];
                contact.distance_km = std::stod(fields[5]);
                contact.priority = std::stoi(fields[6]);
                snapshot.sensor_contacts.push_back(contact);
            } else if (fields[0] == "weather" && fields.size() > 1) {
                snapshot.weather = parseWeather(fields[1]);
            } else if (fields[0] == "terrain" && fields.size() > 1) {
                snapshot.terrain = parseTerrain(fields[1]);
            }
        }
    } catch (const std::exception& exc) {
        pclose(pipe);
        return AlgorithmResult<EnvironmentSnapshot>::fail(
            std::string("Failed to parse Isaac helper output: ") + exc.what());
    }

    const int status = pclose(pipe);
    if (!sawBegin || !sawEnd || !snapshot.initialized) {
        if (status != 0) {
            return AlgorithmResult<EnvironmentSnapshot>::fail(
                "Isaac helper exited with status " + std::to_string(status)
                + ". Output:\n" + rawOutput.str());
        }
        return AlgorithmResult<EnvironmentSnapshot>::fail(
            "Isaac helper did not return a valid snapshot. Output:\n" + rawOutput.str());
    }
    if (snapshot.name.empty()) {
        snapshot.name = "IsaacAirCombatEnvironment";
    }
    return AlgorithmResult<EnvironmentSnapshot>::ok(snapshot);
}

}  // namespace brain_cpp
