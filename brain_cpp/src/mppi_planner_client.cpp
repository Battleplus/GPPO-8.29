#include "brain_cpp/mppi_planner_client.hpp"

#include "ql_path_planner.h"

#include <cctype>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace brain_cpp::mppi_client {
namespace {

struct JsonValue {
    enum class Type { Null, Boolean, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<JsonValue> array;
    std::map<std::string, JsonValue> object;
};

class JsonParser {
public:
    explicit JsonParser(const std::string& input) : input_(input) {}

    JsonValue parse() {
        JsonValue value = parseValue();
        skipSpace();
        if (position_ != input_.size()) fail("trailing content");
        return value;
    }

private:
    const std::string& input_;
    std::size_t position_ = 0;

    [[noreturn]] void fail(const std::string& reason) const {
        throw std::runtime_error(
            "Invalid MPPI JSON at byte " + std::to_string(position_) + ": " + reason);
    }

    void skipSpace() {
        while (position_ < input_.size()
               && std::isspace(static_cast<unsigned char>(input_[position_]))) {
            ++position_;
        }
    }

    bool consume(char expected) {
        skipSpace();
        if (position_ < input_.size() && input_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    JsonValue parseValue() {
        skipSpace();
        if (position_ >= input_.size()) fail("unexpected end of input");
        const char current = input_[position_];
        if (current == '{') return parseObject();
        if (current == '[') return parseArray();
        if (current == '"') {
            JsonValue value;
            value.type = JsonValue::Type::String;
            value.string = parseString();
            return value;
        }
        if (current == 't') return parseLiteral("true", JsonValue::Type::Boolean, true);
        if (current == 'f') return parseLiteral("false", JsonValue::Type::Boolean, false);
        if (current == 'n') return parseLiteral("null", JsonValue::Type::Null, false);
        return parseNumber();
    }

    JsonValue parseLiteral(const std::string& literal, JsonValue::Type type, bool boolean) {
        if (input_.compare(position_, literal.size(), literal) != 0) fail("invalid literal");
        position_ += literal.size();
        JsonValue value;
        value.type = type;
        value.boolean = boolean;
        return value;
    }

    JsonValue parseObject() {
        JsonValue value;
        value.type = JsonValue::Type::Object;
        ++position_;
        if (consume('}')) return value;
        while (true) {
            skipSpace();
            if (position_ >= input_.size() || input_[position_] != '"') {
                fail("expected object key");
            }
            std::string key = parseString();
            if (!consume(':')) fail("expected ':'");
            value.object.emplace(std::move(key), parseValue());
            if (consume('}')) break;
            if (!consume(',')) fail("expected ',' or '}'");
        }
        return value;
    }

    JsonValue parseArray() {
        JsonValue value;
        value.type = JsonValue::Type::Array;
        ++position_;
        if (consume(']')) return value;
        while (true) {
            value.array.push_back(parseValue());
            if (consume(']')) break;
            if (!consume(',')) fail("expected ',' or ']'");
        }
        return value;
    }

    static int hexDigit(char value) {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        return -1;
    }

    std::string parseString() {
        if (input_[position_++] != '"') fail("expected string");
        std::string result;
        while (position_ < input_.size()) {
            const char current = input_[position_++];
            if (current == '"') return result;
            if (current != '\\') {
                result.push_back(current);
                continue;
            }
            if (position_ >= input_.size()) fail("unfinished escape");
            const char escaped = input_[position_++];
            switch (escaped) {
            case '"': result.push_back('"'); break;
            case '\\': result.push_back('\\'); break;
            case '/': result.push_back('/'); break;
            case 'b': result.push_back('\b'); break;
            case 'f': result.push_back('\f'); break;
            case 'n': result.push_back('\n'); break;
            case 'r': result.push_back('\r'); break;
            case 't': result.push_back('\t'); break;
            case 'u': {
                if (position_ + 4 > input_.size()) fail("unfinished unicode escape");
                unsigned code = 0;
                for (int i = 0; i < 4; ++i) {
                    const int digit = hexDigit(input_[position_++]);
                    if (digit < 0) fail("invalid unicode escape");
                    code = code * 16U + static_cast<unsigned>(digit);
                }
                if (code <= 0x7fU) result.push_back(static_cast<char>(code));
                else if (code <= 0x7ffU) {
                    result.push_back(static_cast<char>(0xc0U | (code >> 6U)));
                    result.push_back(static_cast<char>(0x80U | (code & 0x3fU)));
                } else {
                    result.push_back(static_cast<char>(0xe0U | (code >> 12U)));
                    result.push_back(static_cast<char>(0x80U | ((code >> 6U) & 0x3fU)));
                    result.push_back(static_cast<char>(0x80U | (code & 0x3fU)));
                }
                break;
            }
            default: fail("invalid escape");
            }
        }
        fail("unterminated string");
    }

    JsonValue parseNumber() {
        skipSpace();
        const std::size_t begin = position_;
        if (position_ < input_.size() && input_[position_] == '-') ++position_;
        while (position_ < input_.size()
               && std::isdigit(static_cast<unsigned char>(input_[position_]))) ++position_;
        if (position_ < input_.size() && input_[position_] == '.') {
            ++position_;
            while (position_ < input_.size()
                   && std::isdigit(static_cast<unsigned char>(input_[position_]))) ++position_;
        }
        if (position_ < input_.size()
            && (input_[position_] == 'e' || input_[position_] == 'E')) {
            ++position_;
            if (position_ < input_.size()
                && (input_[position_] == '+' || input_[position_] == '-')) ++position_;
            while (position_ < input_.size()
                   && std::isdigit(static_cast<unsigned char>(input_[position_]))) ++position_;
        }
        if (begin == position_) fail("expected value");
        JsonValue value;
        value.type = JsonValue::Type::Number;
        try {
            value.number = std::stod(input_.substr(begin, position_ - begin));
        } catch (const std::exception&) {
            fail("invalid number");
        }
        return value;
    }
};

std::string jsonEscape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char ch : value) {
        switch (ch) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (ch < 0x20U) {
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<int>(ch) << std::dec;
            } else {
                output << static_cast<char>(ch);
            }
        }
    }
    return output.str();
}

const JsonValue& required(const JsonValue& object, const std::string& key, JsonValue::Type type) {
    if (object.type != JsonValue::Type::Object) throw std::runtime_error("MPPI response is not an object");
    const auto found = object.object.find(key);
    if (found == object.object.end() || found->second.type != type) {
        throw std::runtime_error("MPPI response field has wrong type or is missing: " + key);
    }
    return found->second;
}

Point3 point(const JsonValue& value) {
    if (value.type != JsonValue::Type::Array || value.array.size() != 3) {
        throw std::runtime_error("MPPI waypoint must contain three numbers");
    }
    Point3 result{};
    for (std::size_t i = 0; i < result.size(); ++i) {
        if (value.array[i].type != JsonValue::Type::Number
            || !std::isfinite(value.array[i].number)) {
            throw std::runtime_error("MPPI waypoint contains a non-finite value");
        }
        result[i] = value.array[i].number;
    }
    return result;
}

std::vector<Point3> path(const JsonValue& value) {
    if (value.type != JsonValue::Type::Array) throw std::runtime_error("MPPI path is not an array");
    std::vector<Point3> result;
    result.reserve(value.array.size());
    for (const auto& waypoint : value.array) result.push_back(point(waypoint));
    return result;
}

std::string scalarText(const JsonValue& value) {
    std::ostringstream output;
    output << std::setprecision(15);
    if (value.type == JsonValue::Type::String) return value.string;
    if (value.type == JsonValue::Type::Number) output << value.number;
    else if (value.type == JsonValue::Type::Boolean) output << (value.boolean ? "true" : "false");
    else if (value.type == JsonValue::Type::Null) output << "null";
    else return "[structured]";
    return output.str();
}

std::string requestJson(const PlanRequest& request) {
    if (request.team_count <= 0) throw std::invalid_argument("MPPI team_count must be positive");
    if (request.planner_config.num_samples <= 0
        || request.planner_config.num_iterations <= 0
        || request.planner_config.horizon <= 0) {
        throw std::invalid_argument("MPPI planner configuration values must be positive");
    }
    std::ostringstream output;
    output << std::setprecision(17)
           << "{\"team_count\":" << request.team_count
           << ",\"start\":[" << request.start[0] << ',' << request.start[1] << ',' << request.start[2] << ']'
           << ",\"goal\":[" << request.goal[0] << ',' << request.goal[1] << ',' << request.goal[2] << ']'
           << ",\"formation\":\"" << jsonEscape(request.formation) << "\""
           << ",\"spacing\":" << request.spacing
           << ",\"map_size_units\":" << request.map_size_units
           << ",\"meters_per_unit\":" << request.meters_per_unit
           << ",\"terrain_vertical_exaggeration\":" << request.terrain_vertical_exaggeration
           << ",\"verbose\":" << (request.verbose ? "true" : "false")
           << ",\"planner_config\":{\"map_size_units\":" << request.map_size_units
           << ",\"map_origin\":[" << -request.map_size_units * 0.5 << ',' << -request.map_size_units * 0.5 << ']'
           << ",\"num_samples\":" << request.planner_config.num_samples
           << ",\"num_iterations\":" << request.planner_config.num_iterations
           << ",\"horizon\":" << request.planner_config.horizon << '}';
    if (!request.member_assignments.empty()) {
        output << ",\"member_assignments\":{";
        bool first = true;
        for (const auto& assignment : request.member_assignments) {
            if (!first) output << ',';
            first = false;
            output << '"' << jsonEscape(assignment.first) << "\":" << assignment.second;
        }
        output << '}';
    }
    output << '}';
    return output.str();
}

PlanResult resultFromJson(const std::string& json) {
    const JsonValue root = JsonParser(json).parse();
    PlanResult result;
    result.team_count = static_cast<int>(required(root, "team_count", JsonValue::Type::Number).number);
    result.formation_type = required(root, "formation_type", JsonValue::Type::String).string;
    result.success = required(root, "success", JsonValue::Type::Boolean).boolean;
    result.center_path = path(required(root, "center_path", JsonValue::Type::Array));
    for (const auto& rawPath : required(root, "team_paths", JsonValue::Type::Array).array) {
        result.team_paths.push_back(path(rawPath));
    }
    for (const auto& role : required(root, "formation_roles", JsonValue::Type::Array).array) {
        if (role.type != JsonValue::Type::String) throw std::runtime_error("MPPI role must be a string");
        result.formation_roles.push_back(role.string);
    }
    const auto& stats = required(root, "stats", JsonValue::Type::Object);
    for (const auto& item : stats.object) result.planner_stats[item.first] = scalarText(item.second);
    return result;
}

}  // namespace

class PlannerClient::Impl {
public:
    explicit Impl(const std::string& projectRoot) : planner(projectRoot) {}
    ql::PathPlanner planner;
};

PlannerClient::PlannerClient(const std::string& projectRoot)
    : impl_(std::make_unique<Impl>(projectRoot)) {}

PlannerClient::~PlannerClient() = default;

PlanResult PlannerClient::plan(const PlanRequest& request) const {
    return resultFromJson(impl_->planner.plan(requestJson(request)));
}

}  // namespace brain_cpp::mppi_client
