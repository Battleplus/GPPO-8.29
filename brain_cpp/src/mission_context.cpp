#include "brain_cpp/mission_context.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace brain_cpp {

std::string timestampNow() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return out.str();
}

void MissionContext::recordEvent(const std::string& event, const std::string& detail) {
    history.push_back({
        timestampNow(),
        mission_id,
        toString(state),
        event,
        detail,
    });
}

bool MissionContext::hasPendingActionTasks() const {
    if (!pending_strike_targets.empty()) {
        return true;
    }
    for (const auto& target : world.targets) {
        if (target.alive
            && target.confirmed
            && engaged_targets.find(target.tid) == engaged_targets.end()) {
            return true;
        }
    }
    return false;
}

bool containsString(const std::vector<std::string>& values, const std::string& item) {
    return std::find(values.begin(), values.end(), item) != values.end();
}

void appendUnique(std::vector<std::string>& values, const std::string& item) {
    if (!item.empty() && !containsString(values, item)) {
        values.push_back(item);
    }
}

void removeValue(std::vector<std::string>& values, const std::string& item) {
    values.erase(
        std::remove(values.begin(), values.end(), item),
        values.end());
}

}  // namespace brain_cpp
