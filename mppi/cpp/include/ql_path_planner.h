#pragma once

#include <stdexcept>
#include <string>

#if defined(_WIN32)
#  if defined(QL_PATH_PLANNER_BUILD)
#    define QL_API __declspec(dllexport)
#  else
#    define QL_API __declspec(dllimport)
#  endif
#else
#  define QL_API
#endif

extern "C" {
QL_API int ql_planner_initialize(const char* project_root);
QL_API char* ql_plan_json(const char* request_json);
QL_API const char* ql_planner_last_error();
QL_API void ql_planner_free(char* value);
QL_API void ql_planner_shutdown();
}

namespace ql {
class PathPlanner {
public:
    explicit PathPlanner(const std::string& project_root) {
        if (ql_planner_initialize(project_root.c_str()) != 0) {
            throw std::runtime_error(ql_planner_last_error());
        }
    }

    std::string plan(const std::string& request_json) const {
        char* output = ql_plan_json(request_json.c_str());
        if (!output) {
            throw std::runtime_error(ql_planner_last_error());
        }
        std::string result(output);
        ql_planner_free(output);
        return result;
    }
};
}  // namespace ql
