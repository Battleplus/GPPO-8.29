#pragma once

#include "brain_cpp/environment_runtime.hpp"
#include "brain_cpp/mission_context.hpp"

#include <string>

namespace brain_cpp {

class IsaacPythonEnvironment : public IEnvironmentRuntime {
public:
    IsaacPythonEnvironment(
        std::string pythonExecutable = "",
        std::string helperScript = "brain_cpp/tools/isaac_snapshot.py",
        bool headless = true);

    AlgorithmResult<EnvironmentSnapshot>
    initialize(const MissionContext& context) override;

    AlgorithmResult<EnvironmentSnapshot>
    reset(const MissionContext& context) override;

    AlgorithmResult<EnvironmentSnapshot>
    step(const MissionContext& context, double dt) override;

private:
    std::string pythonExecutable_;
    std::string helperScript_;
    bool headless_ = true;

    AlgorithmResult<EnvironmentSnapshot>
    runHelper(const MissionContext& context, const std::string& command, double dt = 0.0);
};

}  // namespace brain_cpp
