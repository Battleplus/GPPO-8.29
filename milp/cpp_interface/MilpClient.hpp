#ifndef MILP_CPP_INTERFACE_MILP_CLIENT_HPP_
#define MILP_CPP_INTERFACE_MILP_CLIENT_HPP_

#include "milp_c_api.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace milp {

class MilpException : public std::runtime_error {
public:
    MilpException(int status_code, const std::string& message)
        : std::runtime_error(message), status_code_(status_code) {}

    int status_code() const noexcept {
        return status_code_;
    }

private:
    int status_code_;
};

class JsonMilpClient {
public:
    JsonMilpClient(
        const std::string& milp_dir,
        const std::string& solver = "cbc",
        double time_limit_s = 3.0,
        int verbose = 0) {
        char* error = nullptr;
        handle_ = milp_create(
            milp_dir.c_str(),
            solver.c_str(),
            time_limit_s,
            verbose,
            &error);
        if (handle_ == nullptr) {
            std::string message = takeString(error);
            if (message.empty()) {
                message = "milp_create failed";
            }
            throw MilpException(MILP_STATUS_PYTHON_ERROR, message);
        }
    }

    JsonMilpClient(const JsonMilpClient&) = delete;
    JsonMilpClient& operator=(const JsonMilpClient&) = delete;

    JsonMilpClient(JsonMilpClient&& other) noexcept
        : handle_(std::exchange(other.handle_, nullptr)) {}

    JsonMilpClient& operator=(JsonMilpClient&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    virtual ~JsonMilpClient() {
        reset();
    }

protected:
    std::string callSingleAoi(const std::string& input_json) {
        return callJson(milp_single_aoi_solve_json, input_json);
    }

    std::string callSingleAoiFile(const std::string& input_path) {
        return callJson(milp_single_aoi_solve_file, input_path);
    }

    std::string callMultiAoi(const std::string& input_json) {
        return callJson(milp_multi_aoi_run_json, input_json);
    }

    std::string callMultiAoiFile(const std::string& input_path) {
        return callJson(milp_multi_aoi_run_file, input_path);
    }

private:
    using JsonCall = int (*)(MilpHandle, const char*, char**, char**);

    std::string callJson(JsonCall call, const std::string& input_json) {
        char* output = nullptr;
        char* error = nullptr;
        int status = call(handle_, input_json.c_str(), &output, &error);
        if (status != MILP_STATUS_OK) {
            std::string message = takeString(error);
            if (message.empty()) {
                message = milp_status_message(status);
            }
            milp_free_string(output);
            throw MilpException(status, message);
        }
        return takeString(output);
    }

    static std::string takeString(char* value) {
        if (value == nullptr) {
            return {};
        }
        std::string result(value);
        milp_free_string(value);
        return result;
    }

    void reset() noexcept {
        if (handle_ != nullptr) {
            milp_destroy(handle_);
            handle_ = nullptr;
        }
    }

    MilpHandle handle_ = nullptr;
};

class SingleAoiMilpClient final : public JsonMilpClient {
public:
    using JsonMilpClient::JsonMilpClient;

    std::string solveJson(const std::string& input_json) {
        return callSingleAoi(input_json);
    }

    std::string solveFile(const std::string& input_path) {
        return callSingleAoiFile(input_path);
    }
};

class MultiAoiMilpClient final : public JsonMilpClient {
public:
    using JsonMilpClient::JsonMilpClient;

    std::string runJson(const std::string& input_json) {
        return callMultiAoi(input_json);
    }

    std::string runFile(const std::string& input_path) {
        return callMultiAoiFile(input_path);
    }
};

}  // namespace milp

#endif  // MILP_CPP_INTERFACE_MILP_CLIENT_HPP_
