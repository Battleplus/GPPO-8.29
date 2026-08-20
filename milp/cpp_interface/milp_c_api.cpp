#include "milp_c_api.h"

#include <Python.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <string>

struct MilpContext {
    PyObject* bridge = nullptr;
};

namespace {

std::mutex g_python_init_mutex;

char* duplicate_c_string(const std::string& value) {
    char* buffer = static_cast<char*>(std::malloc(value.size() + 1));
    if (buffer == nullptr) {
        return nullptr;
    }
    std::memcpy(buffer, value.c_str(), value.size() + 1);
    return buffer;
}

void set_c_string(char** target, const std::string& value) {
    if (target == nullptr) {
        return;
    }
    *target = duplicate_c_string(value);
}

void clear_c_string(char** target) {
    if (target != nullptr) {
        *target = nullptr;
    }
}

void ensure_python_initialized() {
    std::lock_guard<std::mutex> lock(g_python_init_mutex);
    if (!Py_IsInitialized()) {
        Py_InitializeEx(0);
    }
}

std::string object_to_utf8(PyObject* object) {
    if (object == nullptr) {
        return {};
    }
    PyObject* text_obj = PyObject_Str(object);
    if (text_obj == nullptr) {
        return {};
    }
    const char* text = PyUnicode_AsUTF8(text_obj);
    std::string result = text != nullptr ? text : "";
    Py_DECREF(text_obj);
    return result;
}

std::string fetch_python_error() {
    PyObject* exc_type = nullptr;
    PyObject* exc_value = nullptr;
    PyObject* exc_traceback = nullptr;
    PyErr_Fetch(&exc_type, &exc_value, &exc_traceback);
    PyErr_NormalizeException(&exc_type, &exc_value, &exc_traceback);

    std::string message;
    PyObject* traceback_module = PyImport_ImportModule("traceback");
    if (traceback_module != nullptr) {
        PyObject* type_arg = exc_type != nullptr ? exc_type : Py_None;
        PyObject* value_arg = exc_value != nullptr ? exc_value : Py_None;
        PyObject* traceback_arg = exc_traceback != nullptr ? exc_traceback : Py_None;
        PyObject* formatted = PyObject_CallMethod(
            traceback_module,
            "format_exception",
            "OOO",
            type_arg,
            value_arg,
            traceback_arg);
        if (formatted != nullptr) {
            PyObject* separator = PyUnicode_FromString("");
            PyObject* joined = separator != nullptr
                ? PyUnicode_Join(separator, formatted)
                : nullptr;
            if (joined != nullptr) {
                const char* text = PyUnicode_AsUTF8(joined);
                if (text != nullptr) {
                    message = text;
                }
                Py_DECREF(joined);
            }
            Py_XDECREF(separator);
            Py_DECREF(formatted);
        }
        Py_DECREF(traceback_module);
    } else {
        PyErr_Clear();
    }

    if (message.empty()) {
        message = object_to_utf8(exc_value);
        if (message.empty()) {
            message = object_to_utf8(exc_type);
        }
    }
    if (message.empty()) {
        message = "unknown Python error";
    }

    Py_XDECREF(exc_type);
    Py_XDECREF(exc_value);
    Py_XDECREF(exc_traceback);
    return message;
}

PyObject* decode_path_string(const char* path) {
    PyObject* decoded = PyUnicode_DecodeFSDefault(path);
    if (decoded != nullptr) {
        return decoded;
    }
    PyErr_Clear();

#if defined(_WIN32)
    decoded = PyUnicode_DecodeMBCS(path, static_cast<Py_ssize_t>(std::strlen(path)), nullptr);
    if (decoded != nullptr) {
        return decoded;
    }
    PyErr_Clear();
#endif

    return PyUnicode_FromString(path);
}

int insert_milp_dir_to_sys_path(const char* milp_dir, char** error_message) {
    PyObject* sys_path = PySys_GetObject("path");  // borrowed reference
    if (sys_path == nullptr || !PyList_Check(sys_path)) {
        set_c_string(error_message, "failed to access Python sys.path");
        return MILP_STATUS_PYTHON_ERROR;
    }

    PyObject* path = decode_path_string(milp_dir);
    if (path == nullptr) {
        set_c_string(error_message, fetch_python_error());
        return MILP_STATUS_PYTHON_ERROR;
    }

    int contains = PySequence_Contains(sys_path, path);
    if (contains < 0) {
        std::string error = fetch_python_error();
        Py_DECREF(path);
        set_c_string(error_message, error);
        return MILP_STATUS_PYTHON_ERROR;
    }

    if (contains == 0 && PyList_Insert(sys_path, 0, path) != 0) {
        std::string error = fetch_python_error();
        Py_DECREF(path);
        set_c_string(error_message, error);
        return MILP_STATUS_PYTHON_ERROR;
    }

    Py_DECREF(path);
    return MILP_STATUS_OK;
}

int call_bridge_json_method(
    MilpHandle handle,
    const char* method_name,
    const char* input_text,
    char** output_json,
    char** error_message,
    bool input_is_path = false) {
    clear_c_string(output_json);
    clear_c_string(error_message);

    if (handle == nullptr || handle->bridge == nullptr) {
        set_c_string(error_message, "MILP handle is null");
        return MILP_STATUS_INVALID_ARGUMENT;
    }
    if (input_text == nullptr) {
        set_c_string(error_message, input_is_path ? "input_path is null" : "input_json is null");
        return MILP_STATUS_INVALID_ARGUMENT;
    }
    if (output_json == nullptr) {
        set_c_string(error_message, "output_json is null");
        return MILP_STATUS_INVALID_ARGUMENT;
    }

    PyGILState_STATE gil_state = PyGILState_Ensure();

    PyObject* method = PyUnicode_FromString(method_name);
    PyObject* input = input_is_path
        ? decode_path_string(input_text)
        : PyUnicode_FromString(input_text);
    if (method == nullptr || input == nullptr) {
        std::string error = fetch_python_error();
        Py_XDECREF(method);
        Py_XDECREF(input);
        PyGILState_Release(gil_state);
        set_c_string(error_message, error);
        return MILP_STATUS_PYTHON_ERROR;
    }

    PyObject* result = PyObject_CallMethodObjArgs(handle->bridge, method, input, nullptr);
    Py_DECREF(method);
    Py_DECREF(input);

    if (result == nullptr) {
        std::string error = fetch_python_error();
        PyGILState_Release(gil_state);
        set_c_string(error_message, error);
        return MILP_STATUS_PYTHON_ERROR;
    }

    if (!PyUnicode_Check(result)) {
        Py_DECREF(result);
        PyGILState_Release(gil_state);
        set_c_string(error_message, "Python bridge returned a non-string value");
        return MILP_STATUS_PYTHON_ERROR;
    }

    const char* output_text = PyUnicode_AsUTF8(result);
    if (output_text == nullptr) {
        std::string error = fetch_python_error();
        Py_DECREF(result);
        PyGILState_Release(gil_state);
        set_c_string(error_message, error);
        return MILP_STATUS_PYTHON_ERROR;
    }

    *output_json = duplicate_c_string(output_text);
    Py_DECREF(result);
    PyGILState_Release(gil_state);

    if (*output_json == nullptr) {
        set_c_string(error_message, "failed to allocate output_json");
        return MILP_STATUS_ALLOCATION_ERROR;
    }
    return MILP_STATUS_OK;
}

}  // namespace

MilpHandle milp_create(
    const char* milp_dir,
    const char* solver,
    double time_limit_s,
    int verbose,
    char** error_message) {
    clear_c_string(error_message);

    if (milp_dir == nullptr || milp_dir[0] == '\0') {
        set_c_string(error_message, "milp_dir is null or empty");
        return nullptr;
    }
    if (time_limit_s <= 0.0) {
        set_c_string(error_message, "time_limit_s must be positive");
        return nullptr;
    }

    ensure_python_initialized();
    PyGILState_STATE gil_state = PyGILState_Ensure();

    int path_status = insert_milp_dir_to_sys_path(milp_dir, error_message);
    if (path_status != MILP_STATUS_OK) {
        PyGILState_Release(gil_state);
        return nullptr;
    }

    PyObject* module = PyImport_ImportModule("cpp_bridge");
    if (module == nullptr) {
        std::string error = fetch_python_error();
        PyGILState_Release(gil_state);
        set_c_string(error_message, error);
        return nullptr;
    }

    PyObject* factory = PyObject_GetAttrString(module, "create_bridge");
    Py_DECREF(module);
    if (factory == nullptr) {
        std::string error = fetch_python_error();
        PyGILState_Release(gil_state);
        set_c_string(error_message, error);
        return nullptr;
    }

    const char* solver_name = solver != nullptr && solver[0] != '\0' ? solver : "cbc";
    PyObject* bridge = PyObject_CallFunction(factory, "sdi", solver_name, time_limit_s, verbose);
    Py_DECREF(factory);
    if (bridge == nullptr) {
        std::string error = fetch_python_error();
        PyGILState_Release(gil_state);
        set_c_string(error_message, error);
        return nullptr;
    }

    MilpContext* context = new (std::nothrow) MilpContext();
    if (context == nullptr) {
        Py_DECREF(bridge);
        PyGILState_Release(gil_state);
        set_c_string(error_message, "failed to allocate MILP context");
        return nullptr;
    }
    context->bridge = bridge;

    PyGILState_Release(gil_state);
    return context;
}

void milp_destroy(MilpHandle handle) {
    if (handle == nullptr) {
        return;
    }
    if (Py_IsInitialized()) {
        PyGILState_STATE gil_state = PyGILState_Ensure();
        Py_XDECREF(handle->bridge);
        handle->bridge = nullptr;
        PyGILState_Release(gil_state);
    }
    delete handle;
}

int milp_single_aoi_solve_json(
    MilpHandle handle,
    const char* input_json,
    char** output_json,
    char** error_message) {
    return call_bridge_json_method(
        handle,
        "solve_single_aoi_json",
        input_json,
        output_json,
        error_message);
}

int milp_single_aoi_solve_file(
    MilpHandle handle,
    const char* input_path,
    char** output_json,
    char** error_message) {
    return call_bridge_json_method(
        handle,
        "solve_single_aoi_file",
        input_path,
        output_json,
        error_message,
        true);
}

int milp_multi_aoi_run_json(
    MilpHandle handle,
    const char* input_json,
    char** output_json,
    char** error_message) {
    return call_bridge_json_method(
        handle,
        "run_multi_aoi_json",
        input_json,
        output_json,
        error_message);
}

int milp_multi_aoi_run_file(
    MilpHandle handle,
    const char* input_path,
    char** output_json,
    char** error_message) {
    return call_bridge_json_method(
        handle,
        "run_multi_aoi_file",
        input_path,
        output_json,
        error_message,
        true);
}

void milp_free_string(char* value) {
    std::free(value);
}

const char* milp_status_message(int status_code) {
    switch (status_code) {
        case MILP_STATUS_OK:
            return "OK";
        case MILP_STATUS_INVALID_ARGUMENT:
            return "invalid argument";
        case MILP_STATUS_PYTHON_ERROR:
            return "Python error";
        case MILP_STATUS_ALLOCATION_ERROR:
            return "allocation error";
        default:
            return "unknown status";
    }
}

const char* milp_cpp_interface_version(void) {
    return "1.1.0";
}
