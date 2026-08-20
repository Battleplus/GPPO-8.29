#include "ql_path_planner.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstring>
#include <mutex>

namespace {
std::mutex g_mutex;
std::string g_error;
PyObject* g_plan_function = nullptr;

void set_python_error() {
    PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_NormalizeException(&type, &value, &traceback);
    PyObject* text = value ? PyObject_Str(value) : nullptr;
    const char* utf8 = text ? PyUnicode_AsUTF8(text) : nullptr;
    g_error = utf8 ? utf8 : "unknown Python error";
    Py_XDECREF(text);
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
}
}

int ql_planner_initialize(const char* project_root) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_error.clear();
    if (!project_root || !*project_root) {
        g_error = "project_root must not be empty";
        return -1;
    }
    const bool initialized_here = !Py_IsInitialized();
    PyGILState_STATE gil{};
    if (initialized_here) {
        Py_Initialize();
    } else {
        gil = PyGILState_Ensure();
    }

    PyObject* path = PySys_GetObject("path");  // borrowed reference
    PyObject* root = PyUnicode_DecodeFSDefault(project_root);
    if (!path || !root || PyList_Insert(path, 0, root) != 0) {
        Py_XDECREF(root);
        set_python_error();
        if (initialized_here) PyEval_SaveThread(); else PyGILState_Release(gil);
        return -1;
    }
    Py_DECREF(root);

    Py_XDECREF(g_plan_function);
    g_plan_function = nullptr;
    PyObject* module = PyImport_ImportModule("ql.scripts.cpp_bridge");
    if (!module) {
        set_python_error();
        if (initialized_here) PyEval_SaveThread(); else PyGILState_Release(gil);
        return -1;
    }
    g_plan_function = PyObject_GetAttrString(module, "plan_json");
    Py_DECREF(module);
    if (!g_plan_function || !PyCallable_Check(g_plan_function)) {
        g_error = "ql.scripts.cpp_bridge.plan_json is not callable";
        Py_CLEAR(g_plan_function);
        if (initialized_here) PyEval_SaveThread(); else PyGILState_Release(gil);
        return -1;
    }
    if (initialized_here) PyEval_SaveThread(); else PyGILState_Release(gil);
    return 0;
}

char* ql_plan_json(const char* request_json) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_error.clear();
    if (!g_plan_function) {
        g_error = "planner is not initialized";
        return nullptr;
    }
    if (!request_json) {
        g_error = "request_json must not be null";
        return nullptr;
    }
    PyGILState_STATE gil = PyGILState_Ensure();
    PyObject* argument = PyUnicode_FromString(request_json);
    PyObject* result = argument
        ? PyObject_CallFunctionObjArgs(g_plan_function, argument, nullptr)
        : nullptr;
    Py_XDECREF(argument);
    if (!result) {
        set_python_error();
        PyGILState_Release(gil);
        return nullptr;
    }
    const char* utf8 = PyUnicode_AsUTF8(result);
    if (!utf8) {
        Py_DECREF(result);
        set_python_error();
        PyGILState_Release(gil);
        return nullptr;
    }
    const std::size_t size = std::strlen(utf8) + 1;
    char* output = new char[size];
    std::memcpy(output, utf8, size);
    Py_DECREF(result);
    PyGILState_Release(gil);
    return output;
}

const char* ql_planner_last_error() { return g_error.c_str(); }
void ql_planner_free(char* value) { delete[] value; }

void ql_planner_shutdown() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!Py_IsInitialized()) return;
    PyGILState_STATE gil = PyGILState_Ensure();
    Py_CLEAR(g_plan_function);
    PyGILState_Release(gil);
    // Do not finalize CPython: the host process may own or still use it.
}
