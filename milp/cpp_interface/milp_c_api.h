#ifndef MILP_CPP_INTERFACE_MILP_C_API_H_
#define MILP_CPP_INTERFACE_MILP_C_API_H_

#ifdef __cplusplus
extern "C" {
#endif

#if defined(MILP_STATIC)
#  define MILP_API
#elif defined(_WIN32)
#  if defined(MILP_CPP_INTERFACE_BUILD)
#    define MILP_API __declspec(dllexport)
#  else
#    define MILP_API __declspec(dllimport)
#  endif
#else
#  define MILP_API __attribute__((visibility("default")))
#endif

typedef struct MilpContext* MilpHandle;

enum MilpStatusCode {
    MILP_STATUS_OK = 0,
    MILP_STATUS_INVALID_ARGUMENT = 1,
    MILP_STATUS_PYTHON_ERROR = 2,
    MILP_STATUS_ALLOCATION_ERROR = 3
};

/*
 * Create a reusable MILP bridge.
 *
 * milp_dir must point to the directory that contains task_interface.py,
 * multi_aoi_interface.py, and cpp_bridge.py.
 *
 * The returned handle owns a Python bridge object. Destroy it with
 * milp_destroy(). If this function fails it returns NULL and, when
 * error_message is not NULL, stores a malloc-allocated error string that must
 * be released with milp_free_string().
 */
MILP_API MilpHandle milp_create(
    const char* milp_dir,
    const char* solver,
    double time_limit_s,
    int verbose,
    char** error_message);

MILP_API void milp_destroy(MilpHandle handle);

/*
 * Run the single-AOI allocation mode.
 *
 * input_json uses the same schema as milp/templates/input_template.json.
 * output_json receives a malloc-allocated JSON string compatible with
 * milp/templates/output_template.json. Release it with milp_free_string().
 */
MILP_API int milp_single_aoi_solve_json(
    MilpHandle handle,
    const char* input_json,
    char** output_json,
    char** error_message);

/*
 * Run the single-AOI allocation mode from a UTF-8 JSON file path.
 *
 * input_path points to a JSON file that uses the same schema as
 * milp/templates/input_template.json. output_json is allocated by this library
 * and must be released with milp_free_string().
 */
MILP_API int milp_single_aoi_solve_file(
    MilpHandle handle,
    const char* input_path,
    char** output_json,
    char** error_message);

/*
 * Run one multi-AOI allocation step.
 *
 * input_json uses the multi-AOI schema accepted by MultiAOITaskAllocator.run().
 * output_json receives a malloc-allocated JSON string with status,
 * aoi_route_state, and current_aoi_plan fields.
 */
MILP_API int milp_multi_aoi_run_json(
    MilpHandle handle,
    const char* input_json,
    char** output_json,
    char** error_message);

/*
 * Run one multi-AOI allocation step from a UTF-8 JSON file path.
 */
MILP_API int milp_multi_aoi_run_file(
    MilpHandle handle,
    const char* input_path,
    char** output_json,
    char** error_message);

MILP_API void milp_free_string(char* value);

MILP_API const char* milp_status_message(int status_code);

MILP_API const char* milp_cpp_interface_version(void);

#ifdef __cplusplus
}
#endif

#endif  // MILP_CPP_INTERFACE_MILP_C_API_H_
