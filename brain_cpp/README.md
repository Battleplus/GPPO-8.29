# brain_cpp

Independent C++ mission-brain implementation.

This directory intentionally does not replace `../brain`.  The Python brain
remains available while this module grows into the C++ orchestration layer.

Current scope:

- Mission context and domain models.
- Mission states and events.
- Isaac runtime initialization before planning.
- FSM-driven recon-to-action task flow.
- Runtime event handlers for target detection, platform loss, and attack
  completion.
- C++ algorithm interfaces for MILP, MPPI, PositionSelector, and PPO.
- `IsaacPythonEnvironment`, which initializes the repository's existing
  `brain.integration.IsaacAirCombatEnvironment` through an Isaac Python helper.
- `MilpTaskAllocator`, which calls `milp/cpp_interface` in-process for recon
  and strike allocation.
- `MppiRoutePlanner`, which calls the embedded `mppi/cpp` interface in-process
  for recon formations and independently matched strike routes.
- `PerchPositionSelector`, which calls `perch/cpp_interface` and the file-based
  Python bridge to run situation understanding, attack-region selection, and
  FREA position optimisation from the C++ mission FSM.

`SarSearchPatrolPlanner` owns the integration mapping from MILP sensor to the
search pattern passed into `search_planner`:

| Sensor | Search pattern |
| --- | --- |
| SAR | `sar_polygon` |
| EO / EOIR | `racetrack` |
| ESM | `figure_eight` |
| MMW | `sar_rounded` |

The `search_planner` module remains pattern-driven and does not depend on these
mission-level sensor semantics.

Build:

```bash
cmake -S brain_cpp -B build/brain_cpp
cmake --build build/brain_cpp
```

Run:

```bash
./build/brain_cpp/brain_cpp_demo --mission-id CPP_BRAIN --aoi A_3_4 \
  --isaac-python /home/isaac/isaacsim/python.sh
```

Export a sensor-assigned patrol as global waypoints:

```bash
./build/brain_cpp/brain_cpp_patrol_export \
  --aoi A_3_6 --cell c3 --sensor SAR --platform Blue_CH4_Recon \
  --output drl_env/outputs/brain_cpp_ppo_isaac/global_path.csv
```

Run those waypoints with PPO local avoidance in Isaac Sim:

```bash
/home/isaac/isaacsim/python.sh brain_cpp/tools/run_patrol_ppo_isaac.py \
  --headless \
  --global-plan drl_env/outputs/brain_cpp_ppo_isaac/global_path.csv \
  --policy drl_env/models/ppo_drone_best_v6.npz \
  --output-dir drl_env/outputs/brain_cpp_ppo_isaac
```

Run the C++ mission brain with real Perch/FREA position selection and offline
rule-based attack regions:

```bash
./build/brain_cpp/brain_cpp_demo \
  --environment none \
  --perch-region-mode demo \
  --perch-terrain-mode flat \
  --perch-top-k 1 \
  --perch-use-pymoo 0 \
  --patrol-planner none \
  --ppo-adapter none
```

Use `--perch-region-mode llm` to enable the RAG/LLM region recommender. Its
provider and API credentials use the existing `PERCH_LLM_*` and
`PERCH_OPENAI_*` environment variables. `--perch-use-pymoo 1` enables
R-NSGA-II instead of deterministic grid search.

MPPI defaults to 512 samples, 5 iterations, and a 50-step horizon. Override
these with `--mppi-samples`, `--mppi-iterations`, and `--mppi-horizon`.
Routes returned to `brain_cpp` use `mission_km`; conversion to the MPPI scene
coordinate frame is owned by `MppiRoutePlanner`.

Run the six-aircraft allocation and phased local-control audit:

```bash
./build/brain_cpp/brain_cpp_six_uav_mission_export \
  --output-dir brain_cpp/outputs/six_uav_milp_mppi_ppo_20260714

python brain_cpp/tools/execute_mppi_then_ppo.py \
  --plan brain_cpp/outputs/six_uav_milp_mppi_ppo_20260714/mission_plan.csv \
  --output-dir brain_cpp/outputs/six_uav_milp_mppi_ppo_20260714
```

The phase executor enforces `transit=mppi_follow` with zero PPO calls and
`search=ppo_local` with the reactive PPO policy. The current single-AOI MILP
formulation intentionally activates five of six available aircraft: one ESM
aircraft covers `c0`-`c4`, four EO/SAR aircraft cover `c1`-`c4`, and the sixth
aircraft remains reserve. The generated planning summary reports this
explicitly instead of manufacturing a non-MILP assignment.

The Isaac phase runner treats coverage paths as continuous patrol loops. A UAV
starts its next loop after reaching the final waypoint and only holds position
after the JSON control field `patrol_end` becomes `true`. By default the runner
polls the planning summary; use `--patrol-control PATH` when a runtime mission
state file owns that field. `--max-search-steps` bounds a recording run and does
not mean that patrol has completed.

```bash
/home/isaac/isaacsim/python.sh brain_cpp/tools/run_six_uav_phase_isaac.py \
  --headless \
  --plan brain_cpp/outputs/six_uav_milp_mppi_ppo_20260714/mission_plan.csv \
  --planning-summary brain_cpp/outputs/six_uav_milp_mppi_ppo_20260714/planning_summary.json \
  --output-dir brain_cpp/outputs/six_uav_milp_mppi_ppo_20260714/isaac_run
```

Configure MILP with `--milp-dir`, `--milp-solver`, and `--milp-time-limit`.
Pass `--milp-verbose 1` to print the complete MILP input and output JSON.
When `--aois` contains two or more areas, `MilpTaskAllocator` uses the
multi-AOI interface and the returned `aoi_route_state.aoi_sequence` defines
the area scan order.

Test:

```bash
ctest --test-dir build/brain_cpp --output-on-failure
```

`--mission-input`, `--aoi`, and `--aois` are forwarded to the Isaac helper,
which reuses the existing Python mission input normalization before creating
the air-combat scene.
# Parallel multi-platform execution

`brain_cpp_demo --execute-parallel` executes every unique platform route in one
synchronous control loop. Multiple MILP tasks assigned to the same aircraft are
joined into that aircraft's waypoint queue. This validates the C++ scheduling
and completion barrier using routes from the selected route planner. The
current position integrator remains a temporary control backend until the
persistent Isaac command bridge is available.
