"""Start MissionBrain with Isaac air-combat environment + full execution loop.

Run::
    python brain/isaac_main.py [--headless] [--ppo-model PATH] [--drl-policy PATH]

Flow
----
1. IsaacAirCombatEnvironment initialises Isaac Sim and the air-combat scene.
2. brain.start() runs MILP recon allocation + MPPI route planning (once).
3. MissionExecutor drives the Isaac step loop:
   - Formation transit → DRL patrol → detect targets
   - PPO dynamic reallocation on UAV loss (no second MILP call)
   - After all targets confirmed: MILP strike allocation + MPPI strike routes (once)
   - Strike transit → mission complete
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from brain.adapters import (  # noqa: E402
    MILPTaskAllocator,
    MPPIFormationPlanner,
    PPOAllocationAdapter,
    PositionSelector,
)
from brain.core import MissionBrain, make_context  # noqa: E402
from brain.execution import MissionExecutor  # noqa: E402
from brain.integration import (  # noqa: E402
    IsaacAirCombatEnvironment,
    apply_task_area_overrides,
    load_mission_world,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def build_brain(
    *,
    headless: bool = False,
    scene_config: dict | None = None,
    mission_world: dict | None = None,
    ppo_model_path: str | None = None,
) -> MissionBrain:
    """Construct a MissionBrain backed by a real Isaac environment."""
    mission_world = dict(mission_world or {})
    scene_config = dict(
        scene_config
        or mission_world.get("scene_config")
        or mission_world.get("isaac_scene_config")
        or {}
    )
    if mission_world.get("aoi"):
        scene_config.setdefault("mission", {})["aoi"] = dict(mission_world["aoi"])

    runtime = IsaacAirCombatEnvironment(
        scene_config=scene_config,
        headless=headless,
    )
    world_state = {
        "environment_type": "isaac_air_combat",
        **mission_world,
    }
    context = make_context(
        world_state=world_state
    )
    return MissionBrain(
        context,
        MILPTaskAllocator(),
        MPPIFormationPlanner(),
        PositionSelector(),
        environment=runtime,
        ppo_reallocator=PPOAllocationAdapter(model_path=ppo_model_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac air-combat mission runner")
    parser.add_argument("--headless", action="store_true", help="Run Isaac headless")
    parser.add_argument("--ppo-model", default=None, metavar="PATH",
                        help="PPO reallocation model .zip (UAV loss reallocation)")
    parser.add_argument("--drl-policy", default=None, metavar="PATH",
                        help="DRL patrol policy .npz (UAV patrol inside AOI cells)")
    parser.add_argument("--max-steps", type=int, default=100_000,
                        help="Safety limit on Isaac step count")
    parser.add_argument("--mission-input", default=None, metavar="PATH",
                        help="External mission JSON; accepts aoi/aois/task_area/task_areas")
    parser.add_argument("--aoi", default=None, metavar="A_ROW_COL",
                        help="Override the single task area, e.g. A_3_4 or 3,4")
    parser.add_argument("--aois", default=None, metavar="A_3_4,A_3_5",
                        help="Override multiple task areas for multi-AOI recon")
    args = parser.parse_args()
    if args.aoi and args.aois:
        parser.error("--aoi and --aois cannot be used together")

    mission_world = load_mission_world(args.mission_input)
    apply_task_area_overrides(
        mission_world,
        aoi=args.aoi,
        aois=args.aois,
    )

    brain = build_brain(
        headless=args.headless,
        mission_world=mission_world,
        ppo_model_path=args.ppo_model,
    )

    executor = MissionExecutor(
        brain=brain,
        environment=brain.environment,
        drl_policy_path=args.drl_policy,
        max_steps=args.max_steps,
    )

    try:
        executor.run()
    finally:
        brain.close()

    final = brain.current_state
    logger.info("Mission ended: %s", final.value)
    logger.info(
        "History entries: %d | agents: %d | targets: %d",
        len(brain.context.history),
        len(brain.context.agents),
        len(brain.context.world_state.get("targets", [])),
    )


if __name__ == "__main__":
    main()
