import json

from brain.adapters.milp_task_allocator import MILPTaskAllocator
from brain.core import make_context
from brain.integration import (
    ScenarioInitializer,
    apply_task_area_overrides,
    load_mission_world,
)


def test_external_mission_input_normalizes_task_areas(tmp_path):
    path = tmp_path / "mission.json"
    path.write_text(
        json.dumps({
            "world_state": {
                "commander_AOI": ["A_2_5"],
                "grid_weather": {
                    "_说明": "template comments must not enter MILP",
                    "c0": 0.25,
                },
                "platforms": [{
                    "pid": "U1",
                    "type": "UAV",
                    "pos": [10, 20],
                    "sensors_mounted": ["SAR"],
                }],
                "targets": [],
            }
        }),
        encoding="utf-8",
    )

    world = load_mission_world(path)
    context = make_context(world_state=world)
    ScenarioInitializer().normalize(context)

    assert context.world_state["aoi"] == {"row": 2, "col": 5}
    assert context.world_state["commander_AOI"] == ["A_2_5"]
    assert context.world_state["task_areas"][0]["id"] == "A_2_5"
    assert context.world_state["weather"] == {"c0": 0.25}
    assert context.agents[0].sensors == ["SAR"]


def test_startup_aoi_overrides_feed_milp_dict(sample_agents):
    world = {
        "task_area": "A_1_2",
        "targets": [],
    }
    apply_task_area_overrides(world, aois="A_3_4,A_3_5")
    context = make_context(agents=sample_agents, world_state=world)
    ScenarioInitializer().normalize(context)

    input_dict = MILPTaskAllocator.context_to_dict(context, phase="recon")

    assert input_dict["aoi"] == {"row": 3, "col": 4}
    assert input_dict["commander_AOI"] == ["A_3_4", "A_3_5"]
    assert [item["id"] for item in input_dict["aois"]] == [
        "A_3_4",
        "A_3_5",
    ]
