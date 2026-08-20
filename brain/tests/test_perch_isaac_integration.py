from types import SimpleNamespace

import numpy as np

from brain.adapters import mppi_planner as mppi_module
from brain.adapters.mppi_planner import MPPIFormationPlanner
from brain.domain.position import Position
from brain.domain.task import StrikeTask, one_to_one_strike_tasks
from brain.integration import sync_context_from_air_combat_scene
from perch import FREAPositionSelector


def test_one_to_one_matching_merges_munitions():
    tasks = [
        StrikeTask("H1", "g1", "HF", 1),
        StrikeTask("H1", "g1", "RKT", 4),
        StrikeTask("H2", "g2", "GUN", 50),
    ]
    result = one_to_one_strike_tasks(tasks)

    assert {(task.platform, task.target) for task in result} == {
        ("H1", "g1"),
        ("H2", "g2"),
    }
    h1 = next(task for task in result if task.platform == "H1")
    assert h1.munition == "HF"
    assert h1.assigned_munitions == {"HF": 1, "RKT": 4}


def test_perch_uses_agl_weapon_envelope_obstacles_and_explanations():
    world = {
        "meters_per_unit": 100.0,
        "map_size_units": 3000.0,
        "map_size_km": 300.0,
        "terrain_fn": lambda x, y: 10.0 + 0.001 * x,
        "attack_min_agl_m": 30.0,
        "attack_max_agl_m": 300.0,
        "platform_states": [
            {"platform_id": "H1", "position_scene": [-100.0, 0.0, 15.0]},
            {"platform_id": "H2", "position_scene": [100.0, 0.0, 15.0]},
        ],
        "targets": [
            {
                "tid": "g1",
                "position_scene": [0.0, 0.0, 10.02],
                "alive": True,
                "confirmed": True,
                "threat": 0.8,
            },
            {
                "tid": "g2",
                "position_scene": [200.0, 0.0, 10.22],
                "alive": True,
                "confirmed": True,
                "threat": 0.7,
            },
        ],
        "obstacles": [
            {
                "obstacle_id": "ridge",
                "category": "mountain",
                "position_scene": [40.0, 0.0, 18.0],
                "radius_units": 3.0,
                "base_z": 10.04,
                "top_z": 18.0,
                "blocks_los": True,
            }
        ],
    }
    context = SimpleNamespace(world_state=world, agents=[])
    tasks = [
        StrikeTask("H1", "g1", "HF", 1),
        StrikeTask("H2", "g2", "GUN", 20),
    ]

    result = FREAPositionSelector(
        top_k=1, use_pymoo=False
    ).select(context, tasks)

    assert result.success, result.reason
    assert len(result.data) == 2
    assert {p.metadata["platform_id"] for p in result.data} == {"H1", "H2"}
    assert {p.metadata["target_id"] for p in result.data} == {"g1", "g2"}
    for position in result.data:
        metadata = position.metadata
        assert 30.0 <= metadata["agl_m"] <= 300.0
        envelope = metadata["weapon_envelope"]
        assert (
            envelope["min_range_km"]
            <= metadata["target_range_km"]
            <= envelope["max_range_km"]
        )
        assert metadata["g_violation"] == 0.0
        assert metadata["target_los_clear"]
        assert metadata["explanation"]
        assert all(
            item["satisfied"]
            for item in metadata["constraints"].values()
        )


def test_action_routes_are_independent_and_keep_position_altitude(monkeypatch):
    monkeypatch.setattr(mppi_module, "_MPPI_AVAILABLE", False)
    planner = MPPIFormationPlanner()
    context = SimpleNamespace(
        agents=[],
        world_state={
            "platform_states": [
                {"platform_id": "H1", "position_scene": [0.0, 0.0, 12.0]},
                {"platform_id": "H2", "position_scene": [10.0, 0.0, 13.0]},
            ]
        },
    )
    tasks = [
        StrikeTask("H1", "g1", "HF", 1),
        StrikeTask("H2", "g2", "GUN", 20),
        StrikeTask("H1", "g3", "RKT", 4),
    ]
    positions = [
        Position(
            "H1_g1_POS_00", 50.0, 20.0, 16.0,
            metadata={
                "platform_id": "H1",
                "target_id": "g1",
                "rank": 0,
                "explanation": ["H1 position"],
            },
        ),
        Position(
            "H2_g2_POS_00", 80.0, -30.0, 18.0,
            metadata={
                "platform_id": "H2",
                "target_id": "g2",
                "rank": 0,
                "explanation": ["H2 position"],
            },
        ),
    ]

    result = planner.plan_action_route(context, tasks, positions)

    assert result.success, result.reason
    plan = result.data
    assert plan.formation_type == "individual_strike"
    assert plan.team_count == 2
    assert len(plan.routes) == 2
    assert [(r.platform, r.target_id) for r in plan.routes] == [
        ("H1", "g1"),
        ("H2", "g2"),
    ]
    assert plan.routes[0].waypoints[-1] == [50.0, 20.0, 16.0]
    assert plan.routes[1].waypoints[-1] == [80.0, -30.0, 18.0]
    assert plan.planner_stats["mode"] == "one_aircraft_one_target"


def test_isaac_bridge_exports_exact_scene_state():
    missile = SimpleNamespace(
        name="Test AGM",
        role="anti-armor",
        max_range_km=12.0,
        guidance=("semi_active_laser",),
        fire_and_forget=False,
    )
    weapon_suite = SimpleNamespace(missiles=(missile,), guns=())
    platform_spec = SimpleNamespace(weapons=weapon_suite)
    platform = SimpleNamespace(
        entity_id="H1",
        position=np.array([1.0, 2.0, 13.0]),
        velocity=np.array([0.1, 0.2, 0.0]),
        motion_model=SimpleNamespace(
            state=SimpleNamespace(yaw_deg=25.0)
        ),
        spec=platform_spec,
    )
    target_spec = SimpleNamespace(
        category="armor",
        is_fixed=False,
        is_radiating=False,
        mobility_speed_mps=8.0,
        priority=70,
    )
    target = SimpleNamespace(
        target_id="g1",
        position=np.array([20.0, 30.0, 10.5]),
        spec=target_spec,
        destroyed=False,
    )
    obstacle = SimpleNamespace(
        obstacle_id="tree_1",
        category="tree",
        position=np.array([4.0, 5.0, 12.0]),
        radius_units=1.5,
        height_units=4.0,
        priority=40,
        blocks_los=False,
    )
    scene = SimpleNamespace(
        meters_per_unit=100.0,
        map_size_units=3000.0,
        terrain_visual_height_units=150.0,
        tactical_time_s=3.0,
        surface_height=lambda x, y: 10.0,
        platforms=[platform],
        targets=[target],
        obstacles=[obstacle],
        contacts=[{"target_id": "g1"}],
        obstacle_contacts=[],
    )
    context = SimpleNamespace(world_state={})

    world = sync_context_from_air_combat_scene(context, scene)

    assert world["terrain_fn"](0.0, 0.0) == 10.0
    assert world["platform_states"][0]["position_scene"] == [1.0, 2.0, 13.0]
    assert world["targets"][0]["position_scene"] == [20.0, 30.0, 10.5]
    assert world["targets"][0]["confirmed"]
    assert world["obstacles"][0]["base_z"] == 10.0
    assert world["obstacles"][0]["top_z"] == 14.0
    assert (
        world["platform_weapon_envelopes"]["H1"]["HF"]["max_range_km"]
        == 12.0
    )
