from types import SimpleNamespace

from brain.adapters.ppo_reallocator import PPOAllocationAdapter
from brain.core import MissionBrain, MissionState, make_context
from brain.domain.agent import AgentSpec
from brain.domain.result import AlgorithmResult
from brain.domain.task import ReconTask


class _FakeScene:
    def __init__(self):
        self.meters_per_unit = 100.0
        self.map_size_units = 3000.0
        self.terrain_visual_height_units = 150.0
        self.tactical_time_s = 0.0
        self.config = {
            "mission": {},
            "weather": {"rain": False},
            "simulation": {"time_scale": 1.0},
        }
        self.platforms = []
        self.targets = []
        self.obstacles = []
        self.contacts = []
        self.obstacle_contacts = []

    @staticmethod
    def surface_height(x, y):
        return 0.0


class _FakeEnvironment:
    def __init__(self):
        self.scene = _FakeScene()
        self.initialized = False
        self.initialize_calls = 0
        self.step_calls = 0

    def initialize(self):
        self.initialize_calls += 1
        self.initialized = True
        return self.scene

    def reset(self):
        return self.scene

    def step(self, dt):
        self.step_calls += 1
        self.scene.tactical_time_s += dt
        return self.scene


class _StubMILP:
    def allocate_recon(self, context):
        return AlgorithmResult.ok([
            ReconTask("U1", "c1", "SAR", "subarea_search")
        ])


class _StubMPPI:
    def __init__(self):
        self.recon_calls = 0

    def plan_recon_route(self, context, allocation):
        self.recon_calls += 1
        return AlgorithmResult.ok({"routes": 1})


class _UnusedPosition:
    pass


def test_brain_start_initializes_environment_before_recon_and_step_does_not_replan():
    context = make_context(world_state={})
    environment = _FakeEnvironment()
    mppi = _StubMPPI()
    brain = MissionBrain(
        context,
        _StubMILP(),
        mppi,
        _UnusedPosition(),
        environment=environment,
    )

    state = brain.start()

    assert state == MissionState.RECON_PLAN_READY
    assert environment.initialize_calls == 1
    assert context.world_state["isaac_scene"] is environment.scene
    assert any(
        item["event"] == "ENVIRONMENT_INITIALIZED"
        for item in context.history
    )
    assert mppi.recon_calls == 1

    stepped = brain.step_environment(0.1)
    assert stepped.success
    assert environment.step_calls == 1
    assert mppi.recon_calls == 1


class _FakePPOService:
    def __init__(self):
        self.scenario = None
        self.events = []

    def init(self, scenario):
        self.scenario = scenario
        return {"initialized": True}

    def handle_event(self, event):
        self.events.append(event)
        return {
            "uav_tasks": {
                "U0": {
                    "alive": False,
                    "task": "IDLE",
                    "sensor": "SAR",
                    "regions": [],
                },
                "U1": {
                    "alive": True,
                    "task": "SEARCH",
                    "sensor": "SAR",
                    "regions": ["R0", "R1"],
                },
                "U2": {
                    "alive": True,
                    "task": "SEARCH",
                    "sensor": "EO",
                    "regions": ["R2"],
                },
                "U3": {
                    "alive": True,
                    "task": "SEARCH",
                    "sensor": "SAR",
                    "regions": ["R3"],
                },
            },
            "region_assignments": {
                "R0": "U1",
                "R1": "U1",
                "R2": "U2",
                "R3": "U3",
            },
            "event_applied": "U0 damaged",
            "action_detail": "regions repaired",
        }


def test_uav_loss_calls_ppo_allocation_and_translates_back_to_platforms():
    agents = [
        AgentSpec(f"Recon_{index}", "UAV", (0.0, 0.0))
        for index in range(4)
    ]
    tasks = [
        ReconTask(
            f"Recon_{index}",
            f"c{index + 1}",
            "SAR",
            "subarea_search",
        )
        for index in range(4)
    ]
    context = SimpleNamespace(
        agents=agents,
        recon_allocation=tasks,
        world_state={"aoi": {"row": 3, "col": 4}},
    )
    service = _FakePPOService()
    adapter = PPOAllocationAdapter(service=service)

    result = adapter.handle_platform_loss(context, "Recon_0")

    assert result.success, result.reason
    assert service.events == [
        {"event_type": "UAV_DAMAGE", "uav_id": 0}
    ]
    assert agents[0].lost
    assert all(
        task.platform != "Recon_0"
        for task in result.data["recon_allocation"]
    )
    reassigned = {
        (task.platform, task.cell)
        for task in result.data["recon_allocation"]
        if task.role == "subarea_search"
    }
    assert ("Recon_1", "c1") in reassigned
    assert ("Recon_1", "c2") in reassigned
