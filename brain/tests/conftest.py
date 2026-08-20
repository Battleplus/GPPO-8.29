"""Shared pytest fixtures for brain tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the parent of brain is importable
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from brain.adapters import MILPTaskAllocator, MPPIFormationPlanner, PositionSelector
from brain.core import MissionBrain, make_context
from brain.domain.agent import AgentSpec


# ---------------------------------------------------------------------------
# Standard force: 5 UAVs + 2 HELIs
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_agents() -> list[AgentSpec]:
    agents: list[AgentSpec] = []
    for i in range(1, 6):
        agents.append(
            AgentSpec(
                pid=f"U{i}",
                type="UAV",
                position=(150.0, -50.0),
                sensors=["EO", "SAR", "ESM"],
                altitude_km=2.0,
            )
        )
    for i in range(1, 3):
        agents.append(
            AgentSpec(
                pid=f"H{i}",
                type="HELI",
                position=(150.0, -50.0),
                sensors=["MMW", "EOIR"],
                munitions={"HF": 16, "RKT": 76, "GUN": 1200},
                altitude_km=3.0,
            )
        )
    return agents


# ---------------------------------------------------------------------------
# Standard world state with 3 targets
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_world() -> dict:
    return {
        "aoi": {"row": 3, "col": 4},
        "staging_position": [150.0, -50.0],
        "weather": {"c0": 0.2, "c1": 0.15, "c2": 0.4, "c3": 0.55, "c4": 0.7},
        "terrain": {"c0": 0, "c1": 0, "c2": 1, "c3": 0, "c4": 2},
        "targets": [
            {
                "tid": "g1",
                "type": "RADAR",
                "pos": [270, 260],
                "value": 1.0,
                "threat": 0.9,
                "confirmed": False,
                "alive": True,
            },
            {
                "tid": "g2",
                "type": "CP",
                "pos": [310, 180],
                "value": 0.95,
                "threat": 0.85,
                "confirmed": False,
                "alive": True,
            },
            {
                "tid": "g3",
                "type": "AV",
                "pos": [220, 310],
                "value": 0.7,
                "threat": 0.65,
                "confirmed": False,
                "alive": True,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Standard context
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_context(sample_agents, sample_world):
    return make_context(
        mission_id="TEST_001",
        agents=sample_agents,
        world_state=sample_world,
        max_retry=3,
    )


# ---------------------------------------------------------------------------
# Standard adapters (stub, successful)
# ---------------------------------------------------------------------------

@pytest.fixture
def milp_ok():
    return MILPTaskAllocator()


@pytest.fixture
def mppi_ok():
    return MPPIFormationPlanner()


@pytest.fixture
def pos_ok():
    return PositionSelector()


# ---------------------------------------------------------------------------
# Standard brain
# ---------------------------------------------------------------------------

@pytest.fixture
def brain(sample_context, milp_ok, mppi_ok, pos_ok):
    return MissionBrain(sample_context, milp_ok, mppi_ok, pos_ok)
