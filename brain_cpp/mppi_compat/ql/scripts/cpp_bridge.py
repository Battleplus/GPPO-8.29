"""Adapt the packaged QL module name to this repository's MPPI package."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mppi.cpp_bridge import plan_json as _plan_json  # noqa: E402
from mppi.planner import FormationMPPIPlanner  # noqa: E402


def _path_curvature(path: list[np.ndarray]) -> float:
    """NumPy 2-compatible implementation of the packaged curvature metric."""
    if len(path) < 3:
        return 0.0
    maximum = 0.0
    for index in range(1, len(path) - 1):
        first = path[index] - path[index - 1]
        second = path[index + 1] - path[index]
        cross = abs(float(first[0] * second[1] - first[1] * second[0]))
        denominator = float(np.linalg.norm(first[:2]))
        denominator *= float(np.linalg.norm(second[:2]))
        if denominator > 1e-9:
            maximum = max(maximum, cross / denominator)
    return maximum


FormationMPPIPlanner._path_curvature = staticmethod(_path_curvature)


def plan_json(request_json: str) -> str:
    return _plan_json(request_json)
