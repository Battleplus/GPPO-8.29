"""
CPython bridge used by the C++ MILP interface.

This module is intentionally small and imports only the core allocation
interfaces. It must not import frontend_app, visualization, streamlit, or
plotly, so an embedded C++ caller executes only the allocation path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


_MILP_DIR = Path(__file__).resolve().parent
if str(_MILP_DIR) not in sys.path:
    sys.path.insert(0, str(_MILP_DIR))

from execution_output_interface import to_json as execution_order_to_json
from multi_aoi_interface import MultiAOITaskAllocator
from task_interface import solve as solve_single_aoi


class MilpBridge:
    """JSON boundary for single-AOI and multi-AOI allocation calls."""

    def __init__(
        self,
        solver: str = "cbc",
        time_limit_s: float = 3.0,
        verbose: int = 0,
    ) -> None:
        self.solver = solver
        self.time_limit_s = float(time_limit_s)
        self.verbose = int(verbose)
        self._multi_allocator = MultiAOITaskAllocator(
            solver=self.solver,
            time_limit_s=self.time_limit_s,
            verbose=self.verbose,
        )

    def solve_single_aoi_json(self, input_json: str) -> str:
        """Run the single-AOI allocator with a JSON string input."""
        input_data = _loads_object(input_json, "single AOI input")
        result = solve_single_aoi(
            input_data,
            solver=self.solver,
            time_limit_s=self.time_limit_s,
            verbose=self.verbose,
        )
        return execution_order_to_json(result)

    def solve_single_aoi_file(self, input_path: str) -> str:
        """Run the single-AOI allocator with a UTF-8 JSON file input."""
        return self.solve_single_aoi_json(_read_text_file(input_path))

    def run_multi_aoi_json(self, input_json: str) -> str:
        """Run one multi-AOI allocation step with a JSON string input."""
        input_data = _loads_object(input_json, "multi AOI input")
        result = self._multi_allocator.run(input_data)
        return json.dumps(result, ensure_ascii=False, indent=2)

    def run_multi_aoi_file(self, input_path: str) -> str:
        """Run one multi-AOI allocation step with a UTF-8 JSON file input."""
        return self.run_multi_aoi_json(_read_text_file(input_path))


def create_bridge(
    solver: str = "cbc",
    time_limit_s: float = 3.0,
    verbose: int = 0,
) -> MilpBridge:
    """Factory used by the C++ embedding layer."""
    return MilpBridge(solver=solver, time_limit_s=time_limit_s, verbose=verbose)


def _loads_object(input_json: str, name: str) -> dict[str, Any]:
    if not isinstance(input_json, str):
        raise TypeError(f"{name} must be a JSON string")
    try:
        value = json.loads(input_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must decode to a JSON object")
    return value


def _read_text_file(input_path: str) -> str:
    if not isinstance(input_path, str):
        raise TypeError("input_path must be a string")
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"input JSON file does not exist: {input_path}")
    return path.read_text(encoding="utf-8")
