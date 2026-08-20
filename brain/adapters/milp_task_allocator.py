"""MILP task allocation adapter.

Converts :class:`MissionContext` fields into the ``SituationSnapshot``
that ``milp.task_interface.TaskAllocator`` expects, calls the MILP
solver, and converts the resulting ``AllocationPlan`` back into
domain-level ``ReconTask`` / ``StrikeTask`` lists.

When the real MILP solver is not available (e.g. missing ``python-mip``)
the adapter falls back to a deterministic placeholder allocator so that
the mission pipeline remains functional.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..domain.result import AlgorithmResult
from ..domain.task import ReconTask, StrikeTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve the MILP package path
# ---------------------------------------------------------------------------
_PROJECT = Path(__file__).resolve().parents[2]
_MILP_DIR = _PROJECT / "milp"

# Project root first (for solve() in root task_interface.py)
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))
# milp dir second (for milp internal modules: config, core, allocation)
if str(_MILP_DIR) not in sys.path:
    sys.path.insert(1, str(_MILP_DIR))

# ---------------------------------------------------------------------------
# Attempt to import the real MILP solver
# ---------------------------------------------------------------------------
_MILP_AVAILABLE = False
_DICT_SOLVE_AVAILABLE = False

try:
    from task_interface import (  # type: ignore[import-not-found]
        TaskAllocator as _TaskAllocator,
        generate_aoi_grids as _generate_aoi_grids,
        make_platform as _make_platform,
        make_sensor_params as _make_sensor_params,
        make_snapshot as _make_snapshot,
        make_target as _make_target,
    )

    _MILP_AVAILABLE = True
    logger.info("MILP adapter: real solver available (task_interface imported)")
except ImportError:
    logger.warning(
        "MILP adapter: real solver NOT available (missing python-mip or "
        "other dependency). Using deterministic placeholder allocator."
    )

# 方式二: 字典式调用 solve() — 来自项目根 task_interface.py
try:
    from task_interface import solve as _solve_dict  # type: ignore[import-not-found]
    _DICT_SOLVE_AVAILABLE = True
    logger.info("MILP adapter: solve() dict API available (方式二)")
except ImportError:
    logger.warning("MILP adapter: solve() dict API NOT available")


class MILPTaskAllocator:
    """Adapter around ``milp.task_interface.TaskAllocator``.

    The allocator is created once and reused across cycles (the
    underlying MILP solver supports warm-start when available).
    """

    def __init__(
        self,
        solver: str = "cbc",
        time_limit_s: float = 3.0,
        verbose: int = 0,
    ) -> None:
        self._solver = solver
        self._time_limit_s = time_limit_s
        self._verbose = verbose
        self._allocator = None  # lazily created (real path only)
        self._multi_aoi_allocator = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def allocate_recon(self, context: Any) -> AlgorithmResult:
        """Run MILP to produce a **reconnaissance** task allocation."""
        agents = getattr(context, "agents", [])
        if not agents:
            return AlgorithmResult.fail("No agents available for recon allocation")

        uavs = [a for a in agents if getattr(a, "type", None) == "UAV"]
        if not uavs:
            return AlgorithmResult.fail("No UAV platforms for recon allocation")

        if self._is_multi_aoi_context(context):
            multi_result = self._allocate_recon_multi_aoi(context)
            if multi_result.success:
                return multi_result
            logger.warning(
                "MILP multi-AOI recon failed (%s); falling back to single AOI",
                multi_result.reason,
            )

        if _MILP_AVAILABLE:
            return self._allocate_recon_real(context)
        return self._allocate_recon_placeholder(context)

    def allocate_action(
        self,
        context: Any,
        target_ids: Iterable[str] | None = None,
        *,
        include_engaged: bool = False,
    ) -> AlgorithmResult:
        """Run MILP to produce a **strike** task allocation."""
        agents = getattr(context, "agents", [])
        world = getattr(context, "world_state", {})

        helis = [a for a in agents if getattr(a, "type", None) == "HELI"]
        eligible_ids = self._eligible_action_target_ids(
            context, target_ids, include_engaged=include_engaged
        )
        targets = [
            t for t in world.get("targets", [])
            if str(t.get("tid", t.get("target_id", ""))) in eligible_ids
        ]

        if not helis:
            return AlgorithmResult.fail("No HELI platforms for action allocation")
        if not targets:
            return AlgorithmResult.fail("No confirmed targets for strike allocation")

        if _MILP_AVAILABLE:
            return self._allocate_action_real(
                context, target_ids=target_ids, include_engaged=include_engaged
            )
        return self._allocate_action_placeholder(
            context, target_ids=target_ids, include_engaged=include_engaged
        )

    # ------------------------------------------------------------------
    # Real MILP path
    # ------------------------------------------------------------------

    def _get_allocator(self):
        if self._allocator is None:
            self._allocator = _TaskAllocator(
                solver=self._solver,
                time_limit_s=self._time_limit_s,
                verbose=self._verbose,
            )
        return self._allocator

    # ------------------------------------------------------------------
    # 方式二: 字典式输入构造 (dict → solve())
    # ------------------------------------------------------------------

    @staticmethod
    def context_to_dict(
        context: Any,
        phase: str = "recon",
        target_ids: Iterable[str] | None = None,
        *,
        include_engaged: bool = False,
    ) -> dict:
        """将 MissionContext 转换为 ``solve()`` 字典式输入 (方式二).

        这是大脑 → MILP 的标准化数据契约。调用方只需传入 context，
        即可获得可直接传给 ``task_interface.solve()`` 的 dict。

        Parameters
        ----------
        context:
            MissionContext, 含 agents + world_state
        phase:
            ``"recon"`` — 侦察阶段, 所有 alive 目标都参与;
            ``"action"`` — 打击阶段, 仅 confirmed+alive 目标参与.

        Returns
        -------
        dict
            可直接传入 ``solve()`` 的字典, 格式与
            ``templates/input_template.json`` 兼容.

        Example
        -------
            >>> from task_interface import solve
            >>> input_dict = MILPTaskAllocator.context_to_dict(ctx, "recon")
            >>> eo = solve(input_dict)
            >>> for t in eo.tasks:
            ...     print(t.platform, t.task_type, t.cell or t.target)
        """
        agents = getattr(context, "agents", [])
        world = getattr(context, "world_state", {})

        # -- AOI ---------------------------------------------------------
        aoi = world.get("aoi", {"row": 3, "col": 4})

        # -- Staging position --------------------------------------------
        staging_raw = world.get("staging_position", [150.0, -50.0])
        staging = [
            float(staging_raw[0]),
            float(staging_raw[1]),
        ]

        # -- Platforms (逐架列出) -----------------------------------------
        platforms = []
        for a in agents:
            pos = getattr(a, "position", (150.0, -50.0))
            platforms.append({
                "pid": str(getattr(a, "pid", "")),
                "type": str(getattr(a, "type", "UAV")),
                "pos": [float(pos[0]), float(pos[1])] if hasattr(pos, '__getitem__') else [150.0, -50.0],
                "sensors": (
                    list(a.sensors) if getattr(a, "sensors", None)
                    else (["EO", "SAR", "ESM"] if getattr(a, "type", "") == "UAV"
                          else ["MMW", "EOIR"])
                ),
                "munitions": (
                    dict(a.munitions)
                    if getattr(a, "munitions", None)
                    else ({"HF": 0, "RKT": 0, "GUN": 0}
                          if getattr(a, "type", "") == "UAV"
                          else {"HF": 16, "RKT": 76, "GUN": 1200})
                ),
                "alt": float(getattr(a, "altitude_km", 2.0)),
                "lost": bool(getattr(a, "lost", False)),
            })

        # -- Targets -----------------------------------------------------
        raw_targets = world.get("targets", [])
        eligible_action_ids = MILPTaskAllocator._eligible_action_target_ids(
            context, target_ids, include_engaged=include_engaged
        ) if phase == "action" else set()
        targets = []
        for t in raw_targets:
            tid = str(t.get("tid", t.get("target_id", "")))
            if not t.get("alive", True):
                continue
            if phase == "action" and tid not in eligible_action_ids:
                continue
            targets.append({
                "tid": tid,
                "type": str(t.get("type", "AV")),
                "pos": [
                    float(t.get("pos", [0.0, 0.0])[0]),
                    float(t.get("pos", [0.0, 0.0])[1]) if len(t.get("pos", [0, 0])) > 1 else 0.0,
                ],
                "value": float(t.get("value", 0.5)),
                "threat": float(t.get("threat", 0.5)),
                "confirmed": bool(t.get("confirmed", False)),
                "alive": bool(t.get("alive", True)),
            })

        input_dict: dict[str, Any] = {
            "aoi": {"row": int(aoi.get("row", 3)), "col": int(aoi.get("col", 4))},
            "commander_AOI": MILPTaskAllocator._commander_aoi_labels(world),
            "staging_position": staging,
            "platforms": platforms,
            "targets": targets,
        }
        if world.get("aois"):
            input_dict["aois"] = list(world.get("aois", []))

        # -- 可选字段 ----------------------------------------------------
        weather = world.get("weather", {})
        if weather:
            input_dict["grid_weather"] = {
                str(k): float(v)
                for k, v in weather.items()
                if not str(k).startswith("_")
            }
        terrain = world.get("terrain", {})
        if terrain:
            input_dict["grid_terrain"] = {
                str(k): int(v)
                for k, v in terrain.items()
                if not str(k).startswith("_")
            }

        return input_dict

    def _build_snapshot(
        self,
        context: Any,
        phase: str,
        target_ids: Iterable[str] | None = None,
        *,
        include_engaged: bool = False,
    ):
        """Convert MissionContext → SituationSnapshot."""
        agents = getattr(context, "agents", [])
        world = getattr(context, "world_state", {})

        # -- AOI grids ----------------------------------------------------
        aoi = world.get("aoi", {"row": 3, "col": 4})
        grids = _generate_aoi_grids(
            aoi_row=int(aoi.get("row", 3)),
            aoi_col=int(aoi.get("col", 4)),
        )
        weather = world.get("weather", {})
        terrain = world.get("terrain", {})
        for g in grids:
            if g.cell_id in weather:
                g.weather_w = float(weather[g.cell_id])
            if g.cell_id in terrain:
                g.terrain_level = int(terrain[g.cell_id])

        # -- Platforms -----------------------------------------------------
        platforms = [
            _make_platform(
                pid=a.pid,
                platform_type=a.type,
                pos_xy=a.position,
                sensors=list(a.sensors) if a.sensors else None,
                munitions=dict(a.munitions) if a.munitions else None,
                alt=a.altitude_km,
                lost=a.lost,
            )
            for a in agents
        ]

        # -- Targets -------------------------------------------------------
        raw_targets = world.get("targets", [])
        if phase == "recon":
            targets = [
                _make_target(
                    tid=t.get("tid", t.get("target_id", f"g{i}")),
                    target_type=t.get("type", "AV"),
                    pos_xy=tuple(t.get("pos", [0, 0])),
                    confirmed=False,
                    alive=t.get("alive", True),
                    value=float(t.get("value", 0.5)),
                    threat=float(t.get("threat", 0.5)),
                )
                for i, t in enumerate(raw_targets)
                if t.get("alive", True)
            ]
        else:
            eligible_action_ids = self._eligible_action_target_ids(
                context, target_ids, include_engaged=include_engaged
            )
            targets = [
                _make_target(
                    tid=t.get("tid", t.get("target_id", f"g{i}")),
                    target_type=t.get("type", "AV"),
                    pos_xy=tuple(t.get("pos", [0, 0])),
                    confirmed=t.get("confirmed", False),
                    alive=t.get("alive", True),
                    value=float(t.get("value", 0.5)),
                    threat=float(t.get("threat", 0.5)),
                )
                for i, t in enumerate(raw_targets)
                if str(t.get("tid", t.get("target_id", ""))) in eligible_action_ids
            ]

        # -- Sensor params -------------------------------------------------
        sensor_params = [
            _make_sensor_params("EO", P0=0.85, R=15.0, weather_sensitive=True),
            _make_sensor_params("SAR", P0=0.90, R=50.0, weather_sensitive=False),
            _make_sensor_params("ESM", P0=0.80, R=100.0, weather_sensitive=False),
        ]

        staging_raw = world.get("staging_position", [0.0, 0.0])
        staging = np.array(staging_raw, dtype=np.float64)

        return _make_snapshot(
            cycle_id=0,
            grids=grids,
            targets=targets,
            platforms=platforms,
            sensor_params=sensor_params,
            staging_position=staging,
        )

    def _allocate_recon_real(self, context: Any) -> AlgorithmResult:
        """侦察分配 — 优先使用 solve() 字典方式 (方式二)."""
        if _DICT_SOLVE_AVAILABLE:
            return self._allocate_recon_via_dict(context)
        return self._allocate_recon_via_snapshot(context)

    def _allocate_recon_via_dict(self, context: Any) -> AlgorithmResult:
        """方式二: context → dict → solve() → ExecutionOrder → ReconTask[]."""
        try:
            input_dict = self.context_to_dict(context, phase="recon")
            eo = _solve_dict(
                input_dict,
                solver=self._solver,
                time_limit_s=self._time_limit_s,
                verbose=self._verbose,
            )

            if eo.solve_status not in ("OPTIMAL", "FEASIBLE"):
                return AlgorithmResult.fail(
                    f"MILP recon (dict): solver status={eo.solve_status}"
                )

            world = getattr(context, "world_state", {})
            aoi_label = self._aoi_label(world)
            recon_tasks = [
                ReconTask(
                    platform=t.platform,
                    cell=t.cell,
                    sensor=t.sensor_used,  # execution_output 用 sensor_used
                    role=t.role,
                    aoi=t.aoi or aoi_label,
                )
                for t in eo.recon_tasks
            ]
            logger.info(
                "MILP recon (dict): status=%s objective=%.1f tasks=%d",
                eo.solve_status, eo.objective, len(recon_tasks),
            )
            return AlgorithmResult.ok(recon_tasks)
        except Exception as exc:
            logger.exception("MILP recon (dict) failed, falling back to snapshot")
            return self._allocate_recon_via_snapshot(context)

    def _allocate_recon_via_snapshot(self, context: Any) -> AlgorithmResult:
        """Snapshot 方式: 原有的 SituationSnapshot → TaskAllocator 路径."""
        try:
            snapshot = self._build_snapshot(context, phase="recon")
            plan = self._get_allocator().solve(snapshot)

            if plan.status not in ("OPTIMAL", "FEASIBLE"):
                return AlgorithmResult.fail(
                    f"MILP recon allocation: solver status={plan.status}"
                )

            world = getattr(context, "world_state", {})
            aoi_label = self._aoi_label(world)
            recon_tasks = [
                ReconTask(
                    platform=ra.pid, cell=ra.cell, sensor=ra.sensor,
                    role=ra.role, aoi=aoi_label,
                )
                for ra in plan.recon_assignments
            ]
            logger.info(
                "MILP recon (snapshot): status=%s objective=%.1f tasks=%d",
                plan.status, plan.objective, len(recon_tasks),
            )
            return AlgorithmResult.ok(recon_tasks)
        except Exception as exc:
            logger.exception("MILP recon allocation failed")
            return AlgorithmResult.fail(str(exc))

    def _allocate_action_real(
        self,
        context: Any,
        target_ids: Iterable[str] | None = None,
        *,
        include_engaged: bool = False,
    ) -> AlgorithmResult:
        """打击分配 — 优先使用 solve() 字典方式 (方式二)."""
        if _DICT_SOLVE_AVAILABLE:
            return self._allocate_action_via_dict(
                context, target_ids=target_ids, include_engaged=include_engaged
            )
        return self._allocate_action_via_snapshot(
            context, target_ids=target_ids, include_engaged=include_engaged
        )

    def _allocate_action_via_dict(
        self,
        context: Any,
        target_ids: Iterable[str] | None = None,
        *,
        include_engaged: bool = False,
    ) -> AlgorithmResult:
        """方式二: context → dict → solve() → ExecutionOrder → StrikeTask[]."""
        try:
            input_dict = self.context_to_dict(
                context,
                phase="action",
                target_ids=target_ids,
                include_engaged=include_engaged,
            )
            eo = _solve_dict(
                input_dict,
                solver=self._solver,
                time_limit_s=self._time_limit_s,
                verbose=self._verbose,
            )

            if eo.solve_status not in ("OPTIMAL", "FEASIBLE"):
                return AlgorithmResult.fail(
                    f"MILP action (dict): solver status={eo.solve_status}"
                )

            strike_tasks = [
                StrikeTask(
                    platform=t.platform,
                    target=t.target,
                    munition=t.munition,
                    qty=t.qty,
                    role=t.role,
                )
                for t in eo.strike_tasks
            ]
            logger.info(
                "MILP action (dict): status=%s objective=%.1f tasks=%d",
                eo.solve_status, eo.objective, len(strike_tasks),
            )
            return AlgorithmResult.ok(strike_tasks)
        except Exception as exc:
            logger.exception("MILP action (dict) failed, falling back to snapshot")
            return self._allocate_action_via_snapshot(
                context, target_ids=target_ids, include_engaged=include_engaged
            )

    def _allocate_action_via_snapshot(
        self,
        context: Any,
        target_ids: Iterable[str] | None = None,
        *,
        include_engaged: bool = False,
    ) -> AlgorithmResult:
        """Snapshot 方式: 原有的 SituationSnapshot → TaskAllocator 路径."""
        try:
            snapshot = self._build_snapshot(
                context,
                phase="action",
                target_ids=target_ids,
                include_engaged=include_engaged,
            )
            plan = self._get_allocator().solve(snapshot)

            if plan.status not in ("OPTIMAL", "FEASIBLE"):
                return AlgorithmResult.fail(
                    f"MILP action allocation: solver status={plan.status}"
                )

            strike_tasks = [
                StrikeTask(
                    platform=sa.pid, target=sa.target,
                    munition=sa.munition, qty=sa.qty, role=sa.role,
                )
                for sa in plan.strike_assignments
            ]
            logger.info(
                "MILP action (snapshot): status=%s objective=%.1f tasks=%d",
                plan.status, plan.objective, len(strike_tasks),
            )
            return AlgorithmResult.ok(strike_tasks)
        except Exception as exc:
            logger.exception("MILP action allocation failed")
            return AlgorithmResult.fail(str(exc))

    # ------------------------------------------------------------------
    # Placeholder path (deterministic, no solver dependency)
    # ------------------------------------------------------------------

    def _allocate_recon_placeholder(self, context: Any) -> AlgorithmResult:
        agents = getattr(context, "agents", [])
        world = getattr(context, "world_state", {})
        uavs = [a for a in agents if getattr(a, "type", None) == "UAV"]
        aoi = world.get("aoi", {"row": 3, "col": 4})
        cells = ["c0", "c1", "c2", "c3", "c4"]
        sensors_cycle = ["SAR", "EO", "ESM"]
        aoi_label = self._aoi_label(world)

        recon_tasks: list[ReconTask] = []
        for i, uav in enumerate(uavs):
            recon_tasks.append(ReconTask(
                platform=uav.pid,
                cell=cells[i % len(cells)],
                sensor=sensors_cycle[i % len(sensors_cycle)],
                role="area_scan" if cells[i % len(cells)] == "c0" else "subarea_search",
                aoi=aoi_label,
            ))

        logger.info("MILP recon (placeholder): allocated %d tasks", len(recon_tasks))
        return AlgorithmResult.ok(recon_tasks)

    def _allocate_action_placeholder(
        self,
        context: Any,
        target_ids: Iterable[str] | None = None,
        *,
        include_engaged: bool = False,
    ) -> AlgorithmResult:
        agents = getattr(context, "agents", [])
        world = getattr(context, "world_state", {})
        helis = [a for a in agents if getattr(a, "type", None) == "HELI"]
        eligible_ids = self._eligible_action_target_ids(
            context, target_ids, include_engaged=include_engaged
        )
        targets = [
            t for t in world.get("targets", [])
            if str(t.get("tid", t.get("target_id", ""))) in eligible_ids
        ]
        munition_map = {"RADAR": ("HF", 2), "CP": ("HF", 2), "AV": ("HF", 1)}

        strike_tasks: list[StrikeTask] = []
        for i, target in enumerate(targets):
            heli = helis[i % len(helis)]
            ttype = target.get("type", "AV")
            mun, qty = munition_map.get(ttype, ("HF", 1))
            strike_tasks.append(StrikeTask(
                platform=heli.pid,
                target=target.get("tid", target.get("target_id", f"t{i}")),
                munition=mun, qty=qty,
                role="lead" if i % len(helis) == 0 else "wing",
            ))

        logger.info("MILP action (placeholder): allocated %d tasks", len(strike_tasks))
        return AlgorithmResult.ok(strike_tasks)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _target_id_set(target_ids: Iterable[str] | None) -> set[str]:
        if target_ids is None:
            return set()
        return {str(item) for item in target_ids if str(item)}

    @staticmethod
    def _eligible_action_target_ids(
        context: Any,
        target_ids: Iterable[str] | None = None,
        *,
        include_engaged: bool = False,
    ) -> set[str]:
        world = getattr(context, "world_state", {})
        explicit_ids = MILPTaskAllocator._target_id_set(target_ids)
        pending_ids = {
            str(item)
            for item in getattr(context, "pending_strike_targets", [])
            if str(item)
        }
        if not pending_ids:
            pending_ids = {
                str(item)
                for item in world.get("pending_strike_targets", [])
                if str(item)
            }
        eligible = explicit_ids or pending_ids
        confirmed_alive = {
            str(target.get("tid", target.get("target_id", "")))
            for target in world.get("targets", [])
            if target.get("alive", True) and target.get("confirmed", False)
        }
        if eligible:
            eligible &= confirmed_alive
        else:
            eligible = set(confirmed_alive)

        if not include_engaged:
            engaged = {
                str(item)
                for item in getattr(context, "engaged_targets", set())
                if str(item)
            }
            if not engaged:
                engaged = {
                    str(item)
                    for item in world.get("engaged_targets", [])
                    if str(item)
                }
            eligible -= engaged
        return eligible

    @staticmethod
    def _is_multi_aoi_context(context: Any) -> bool:
        world = getattr(context, "world_state", {})
        aois = getattr(context, "aois", None) or world.get("aois", [])
        return isinstance(aois, list) and len(aois) > 1

    def _get_multi_aoi_allocator(self):
        if self._multi_aoi_allocator is not None:
            return self._multi_aoi_allocator
        if str(_PROJECT) not in sys.path:
            sys.path.insert(0, str(_PROJECT))
        from multi_aoi_interface import MultiAOITaskAllocator

        self._multi_aoi_allocator = MultiAOITaskAllocator(
            solver=self._solver,
            time_limit_s=self._time_limit_s,
            verbose=self._verbose,
        )
        return self._multi_aoi_allocator

    def _allocate_recon_multi_aoi(self, context: Any) -> AlgorithmResult:
        try:
            request = self._build_multi_aoi_request(context)
            result = self._get_multi_aoi_allocator().run(request)
        except Exception as exc:
            logger.exception("MILP multi-AOI recon allocation failed")
            return AlgorithmResult.fail(str(exc))

        world = getattr(context, "world_state", {})
        context.aoi_route_state = result.get("aoi_route_state")
        world["aoi_route_state"] = context.aoi_route_state
        world["execution_feedback"] = getattr(context, "execution_feedback", None)
        if result.get("status") == "ALL_AOI_FINISHED":
            return AlgorithmResult.ok([])

        current_plan = result.get("current_aoi_plan") or {}
        current_aoi_id = str(current_plan.get("aoi", ""))
        current_aoi = next(
            (
                aoi for aoi in (getattr(context, "aois", []) or world.get("aois", []))
                if str(aoi.get("id", "")) == current_aoi_id
            ),
            None,
        )
        if current_aoi is not None:
            world["aoi"] = {
                "row": int(current_aoi.get("row", 3)),
                "col": int(current_aoi.get("col", 4)),
            }

        recon_tasks = [
            ReconTask(
                platform=str(item.get("platform", "")),
                cell=str(item.get("cell", "c0")),
                sensor=str(item.get("sensor", "SAR")),
                role=str(item.get("role", "area_scan")),
                aoi=str(item.get("aoi", current_aoi_id)),
            )
            for item in current_plan.get("tasks", [])
            if item.get("task_type") == "recon" and item.get("platform")
        ]
        logger.info(
            "MILP recon (multi-AOI): status=%s aoi=%s tasks=%d",
            result.get("status"), current_aoi_id, len(recon_tasks),
        )
        return AlgorithmResult.ok(recon_tasks)

    def _build_multi_aoi_request(self, context: Any) -> dict[str, Any]:
        world = getattr(context, "world_state", {})
        return {
            "aois": list(getattr(context, "aois", None) or world.get("aois", [])),
            "platforms": self._platform_config_from_agents(context),
            "targets": list(world.get("targets", [])),
            "sensor_params": world.get("sensor_params"),
            "staging_position": world.get("staging_position", [150.0, -50.0]),
            "cycle_id": int(world.get("cycle_id", len(getattr(context, "history", [])))),
            "grid_weather": world.get("grid_weather", world.get("weather", {})),
            "aoi_route_state": getattr(context, "aoi_route_state", None),
            "execution_feedback": getattr(context, "execution_feedback", None),
        }

    @staticmethod
    def _platform_config_from_agents(context: Any) -> dict[str, dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for agent in getattr(context, "agents", []):
            ptype = str(getattr(agent, "type", "UAV"))
            cfg = grouped.setdefault(ptype, {
                "count": 0,
                "sensors": list(getattr(agent, "sensors", []) or []),
                "munitions": dict(getattr(agent, "munitions", {}) or {}),
                "alt": float(getattr(agent, "altitude_km", 2.0)),
                "pos": list(getattr(agent, "position", (150.0, -50.0))),
            })
            cfg["count"] += 1
        return grouped

    @staticmethod
    def _aoi_label(world: dict) -> str:
        aoi = world.get("aoi", {})
        return f"A_{aoi.get('row', 3)}_{aoi.get('col', 4)}"

    @staticmethod
    def _commander_aoi_labels(world: dict) -> list[str]:
        labels = world.get("commander_AOI", world.get("commander_aoi"))
        if labels:
            return [str(item) for item in labels]
        aois = world.get("aois", [])
        if aois:
            result: list[str] = []
            for aoi in aois:
                if isinstance(aoi, dict):
                    result.append(str(
                        aoi.get(
                            "id",
                            f"A_{aoi.get('row', 3)}_{aoi.get('col', 4)}",
                        )
                    ))
                else:
                    result.append(str(aoi))
            return result
        return [MILPTaskAllocator._aoi_label(world)]
