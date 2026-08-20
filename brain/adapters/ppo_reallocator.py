"""Brain adapter for PPO-based reconnaissance UAV loss reallocation."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from ..domain.result import AlgorithmResult
from ..domain.task import ReconTask


_PROJECT = Path(__file__).resolve().parents[2]
_PPO_DIR = _PROJECT / "ppo_allocation"
_REGION_CENTERS = {
    0: (12.5, 37.5),
    1: (37.5, 37.5),
    2: (12.5, 12.5),
    3: (37.5, 12.5),
}


class PPOAllocationAdapter:
    """Call PPO ``UAV_DAMAGE`` reallocation for up to four search UAVs."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        device: str = "cpu",
        service: Any | None = None,
    ) -> None:
        self.model_path = (
            model_path
            or os.environ.get("QL_PPO_ALLOCATION_MODEL", "")
        )
        self.device = device
        self._service = service
        self._initialized = False
        self._platform_to_uid: dict[str, int] = {}
        self._uid_to_platform: dict[int, str] = {}
        self._target_to_tid: dict[str, int] = {}
        self._tid_to_target: dict[int, str] = {}
        self._base_tasks: list[ReconTask] = []

    def _ensure_service(self):
        if self._service is not None:
            return self._service
        if not self.model_path:
            raise RuntimeError(
                "PPO model path is not configured; set "
                "QL_PPO_ALLOCATION_MODEL or pass model_path"
            )
        if str(_PPO_DIR) not in sys.path:
            sys.path.insert(0, str(_PPO_DIR))

        # Both MILP and PPO historically use top-level module names such as
        # ``config``.  Temporarily remove an already loaded MILP config while
        # importing the PPO service, then restore it for the rest of Brain.
        saved_config = sys.modules.pop("config", None)
        try:
            from reallocation_service import ReallocationService
        finally:
            if saved_config is not None:
                sys.modules["config"] = saved_config
        self._service = ReallocationService(
            self.model_path, device=self.device
        )
        return self._service

    def _build_scenario(
        self,
        tasks: list[ReconTask],
        context: Any,
    ) -> dict[str, Any]:
        regions = {
            str(rid): {
                "center_x": center[0],
                "center_y": center[1],
                "weather": 0,
                "assigned_uav": -1,
                "need_reassign": False,
                "priority": 1.0,
            }
            for rid, center in _REGION_CENTERS.items()
        }
        uavs = {
            str(uid): {
                "x": 0.0,
                "y": 0.0,
                "sensor": 1,
                "alive": uid in self._uid_to_platform,
                "sensor_failed": False,
                "task": 2,
                "regions": [],
                "target_id": -1,
            }
            for uid in range(4)
        }
        cell_to_region = {
            "c1": 0, "c2": 1, "c3": 2, "c4": 3
        }
        for task in tasks:
            uid = self._platform_to_uid.get(str(task.platform))
            rid = cell_to_region.get(str(task.cell))
            if uid is None or rid is None:
                continue
            regions[str(rid)]["assigned_uav"] = uid
            uavs[str(uid)]["regions"].append(rid)
            uavs[str(uid)]["sensor"] = (
                0 if str(task.sensor) == "EO" else 1
            )
            uavs[str(uid)]["task"] = 0
        for uid, detail in uavs.items():
            if detail["regions"]:
                center = _REGION_CENTERS[detail["regions"][0]]
                detail["x"], detail["y"] = center

        self._target_to_tid = {}
        self._tid_to_target = {}
        world = getattr(context, "world_state", {})
        raw_targets = list(world.get("targets", []) or [])
        targets: dict[str, dict[str, Any]] = {}
        for tid, target in enumerate(raw_targets[:3]):
            external_id = str(target.get("tid", target.get("target_id", tid)))
            self._target_to_tid[external_id] = tid
            self._tid_to_target[tid] = external_id
            pos = target.get("pos", [0.0, 0.0])
            targets[str(tid)] = {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "region": self._target_region(target),
                "target_type": self._target_type_code(str(target.get("type", "AV"))),
                "movable": not bool(target.get("is_fixed", False)),
                "discovered": bool(target.get("confirmed", False)),
                "tracked": bool(target.get("tracked", False)),
                "destroyed": not bool(target.get("alive", True)),
                "tracker_id": -1,
            }
        for tid in range(len(targets), 3):
            targets[str(tid)] = {
                "x": 0.0,
                "y": 0.0,
                "region": 0,
                "target_type": 1,
                "movable": False,
                "discovered": False,
                "tracked": False,
                "destroyed": True,
                "tracker_id": -1,
            }
        aoi = getattr(context, "world_state", {}).get("aoi", {})
        return {
            "scenario_name": (
                f"A_{aoi.get('row', 3)}_{aoi.get('col', 4)}"
            ),
            "description": "Built directly from Brain recon allocation",
            "regions": regions,
            "uavs": uavs,
            "targets": targets,
        }

    @staticmethod
    def _target_type_code(target_type: str) -> int:
        return {"RADAR": 0, "CP": 1, "AV": 2}.get(target_type.upper(), 1)

    @staticmethod
    def _target_region(target: dict[str, Any]) -> int:
        explicit = target.get("region")
        if explicit is not None:
            try:
                return int(str(explicit).lstrip("R"))
            except ValueError:
                return 0
        pos = target.get("pos", [0.0, 0.0])
        x = float(pos[0])
        y = float(pos[1])
        right = x >= 25.0
        top = y >= 25.0
        if top and not right:
            return 0
        if top and right:
            return 1
        if not top and not right:
            return 2
        return 3

    def _translate_result(
        self,
        context: Any,
        result: dict[str, Any],
        *,
        explanation: str,
        removed_platform: str = "",
    ) -> AlgorithmResult:
        translated: list[ReconTask] = [
            task for task in self._base_tasks
            if (
                getattr(task, "role", "") != "subarea_search"
                and str(getattr(task, "platform", "")) != str(removed_platform)
            )
        ]
        for uid_label, detail in result.get("uav_tasks", {}).items():
            internal_uid = int(str(uid_label).lstrip("U"))
            external_id = self._uid_to_platform.get(internal_uid)
            if external_id is None or not detail.get("alive", True):
                continue
            if detail.get("task") != "SEARCH":
                continue
            for region in detail.get("regions", []):
                rid = int(str(region).lstrip("R"))
                translated.append(ReconTask(
                    platform=external_id,
                    cell=f"c{rid + 1}",
                    sensor=str(detail.get("sensor", "SAR")),
                    role="subarea_search",
                    aoi="",
                ))

        context.recon_allocation = translated
        self._base_tasks = translated
        context.world_state["ppo_reallocation"] = {
            "initialized": True,
            "platform_to_uid": dict(self._platform_to_uid),
            "target_to_tid": dict(self._target_to_tid),
            "event_applied": result.get("event_applied", ""),
            "action_detail": result.get("action_detail", ""),
            "region_assignments": result.get("region_assignments", {}),
        }
        return AlgorithmResult.ok({
            "recon_allocation": translated,
            "raw_result": result,
            "explanation": explanation,
        })

    def initialize(
        self,
        context: Any,
        recon_allocation: Any | None = None,
    ) -> AlgorithmResult:
        tasks = list(
            recon_allocation
            if recon_allocation is not None
            else getattr(context, "recon_allocation", None) or []
        )
        subarea_tasks = [
            task for task in tasks
            if getattr(task, "role", "") == "subarea_search"
        ]
        platforms = sorted({
            str(task.platform) for task in subarea_tasks
        })
        if len(platforms) > 4:
            return AlgorithmResult.fail(
                "PPO allocation supports at most four subarea UAVs"
            )
        if not platforms:
            return AlgorithmResult.fail(
                "No subarea_search recon tasks available for PPO"
            )
        self._platform_to_uid = {
            platform: index for index, platform in enumerate(platforms)
        }
        self._uid_to_platform = {
            index: platform
            for platform, index in self._platform_to_uid.items()
        }
        scenario = self._build_scenario(tasks, context)
        try:
            snapshot = self._ensure_service().init(scenario)
        except Exception as exc:
            return AlgorithmResult.fail(
                f"PPO reallocator initialization failed: {exc}"
            )

        self._base_tasks = tasks
        self._initialized = True
        context.world_state["ppo_reallocation"] = {
            "initialized": True,
            "platform_to_uid": dict(self._platform_to_uid),
            "snapshot": snapshot,
        }
        return AlgorithmResult.ok(snapshot)

    def handle_platform_loss(
        self,
        context: Any,
        platform_id: str,
    ) -> AlgorithmResult:
        platform_id = str(platform_id)
        agent = next(
            (
                item for item in getattr(context, "agents", [])
                if str(getattr(item, "pid", "")) == platform_id
            ),
            None,
        )
        if agent is not None and str(agent.type) != "UAV":
            return AlgorithmResult.fail(
                "ppo_allocation only supports reconnaissance UAV loss; "
                f"{platform_id} is {agent.type}"
            )
        if not self._initialized:
            initialized = self.initialize(context)
            if not initialized.success:
                return initialized
        uid = self._platform_to_uid.get(platform_id)
        if uid is None:
            return AlgorithmResult.fail(
                f"Platform {platform_id} is not a PPO subarea UAV"
            )
        try:
            result = self._ensure_service().handle_event({
                "event_type": "UAV_DAMAGE",
                "uav_id": uid,
            })
        except Exception as exc:
            return AlgorithmResult.fail(
                f"PPO UAV_DAMAGE reallocation failed: {exc}"
            )

        if agent is not None:
            agent.lost = True
        translated = self._translate_result(
            context,
            result,
            removed_platform=platform_id,
            explanation=(
                f"{platform_id} 损失后调用 PPO，对搜索子区域进行局部重分配"
            ),
        )
        if translated.success:
            context.world_state["ppo_reallocation"]["lost_platform"] = platform_id
        return translated

    def handle_target_discovered(
        self,
        context: Any,
        platform_id: str,
        target_id: str,
    ) -> AlgorithmResult:
        platform_id = str(platform_id)
        target_id = str(target_id)
        if (
            not self._initialized
            or target_id not in self._target_to_tid
            or platform_id not in self._platform_to_uid
        ):
            initialized = self.initialize(context)
            if not initialized.success:
                return initialized

        uid = self._platform_to_uid.get(platform_id)
        tid = self._target_to_tid.get(target_id)
        if uid is None:
            return AlgorithmResult.fail(
                f"Platform {platform_id} is not a PPO subarea UAV"
            )
        if tid is None:
            return AlgorithmResult.fail(
                f"Target {target_id} is not represented in PPO scenario"
            )
        try:
            result = self._ensure_service().handle_event({
                "event_type": "TARGET_DISCOVERED",
                "uav_id": uid,
                "target_id": tid,
            })
        except Exception as exc:
            return AlgorithmResult.fail(
                f"PPO TARGET_DISCOVERED reallocation failed: {exc}"
            )

        translated = self._translate_result(
            context,
            result,
            explanation=(
                f"{platform_id} 发现 {target_id} 后转跟踪，PPO 重分配剩余搜索区"
            ),
        )
        if translated.success:
            context.world_state["ppo_reallocation"]["discovered_target"] = target_id
        return translated

    def handle_target_destroyed(
        self,
        context: Any,
        target_id: str,
    ) -> AlgorithmResult:
        target_id = str(target_id)
        if not self._initialized or target_id not in self._target_to_tid:
            initialized = self.initialize(context)
            if not initialized.success:
                return initialized

        tid = self._target_to_tid.get(target_id)
        if tid is None:
            return AlgorithmResult.fail(
                f"Target {target_id} is not represented in PPO scenario"
            )
        try:
            result = self._ensure_service().handle_event({
                "event_type": "TARGET_DESTROYED",
                "target_id": tid,
            })
        except Exception as exc:
            return AlgorithmResult.fail(
                f"PPO TARGET_DESTROYED reallocation failed: {exc}"
            )

        translated = self._translate_result(
            context,
            result,
            explanation=f"{target_id} 摧毁后释放跟踪 UAV，PPO 回收搜索能力",
        )
        if translated.success:
            context.world_state["ppo_reallocation"]["destroyed_target"] = target_id
        return translated
