"""任务重分配服务——供其他模块调用的统一接口。

本模块将 PPO 重分配能力包装为可程序化调用的 ReallocationService 类，
其他模块（预分配、仿真执行、路径规划等）通过 import 即可使用，
不需要通过命令行或文件交互。

用法示例:
    from reallocation_service import ReallocationService

    svc = ReallocationService(model_path="results/models/xxx/model.zip")

    # 1. 初始化态势（从预分配模块获取）
    svc.init({
        "regions": {...},
        "uavs":    {...},
        "targets": {...}
    })

    # 2. 发生事件时触发重分配
    result = svc.handle_event({
        "event_type": "UAV_DAMAGE",
        "uav_id": 1
    })

    # 3. 获取结果
    print(result["region_assignments"])  # {"R0": "U2", ...}
    print(result["uav_tasks"])           # {"U0": {...}, ...}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np

from utils.sb3_compat import prepare_sb3_import

prepare_sb3_import()
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from env.uav_env import UAVTaskAllocationEnv
from preallocation_adapter import adapt
from env.uav import UAV
from env.region import Region
from env.target import Target
from env.event import Event
from config import (
    Weather,
    SensorType,
    TaskType,
    TargetType,
    EventType,
    ActionCode,
    NO_UAV,
    NO_TARGET,
    NUM_REGIONS,
)
from policy.action_repair import repair_action


# ---- 事件类型名称 → EventType 枚举 ----
EVENT_TYPE_MAP = {
    "UAV_DAMAGE":       EventType.UAV_DAMAGE,
    "TARGET_DISCOVERED": EventType.REGION_VACANCY,  # 发现目标 → 无人机转入追踪 → 搜索区域空缺
    "TARGET_DESTROYED":  EventType.TARGET_DESTROYED,
    "REGION_VACANCY":    EventType.REGION_VACANCY,
}


class ReallocationService:
    """任务重分配服务。

    职责：接收场景状态 + 突发事件 → PPO 推理 → 输出新的区域-无人机分配。

    注意：
    - 本模块只负责无人机(UAV)的搜索区域重分配
    - 直升机(H)的打击任务分配由预分配模块负责，不经过这里
    """

    # ---- 生命周期 ----

    def __init__(self, model_path: str, device: str = "cpu"):
        """初始化服务，加载训练好的 PPO 模型。

        Args:
            model_path: 模型 .zip 文件路径
            device:     "cpu" 或 "cuda"
        """
        self.model_path = model_path
        self._env = UAVTaskAllocationEnv(random_event_mode=False)
        self._model = MaskablePPO.load(model_path, env=self._env, device=device)

    # ---- 对外接口 ----

    def init(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """加载初始态势（搜索区 -> 子区域 -> 无人机的初始分配）。

        由预分配模块在编队到达搜索区后调用，向重分配模块注册当前状态。

        Args:
            scenario: 场景 dict，格式与 example_target_discovered.json 一致。
                      必须包含 "regions"、"uavs"、"targets" 三个字段。

        Returns:
            当前完整状态快照 (与 get_snapshot() 格式相同)
        """
        self._load_scenario(scenario)
        self._env._get_obs()
        return self._env.snapshot()

    def handle_event(self, event: Dict[str, Any], deterministic: bool = True) -> Dict[str, Any]:
        """注入事件并执行 PPO 重分配，返回新的分配方案。

        这是核心接口。调用方在搜索执行过程中遇到突发事件时调用此方法。

        Args:
            event:         事件 dict，格式见下方 EVENT FORMAT 注释
            deterministic: True=贪婪决策(推荐生产使用), False=采样决策

        Returns:
            dict，包含:
              - region_assignments: 每个区域的分配结果 {"R0": "U2", "R1": "U3", ...}
              - uav_tasks:          每架无人机的任务详情
              - snapshot:           重分配后的完整状态快照
              - event_applied:      实际应用的事件描述
              - action_detail:      每个区域的 PPO 动作解释

        Raises:
            ValueError: 事件类型不支持或参数不合法

        === EVENT FORMAT ===

        支持 4 种事件类型，各有不同的必填字段：

        1. 无人机损毁 (UAV_DAMAGE):
           {"event_type": "UAV_DAMAGE", "uav_id": 1}

        2. 搜索中发现目标 (TARGET_DISCOVERED):
           {"event_type": "TARGET_DISCOVERED", "uav_id": 0, "target_id": 2}

        3. 目标被摧毁 (TARGET_DESTROYED):
           {"event_type": "TARGET_DESTROYED", "target_id": 0}

        4. 区域空缺 (REGION_VACANCY):
           {"event_type": "REGION_VACANCY", "region_id": 3}

        === RETURN FORMAT ===

        {
            "region_assignments": {"R0": "U2", "R1": "U3", "R2": "U0", "R3": null},
            "uav_tasks": {
                "U0": {
                    "alive": true, "task": "SEARCH", "sensor": "SAR",
                    "regions": ["R2"], "target_id": null,
                    "target_points": [[12.5, 12.5]]
                },
                ...
            },
            "snapshot": { ... },         // 完整状态快照（与 get_snapshot() 相同）
            "event_applied": "U1损毁，其2个区域出现空缺",
            "action_detail": "  R0    U2        U1 → U2\\n  ..."
        }
        """
        # 1. 应用事件到环境
        event_type = event.get("event_type", "")
        if event_type not in EVENT_TYPE_MAP:
            raise ValueError(
                f"不支持的事件类型: '{event_type}'。"
                f"可选: {list(EVENT_TYPE_MAP.keys())}"
            )

        description = self._apply_event(event)

        # 2. 记录 PPO 重分配前的分配状态
        old_assignments = {rid: self._env.regions[rid].assigned_uav for rid in range(NUM_REGIONS)}

        # 3. PPO 推理
        masks = get_action_masks(self._env)
        obs = self._env._get_obs()
        raw_action, _ = self._model.predict(obs, deterministic=deterministic, action_masks=masks)

        # 4. 动作修复（确保合法性）+ 执行
        repaired_action, repair_log = repair_action(self._env, raw_action)
        self._env._execute_action(repaired_action)

        # 5. 构建返回结果
        assignment = self._env.export_assignment_json()
        snapshot = self._env.snapshot()
        action_detail = self._build_action_detail(old_assignments, repaired_action)

        assignment["snapshot"] = snapshot
        assignment["event_applied"] = description
        assignment["action_detail"] = action_detail
        assignment["repair_log"] = repair_log

        return assignment

    def get_snapshot(self) -> Dict[str, Any]:
        """获取当前环境完整状态快照。

        可用于：
        - 查询当前区域分配情况
        - 同步状态给其他模块
        - 可视化/日志

        Returns:
            dict，包含 regions, uavs, targets, event 四个键。
            各字段含义参见 example_target_discovered.json 中的注释。
        """
        return self._env.snapshot()

    def get_assignments(self) -> Dict[str, Any]:
        """获取当前分配结果（不重新推理）。

        Returns:
            dict，包含 region_assignments 和 uav_tasks。
            格式与 handle_event() 返回的顶层结构一致（不含 snapshot/event_applied 等）。
        """
        return self._env.export_assignment_json()

    # ============ 内部方法 ============

    def _load_scenario(self, scenario: Dict[str, Any]) -> None:
        """将场景 dict 加载到环境。逻辑与 apply.py:load_scenario() 一致。"""
        self._env.decision_step = 0

        self._env.regions.clear()
        for rid_str, r in scenario["regions"].items():
            rid = int(rid_str)
            self._env.regions[rid] = Region(
                rid=rid,
                center_x=r["center_x"],
                center_y=r["center_y"],
                weather=Weather(r.get("weather", 0)),
                assigned_uav=r.get("assigned_uav", NO_UAV),
                need_reassign=r.get("need_reassign", False),
                priority=r.get("priority", 1.0),
            )

        self._env.uavs.clear()
        for uid_str, u in scenario["uavs"].items():
            uid = int(uid_str)
            self._env.uavs[uid] = UAV(
                uid=uid,
                x=u["x"],
                y=u["y"],
                sensor=SensorType(u.get("sensor", 1)),
                alive=u.get("alive", True),
                sensor_failed=u.get("sensor_failed", False),
                task=TaskType(u.get("task", 0)),
                regions=set(u.get("regions", [])),
                target_id=u.get("target_id", NO_TARGET),
            )

        self._env.targets.clear()
        for tid_str, t in scenario.get("targets", {}).items():
            tid = int(tid_str)
            self._env.targets[tid] = Target(
                tid=tid,
                target_type=TargetType(t.get("target_type", 0)),
                x=t["x"],
                y=t["y"],
                region=t.get("region", 0),
                movable=t.get("movable", False),
                discovered=t.get("discovered", False),
                tracked=t.get("tracked", False),
                destroyed=t.get("destroyed", False),
                tracker_id=t.get("tracker_id", NO_UAV),
            )

        self._env.current_event = Event(
            event_type=EventType.REGION_VACANCY,
            affected_regions=[],
            description="(initialized)",
        )

    def _apply_event(self, event: Dict[str, Any]) -> str:
        """解析事件 dict 并应用到环境，返回事件描述文本。"""
        event_type = event["event_type"]

        if event_type == "UAV_DAMAGE":
            return self._do_uav_damage(event)

        if event_type == "TARGET_DISCOVERED":
            return self._do_target_discovered(event)

        if event_type == "TARGET_DESTROYED":
            return self._do_target_destroyed(event)

        if event_type == "REGION_VACANCY":
            return self._do_region_vacancy(event)

        raise ValueError(f"Unknown event_type: {event_type}")

    def _do_uav_damage(self, e: Dict[str, Any]) -> str:
        uid = int(e["uav_id"])
        u = self._env.uavs[uid]
        affected = list(u.regions)
        u.alive = False
        u.task = TaskType.IDLE
        u.target_id = NO_TARGET
        for rid in affected:
            self._env._clear_region_assignment(rid)
            self._env.regions[rid].need_reassign = True
        self._env.current_event = Event(
            EventType.UAV_DAMAGE, affected, damaged_uav=uid,
            description=f"U{uid}损毁，其{len(affected)}个区域出现空缺"
        )
        return self._env.current_event.description

    def _do_target_discovered(self, e: Dict[str, Any]) -> str:
        uid = int(e["uav_id"])
        tid = int(e["target_id"])
        u = self._env.uavs[uid]
        t = self._env.targets[tid]
        affected = list(u.regions)
        t.discovered = True
        t.tracked = True
        t.tracker_id = uid
        u.regions.clear()
        u.task = TaskType.TRACK
        u.target_id = tid
        for rid in affected:
            self._env.regions[rid].assigned_uav = NO_UAV
            self._env.regions[rid].need_reassign = True
        self._env.current_event = Event(
            EventType.REGION_VACANCY, affected,
            description=f"U{uid}发现T{tid}，转入TRACK，其{len(affected)}个搜索区域出现空缺"
        )
        return self._env.current_event.description

    def _do_target_destroyed(self, e: Dict[str, Any]) -> str:
        tid = int(e["target_id"])
        t = self._env.targets[tid]
        tracker = int(t.tracker_id)
        t.destroyed = True
        t.tracked = False
        t.tracker_id = NO_UAV
        u = self._env.uavs[tracker]
        if u.alive:
            u.task = TaskType.IDLE
            u.target_id = NO_TARGET
        self._env.current_event = Event(
            EventType.TARGET_DESTROYED, [], released_uav=tracker,
            description=f"T{tid}被摧毁，U{tracker}释放，可重新加入搜索"
        )
        return self._env.current_event.description

    def _do_region_vacancy(self, e: Dict[str, Any]) -> str:
        rid = int(e["region_id"])
        self._env._clear_region_assignment(rid)
        self._env.regions[rid].need_reassign = True
        self._env.current_event = Event(
            EventType.REGION_VACANCY, [rid],
            description=f"R{rid}区域出现空缺，需PPO重新分配"
        )
        return self._env.current_event.description

    @staticmethod
    def _build_action_detail(old_assignments: Dict[int, int], repaired_action) -> str:
        """构建每个区域的 PPO 动作可读说明。"""
        action_names = {
            ActionCode.KEEP: "KEEP",
            ActionCode.U0: "U0", ActionCode.U1: "U1",
            ActionCode.U2: "U2", ActionCode.U3: "U3",
            ActionCode.NO_UAV: "NO_UAV",
        }
        lines = ["Region   PPO决策         变化", "-" * 42]
        for rid in range(NUM_REGIONS):
            code = int(repaired_action[rid])
            name = action_names.get(code, f"??({code})")
            old_uid = old_assignments.get(rid, NO_UAV)
            old_name = f"U{old_uid}" if old_uid != NO_UAV else "UNASSIGNED"
            if code == ActionCode.KEEP:
                change = "(不变)"
            elif code == ActionCode.NO_UAV:
                change = f"{old_name} → UNASSIGNED"
            else:
                change = f"{old_name} → U{code - 1}"
            lines.append(f"  R{rid}    {name:<8}    {change}")
        return "\n".join(lines)


# ============================================================
# C++ 风格接口 —— 统一请求/响应 JSON 合约
# ============================================================


def reallocate_cpp_interface(request: Any) -> Dict[str, Any]:
    """C++ 友好接口：接收请求字典/JSON 字符串，返回统一 JSON 响应。

    请求字段：
    - model_path: str, 必填
    - preallocation_path: str, 可选（与 preallocation_json 二选一）
    - preallocation_json: dict|str, 可选（与 preallocation_path 二选一）
    - event: dict|str, 可选（与 event_path 二选一）
    - event_path: str, 可选（与 event 二选一）
    - targets_extra: dict|str, 可选
    - cell_to_region: dict|str, 可选
    - output_path: str, 可选
    - deterministic: bool, 可选，默认 True

    返回字段：
    - success: bool
    - message: str
    - output_path: str | None
    - result: dict | None
    - error: str | None
    """
    if isinstance(request, str):
        try:
            request = json.loads(request)
        except json.JSONDecodeError as exc:
            raise ValueError(f"请求 JSON 格式非法: {exc}") from exc

    if not isinstance(request, dict):
        raise TypeError("request 必须是 dict 或 JSON 字符串")

    model_path = request.get("model_path")
    if not model_path:
        raise ValueError("缺少 model_path")

    preallocation_path = request.get("preallocation_path")
    preallocation_json = request.get("preallocation_json")
    if preallocation_path:
        pre_path = Path(preallocation_path)
        if not pre_path.exists():
            raise FileNotFoundError(f"预分配文件不存在: {pre_path}")
        prealloc = json.loads(pre_path.read_text(encoding="utf-8"))
    elif preallocation_json is not None:
        if isinstance(preallocation_json, str):
            prealloc = json.loads(preallocation_json)
        elif isinstance(preallocation_json, dict):
            prealloc = preallocation_json
        else:
            raise TypeError("preallocation_json 必须是 dict 或 JSON 字符串")
    else:
        raise ValueError("必须提供 preallocation_path 或 preallocation_json")

    event = request.get("event")
    event_path = request.get("event_path")
    if event is None and event_path is None:
        raise ValueError("必须提供 event 或 event_path")
    if event is not None and event_path is not None:
        raise ValueError("event 和 event_path 只能提供一个")

    if event is None and event_path:
        ep = Path(event_path)
        if not ep.exists():
            raise FileNotFoundError(f"事件文件不存在: {ep}")
        event = json.loads(ep.read_text(encoding="utf-8"))
    elif isinstance(event, str):
        event = json.loads(event)

    targets_extra = request.get("targets_extra")
    if isinstance(targets_extra, str):
        targets_extra = json.loads(targets_extra)

    cell_to_region = request.get("cell_to_region")
    if isinstance(cell_to_region, str):
        cell_to_region = json.loads(cell_to_region)

    deterministic = request.get("deterministic", True)
    output_path = request.get("output_path")

    scenario = adapt(prealloc, targets_extra=targets_extra, cell_to_region=cell_to_region)
    svc = ReallocationService(model_path)
    svc.init(scenario)
    result = svc.handle_event(event, deterministic=deterministic)

    if output_path is None:
        output_path = "reallocation_result.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "aoi_id": prealloc.get("aoi_id", ""),
        "event": result["event_applied"],
        "region_assignments": result["region_assignments"],
        "uav_tasks": result["uav_tasks"],
        "action_detail": result["action_detail"],
        "repair_log": result["repair_log"],
        "snapshot": result["snapshot"],
    }
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "success": True,
        "message": "重分配完成",
        "output_path": str(output_path.resolve()),
        "result": output_data,
        "error": None,
    }


def reallocate_from_preallocation_cpp(request: Any) -> str:
    """C++ 调用友好的 JSON 字符串接口。

    直接接收请求对象或 JSON 字符串，返回 JSON 字符串响应。
    """
    response = reallocate_cpp_interface(request)
    return json.dumps(response, ensure_ascii=False, indent=2)


# ============================================================
# 文件式接口 —— 供其他模块通过 JSON 文件对接
# ============================================================

def reallocate_from_files(
    model_path: str,
    scenario_path: str,
    event: Dict[str, Any] = None,
    event_path: str = None,
    output_path: str = None,
) -> str:
    """文件式重分配接口：读 JSON → PPO 推理 → 写 JSON，返回输出路径。

    这是供其他模块通过文件系统对接的便捷入口。
    其他模块只需准备一个场景 JSON + 一个事件 JSON，
    调用本函数即可获得重分配结果 JSON。

    Args:
        model_path:   PPO 模型 .zip 路径
        scenario_path: 场景 JSON 文件路径（格式同 example_target_discovered.json）
        event:         事件 dict（同 handle_event 格式）。与 event_path 二选一
        event_path:    事件 JSON 文件路径。格式 {"event_type":"UAV_DAMAGE","uav_id":1}
        output_path:   输出 JSON 路径。默认写到 <scenario所在目录>/reallocation_result.json

    Returns:
        str: 输出 JSON 文件的绝对路径

    Raises:
        FileNotFoundError: 场景或事件文件不存在
        ValueError:        事件格式不合法

    用法示例:
        # 方式1：事件以 dict 传入
        out = reallocate_from_files(
            model_path="results/models/xxx/model.zip",
            scenario_path="scenarios/example_target_discovered.json",
            event={"event_type": "UAV_DAMAGE", "uav_id": 1},
        )

        # 方式2：事件从 JSON 文件读取
        out = reallocate_from_files(
            model_path="results/models/xxx/model.zip",
            scenario_path="scenarios/example_target_discovered.json",
            event_path="scenarios/event_uav_damage.json",
        )

        # 读取结果
        result = json.loads(Path(out).read_text(encoding="utf-8"))
        print(result["region_assignments"])
    """
    # ---- 校验参数 ----
    if event is None and event_path is None:
        raise ValueError("必须提供 event 或 event_path 之一")
    if event is not None and event_path is not None:
        raise ValueError("event 和 event_path 只能提供一个")

    scenario_path = Path(scenario_path)
    if not scenario_path.exists():
        raise FileNotFoundError(f"场景文件不存在: {scenario_path}")

    # ---- 读取输入 ----
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))

    if event_path:
        event_path = Path(event_path)
        if not event_path.exists():
            raise FileNotFoundError(f"事件文件不存在: {event_path}")
        event = json.loads(event_path.read_text(encoding="utf-8"))

    # ---- 确定输出路径 ----
    if output_path is None:
        output_path = scenario_path.parent / "reallocation_result.json"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 执行重分配 ----
    svc = ReallocationService(model_path)
    svc.init(scenario)
    result = svc.handle_event(event)

    # ---- 写入输出 ----
    output_data = {
        "scenario_name": scenario.get("scenario_name", ""),
        "event": result["event_applied"],
        "region_assignments": result["region_assignments"],
        "uav_tasks": result["uav_tasks"],
        "action_detail": result["action_detail"],
        "repair_log": result["repair_log"],
        "snapshot": result["snapshot"],
    }
    output_path.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[重分配完成] 结果: {output_path}")

    return str(output_path.resolve())


# ============================================================
# 从预分配输出直接调用（一站式入口）
# ============================================================

def reallocate_from_preallocation(
    model_path: str,
    preallocation_path: str,
    event: Dict[str, Any] = None,
    event_path: str = None,
    targets_extra: Dict[str, Dict[str, Any]] = None,
    cell_to_region: Dict[str, int] = None,
    output_path: str = None,
) -> str:
    """从预分配输出直接重分配：预分配JSON → 适配 → PPO推理 → 输出JSON。

    兼容旧调用方式，同时由新的 C++ 风格接口内部实现。
    """
    request = {
        "model_path": model_path,
        "preallocation_path": preallocation_path,
        "event": event,
        "event_path": event_path,
        "targets_extra": targets_extra,
        "cell_to_region": cell_to_region,
        "output_path": output_path,
        "deterministic": True,
    }
    response = reallocate_cpp_interface(request)
    if not response["success"]:
        raise RuntimeError(response["error"])
    return response["output_path"]
