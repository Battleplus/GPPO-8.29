"""任务预分配输出 → PPO 重分配场景 适配器。

将 output_template.json 格式转换为 example_target_discovered.json 格式。
regions坐标硬编码(c1~c4对应4象限固定值)，uav坐标自动跟随region中心。
targets坐标/类型填默认值（PPO重分配不依赖）。

用法:
    import json
    from preallocation_adapter import adapt

    prealloc = json.load(open("scenarios/output_template.json"))
    scenario = adapt(prealloc)  # 不需要额外参数
"""

from __future__ import annotations

from typing import Dict, Any, List
from config import NUM_TARGETS, REGION_CENTERS

# ---- 区域坐标硬编码（c1~c4 对应 4 象限，固定地图布局） ----
_REGION_CENTERS = {rid: REGION_CENTERS[rid] for rid in range(4)}


_SENSOR_MAP = {
    "EO": 0, "SAR": 1,
    "ESM": 1, "MMW": 1, "EOIR": 0,
}

_DEFAULT_CELL_TO_REGION = {"c1": 0, "c2": 1, "c3": 2, "c4": 3}


def adapt(
    preallocation: Dict[str, Any],
    *,
    targets_extra: Dict[str, Dict[str, Any]] = None,
    cell_to_region: Dict[str, int] = None,
    scenario_name: str = None,
) -> Dict[str, Any]:
    """将预分配输出转换为 PPO 场景 dict。

    Args:
        preallocation:  output_template.json 格式的 dict
        targets_extra:  目标坐标/类型补充。仅在 TARGET_DISCOVERED/TARGET_DESTROYED
                        事件时需要（事件本身会告知 uav_id/target_id，此处只需创建目标槽位）。
                        UAV_DAMAGE/REGION_VACANCY 事件不需要此参数。
        cell_to_region: cell→region 自定义映射，默认 c1→R0, c2→R1, c3→R2, c4→R3
        scenario_name:  场景名称，默认取 aoi_id

    Returns:
        PPO 场景 dict，格式与 example_target_discovered.json 一致。
        regions坐标硬编码（c1~c4对应固定4象限），uav坐标跟随region中心。
    """
    if targets_extra is None:
        targets_extra = {}
    if cell_to_region is None:
        cell_to_region = _DEFAULT_CELL_TO_REGION
    tasks = preallocation.get("tasks", [])
    aoi_id = preallocation.get("aoi_id", "unknown")

    # ============================================================
    # 1. 提取 subarea_search 任务 → {cell: {platform, sensor}}
    # ============================================================
    cell_assignments: Dict[str, Dict[str, str]] = {}
    for t in tasks:
        if t.get("task_type") != "recon":
            continue
        if t.get("role") != "subarea_search":
            continue
        cell_assignments[t.get("cell", "")] = {
            "platform": t.get("platform", ""),
            "sensor": t.get("sensor_used", "SAR"),
        }

    # ============================================================
    # 2. 平台名 → PPO内部编号  (U2→0, U3→1, U4→2, U5→3)
    # ============================================================
    subarea_platforms = sorted(
        set(a["platform"] for a in cell_assignments.values()),
        key=lambda p: int(p[1:])
    )
    platform_to_uid: Dict[str, int] = {}
    uid_to_platform: Dict[int, str] = {}
    for idx, plat in enumerate(subarea_platforms):
        if idx >= 4:
            break
        platform_to_uid[plat] = idx
        uid_to_platform[idx] = plat

    # ============================================================
    # 3. 构建 regions — 坐标硬编码（c1~c4 对应 4 象限，地图布局固定）
    # ============================================================
    regions: Dict[str, Dict[str, Any]] = {}
    for rid in range(4):
        cx, cy = _REGION_CENTERS[rid]
        regions[str(rid)] = {
            "center_x": cx,
            "center_y": cy,
            "weather": 0,
            "assigned_uav": -1,
            "need_reassign": False,
            "priority": 1.0,
        }

    # ============================================================
    # 4. 构建 uavs — x/y 后面根据分配的 region 中心填充
    # ============================================================
    uavs: Dict[str, Dict[str, Any]] = {}
    for uid in range(4):
        plat = uid_to_platform.get(uid)
        if plat is None:
            uavs[str(uid)] = {
                "x": 0.0, "y": 0.0,
                "sensor": 1,
                "alive": True, "sensor_failed": False,
                "task": 2, "regions": [], "target_id": -1,
            }
        else:
            uavs[str(uid)] = {
                "x": 0.0, "y": 0.0,
                "sensor": 1,
                "alive": True, "sensor_failed": False,
                "task": 0, "regions": [], "target_id": -1,
            }

    # ============================================================
    # 5. 填充 region ↔ UAV 关联 + UAV坐标跟随region中心
    # ============================================================
    for cell_name, assign in cell_assignments.items():
        rid = cell_to_region.get(cell_name)
        if rid is None:
            continue

        plat = assign["platform"]
        uid = platform_to_uid.get(plat)
        if uid is None or uid >= 4:
            continue

        sensor_val = _SENSOR_MAP.get(assign["sensor"], 1)

        regions[str(rid)]["assigned_uav"] = uid
        uavs[str(uid)]["regions"].append(rid)
        uavs[str(uid)]["sensor"] = sensor_val

    # UAV 坐标 = 其负责的第一个 region 中心
    for uid_str, u in uavs.items():
        if u["regions"]:
            first_rid = u["regions"][0]
            cx, cy = _REGION_CENTERS[first_rid]
            u["x"] = cx
            u["y"] = cy
        else:
            u["task"] = 2  # 无 region → IDLE

    # ============================================================
    # 6. 构建 targets
    #    PPO重分配不依赖目标坐标/类型，只依赖 discovered/tracked/destroyed/tracker_id。
    #    预分配输出只有目标名字("g1")，其余全部填 0.0 安全默认值。
    #    只有用到 TARGET_DISCOVERED/TARGET_DESTROYED 事件时才需要 targets_extra。
    # ============================================================
    strike_target_names: List[str] = []
    for t in tasks:
        if t.get("task_type") == "strike" and t.get("target"):
            name = t["target"]
            if name not in strike_target_names:
                strike_target_names.append(name)

    targets: Dict[str, Dict[str, Any]] = {}
    for idx, tgt_name in enumerate(sorted(strike_target_names)):
        extra = targets_extra.get(tgt_name, {}) if targets_extra else {}
        targets[str(idx)] = {
            "x":          extra.get("x", 0.0),
            "y":          extra.get("y", 0.0),
            "region":     extra.get("region", 0),
            "target_type": extra.get("target_type", 1),
            "movable":    extra.get("movable", False),
            "discovered": extra.get("discovered", False),
            "tracked":    extra.get("tracked", False),
            "destroyed":  extra.get("destroyed", False),
            "tracker_id": extra.get("tracker_id", -1),
        }

    # 补齐到 NUM_TARGETS
    while len(targets) < NUM_TARGETS:
        dummy_id = str(len(targets))
        targets[dummy_id] = {
            "x": 0.0, "y": 0.0, "region": 0,
            "target_type": 1, "movable": False,
            "discovered": False, "tracked": False,
            "destroyed": True, "tracker_id": -1,
            "_note": f"占位目标#{dummy_id}（预分配无此目标，自动补齐，destroyed=true不影响PPO）",
        }

    # ============================================================
    # 7. 组装返回
    # ============================================================
    return {
        "scenario_name": scenario_name or aoi_id,
        "description": (
            f"从预分配输出自动转换。"
            f"映射: {uid_to_platform} → U0~U3, "
            f"cells={list(cell_assignments.keys())} → R0~R3。"
            f"regions坐标硬编码(c1~c4对应4象限)，uav坐标跟随region中心。"
        ),
        "regions": regions,
        "uavs": uavs,
        "targets": targets,
        "_field_coverage": {
            "from_preallocation（从预分配输出直接提取）": [
                "regions.*.assigned_uav  ← tasks[role=subarea_search].platform + cell映射",
                "uavs.*.sensor           ← tasks[role=subarea_search].sensor_used",
                "uavs.*.task             ← task_type=recon → SEARCH(0)",
                "uavs.*.regions          ← tasks.cell → region编号",
                "scenario_name           ← aoi_id",
            ],
            "hardcoded（已知地图布局，硬编码）": [
                "regions.*.center_x/y    ← config.py REGION_CENTERS（c1~c4对应4象限固定坐标）",
                "uavs.*.x/y              ← 自动跟随所分配region的中心坐标",
            ],
            "auto_filled_default（PPO不依赖，自动填默认值）": [
                "regions.*.weather=0, need_reassign=false, priority=1.0",
                "uavs.*.alive=true, sensor_failed=false",
                "targets.*.x/y/region/target_type ← 0.0/0/1（仅做占位，事件系统提供真实值）",
            ],
            "ignored（预分配输出有但PPO不需要）": [
                "tasks[role=area_scan]   c0广域扫描",
                "tasks[role=esm_patrol]  ESM巡逻",
                "tasks[task_type=strike]  直升机打击任务",
                "cycle_id/timestamp/objective/solve_time_ms",
            ],
        },
    }
