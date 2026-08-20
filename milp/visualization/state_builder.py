"""
状态构建器 —— 纯函数：snapshot + plan → 可序列化 viz dict。

本模块是后端（MILP）与前端（Streamlit）之间唯一的数据桥梁。
函数对入参 snapshot、plan 只读，绝不原地修改；所有 numpy 数组通过
.tolist() 拷贝输出，保证 dict 可直接 json.dumps 序列化。

设计原则:
  - 纯函数，无副作用
  - 输出 dict 全部为 list / float / str / bool 等 JSON 原生类型
  - 前端只认此 dict，不反向依赖后端 dataclass
"""

from __future__ import annotations
import re
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from visualization.constants import N_ANIM_STEPS, AOI_SIZE_KM


# ── 路径排序 ──────────────────────────────────────────

def _order_waypoints(start: Tuple[float, float],
                     points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """从 start 出发的最近邻排序，让折线更直观，不追求最优 TSP。"""
    remaining = list(points)
    order, cur = [], start
    while remaining:
        nxt = min(remaining, key=lambda p: (p[0] - cur[0]) ** 2 + (p[1] - cur[1]) ** 2)
        order.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    return order


# ── AOI 解析 ──────────────────────────────────────────

def _parse_aoi_bbox(aoi_str: str) -> Optional[Tuple[float, float, float, float]]:
    """解析 "A_r_c" 格式的 AOI 标识，返回包围盒 [x0, y0, x1, y1] (km)。"""
    m = re.match(r"^A_(\d+)_(\d+)$", aoi_str)
    if not m:
        return None
    row, col = int(m.group(1)), int(m.group(2))
    if not (1 <= row <= 6 and 1 <= col <= 6):
        return None
    x0 = (col - 1) * 50.0
    y0 = (row - 1) * 50.0
    return (x0, y0, x0 + 50.0, y0 + 50.0)


# ── 弧长参数化 ─────────────────────────────────────────

def _cumulative_arc_lengths(waypoints: List[Tuple[float, float]]) -> List[float]:
    """计算航点序列的累计弧长，归一化到 [0, 1] 区间的 s_k 列表。"""
    if len(waypoints) < 2:
        return [0.0]
    segs = []
    for i in range(len(waypoints) - 1):
        dx = waypoints[i + 1][0] - waypoints[i][0]
        dy = waypoints[i + 1][1] - waypoints[i][1]
        segs.append(math.sqrt(dx * dx + dy * dy))
    total = sum(segs)
    if total < 1e-9:
        return [0.0] * len(waypoints)
    cum = [0.0]
    for s in segs:
        cum.append(cum[-1] + s / total)
    return cum


def _interpolate_route(waypoints: List[Tuple[float, float]], tau: float) -> Tuple[float, float]:
    """在一条折线路径上按 tau ∈ [0,1] 线性插值当前位置。"""
    if len(waypoints) == 1:
        return waypoints[0]
    s_vals = _cumulative_arc_lengths(waypoints)
    tau = max(0.0, min(1.0, tau))
    for k in range(len(s_vals) - 1):
        if tau <= s_vals[k + 1] + 1e-12:
            seg_len = s_vals[k + 1] - s_vals[k]
            if seg_len < 1e-12:
                return waypoints[k]
            alpha = (tau - s_vals[k]) / seg_len
            x = waypoints[k][0] + alpha * (waypoints[k + 1][0] - waypoints[k][0])
            y = waypoints[k][1] + alpha * (waypoints[k + 1][1] - waypoints[k][1])
            return (x, y)
    return waypoints[-1]


# ── 空计划工厂 ────────────────────────────────────────

def _empty_plan_dict() -> Dict[str, Any]:
    """未求解时的空计划占位 dict（状态 UNSOLVED）。"""
    return {
        "meta": {
            "map_size_km": 300.0,
            "commander_aoi": "",
            "aoi_bbox": None,
            "staging": [150.0, -50.0],
            "status": "UNSOLVED",
            "objective": 0.0,
            "solve_time_ms": 0.0,
            "solver_used": "",
        },
        "platforms": [],
        "grids": [],
        "targets": [],
        "recon_assignments": [],
        "strike_assignments": [],
        "routes": [],
        "animation_frames": [],
    }


# ── 主函数 ────────────────────────────────────────────

def build_visualization_state(snapshot, plan) -> Dict[str, Any]:
    """
    纯函数：将 snapshot + plan 转为前端可消费的 viz dict。

    对 snapshot、plan 只读；所有坐标通过 .tolist() 拷贝输出。
    输出 dict 全部为 JSON 原生类型，可直接 json.dumps。

    Args:
        snapshot: SituationSnapshot
        plan: AllocationPlan (或 dict，未求解时用 _empty_plan_dict 占位)

    Returns:
        可序列化的 viz dict，结构见方案 §3.2
    """
    # ── 处理未求解情况（plan 为 dict） ────────────────
    if isinstance(plan, dict):
        # 仅用 snapshot 构建 t0 预览
        staging = snapshot.staging_position
        if staging is None and snapshot.platforms:
            staging = snapshot.platforms[0].pos.copy()
        staging_xy = staging.tolist() if hasattr(staging, 'tolist') else (
            list(staging) if staging is not None else [0.0, 0.0]
        )
        pid_to_pos_xy: Dict[str, List[float]] = {}
        for p in snapshot.platforms:
            pos = p.pos
            pid_to_pos_xy[p.pid] = pos.tolist() if hasattr(pos, 'tolist') else list(pos)

        platforms = []
        for p in snapshot.platforms:
            xy = pid_to_pos_xy[p.pid]
            platforms.append({
                "pid": p.pid, "type": p.type,
                "start": xy,
                "sensors_mounted": list(p.sensors_mounted),
            })

        grids = []
        for g in snapshot.grids:
            grids.append({
                "cell_id": g.cell_id,
                "center": g.center.tolist() if hasattr(g.center, 'tolist') else list(g.center),
                "w": g.width_km, "h": g.height_km,
                "weather_w": g.weather_w,
            })

        targets = []
        for t in snapshot.targets:
            targets.append({
                "tid": t.tid, "type": t.type,
                "pos": t.pos_est.tolist() if hasattr(t.pos_est, 'tolist') else list(t.pos_est),
                "value": t.value, "threat": t.threat,
            })

        aoi_str = snapshot.commander_AOI[0] if snapshot.commander_AOI else ""
        aoi_bbox = _parse_aoi_bbox(aoi_str) if aoi_str else None

        result = {
            "meta": {
                "map_size_km": 300.0,
                "commander_aoi": aoi_str,
                "aoi_bbox": list(aoi_bbox) if aoi_bbox else None,
                "staging": staging_xy,
                "status": "UNSOLVED",
                "objective": 0.0,
                "solve_time_ms": 0.0,
                "solver_used": "",
            },
            "platforms": platforms,
            "grids": grids,
            "targets": targets,
            "recon_assignments": [],
            "strike_assignments": [],
            "routes": [],
            "animation_frames": [],
        }

        # 生成 t0 帧（各平台在自身位置）
        t0_positions = [{"pid": pl["pid"], "xy": pid_to_pos_xy[pl["pid"]], "type": pl["type"]}
                        for pl in platforms]
        result["animation_frames"] = [
            {"phase": "t0", "tau": 0.0, "positions": t0_positions},
        ]
        return result

    # ── 正常求解路径 ───────────────────────────────────
    # 1. meta
    staging = snapshot.staging_position
    if staging is None and snapshot.platforms:
        staging = snapshot.platforms[0].pos.copy()
    staging_xy: List[float] = staging.tolist() if hasattr(staging, 'tolist') else (
        list(staging) if staging is not None else [0.0, 0.0]
    )
    pid_to_pos_xy: Dict[str, List[float]] = {}
    for p in snapshot.platforms:
        pos = p.pos
        pid_to_pos_xy[p.pid] = pos.tolist() if hasattr(pos, 'tolist') else list(pos)

    aoi_str: str = snapshot.commander_AOI[0] if snapshot.commander_AOI else ""
    aoi_bbox = _parse_aoi_bbox(aoi_str) if aoi_str else None

    meta = {
        "map_size_km": 300.0,
        "commander_aoi": aoi_str,
        "aoi_bbox": list(aoi_bbox) if aoi_bbox else None,
        "staging": staging_xy,
        "status": plan.status,
        "objective": plan.objective,
        "solve_time_ms": plan.solve_time_ms,
        "solver_used": plan.solver_used,
    }

    # 2. platforms（各平台在自身位置 + 搭载传感器）
    platforms = []
    for p in snapshot.platforms:
        xy = pid_to_pos_xy[p.pid]
        platforms.append({
            "pid": p.pid, "type": p.type,
            "start": xy,
            "sensors_mounted": plan.mounted_sensors.get(p.pid, list(p.sensors_mounted)),
        })

    # 3. grids
    cell_center: Dict[str, List[float]] = {}
    grids = []
    for g in snapshot.grids:
        center = g.center.tolist() if hasattr(g.center, 'tolist') else list(g.center)
        cell_center[g.cell_id] = center
        grids.append({
            "cell_id": g.cell_id,
            "center": center,
            "w": g.width_km, "h": g.height_km,
            "weather_w": g.weather_w,
        })

    # 4. targets
    target_pos: Dict[str, List[float]] = {}
    targets = []
    for t in snapshot.targets:
        pos = t.pos_est.tolist() if hasattr(t.pos_est, 'tolist') else list(t.pos_est)
        target_pos[t.tid] = pos
        targets.append({
            "tid": t.tid, "type": t.type,
            "pos": pos,
            "value": t.value, "threat": t.threat,
        })

    # 5. recon_assignments
    recon_assignments = []
    for ra in plan.recon_assignments:
        recon_assignments.append({
            "pid": ra.pid, "sensor": ra.sensor_used,
            "sensors_mounted": ra.sensors_mounted,
            "cell": ra.cell, "role": ra.role,
        })

    # 6. strike_assignments
    strike_assignments = []
    for sa in plan.strike_assignments:
        strike_assignments.append({
            "pid": sa.pid, "target": sa.target,
            "munition": sa.munition, "qty": sa.qty, "role": sa.role,
            "sensors_mounted": plan.mounted_sensors.get(sa.pid, []),
        })

    # 7. routes —— 核心路径生成
    # 按 pid 收集 UAV 分配
    uav_cells: Dict[str, List[str]] = {}
    for ra in plan.recon_assignments:
        pid = ra.pid
        ptype = "UAV"  # recon 只有 UAV
        if pid not in uav_cells:
            uav_cells[pid] = []
        if ra.cell not in uav_cells[pid]:
            uav_cells[pid].append(ra.cell)

    # 按 pid 收集 HELI 分配（不过滤 support，只要有 target 就飞过去）
    heli_targets: Dict[str, List[str]] = {}
    for sa in plan.strike_assignments:
        pid = sa.pid
        if pid not in heli_targets:
            heli_targets[pid] = []
        if sa.target not in heli_targets[pid]:
            heli_targets[pid].append(sa.target)

    # 构建每条 route
    routes: List[Dict[str, Any]] = []
    for p in snapshot.platforms:
        pid = p.pid
        ptype = p.type
        start_pos = tuple(pid_to_pos_xy[pid])
        waypoints: List[Tuple[float, float]] = [start_pos]
        labels: List[str] = ["start"]

        if ptype == "UAV" and pid in uav_cells:
            cell_coords = [tuple(cell_center[c]) for c in uav_cells[pid]
                           if c in cell_center]
            ordered = _order_waypoints(start_pos, cell_coords)
            for coord in ordered:
                waypoints.append(coord)
                for cid, cxy in cell_center.items():
                    if abs(cxy[0] - coord[0]) < 0.01 and abs(cxy[1] - coord[1]) < 0.01:
                        labels.append(cid)
                        break
                else:
                    labels.append("?")

        elif ptype == "HELI" and pid in heli_targets:
            tgt_coords = [tuple(target_pos[t]) for t in heli_targets[pid]
                          if t in target_pos]
            ordered = _order_waypoints(start_pos, tgt_coords)
            for coord in ordered:
                waypoints.append(coord)
                for tid, txy in target_pos.items():
                    if abs(txy[0] - coord[0]) < 0.01 and abs(txy[1] - coord[1]) < 0.01:
                        labels.append(tid)
                        break
                else:
                    labels.append("?")

        routes.append({
            "pid": pid, "type": ptype,
            "waypoints": [list(w) for w in waypoints],
            "labels": labels,
        })

    # 8. animation_frames
    animation_frames: List[Dict[str, Any]] = []

    # t0: 各平台在自身位置
    t0_positions = [{"pid": pl["pid"], "xy": pid_to_pos_xy[pl["pid"]], "type": pl["type"]}
                    for pl in platforms]
    animation_frames.append({"phase": "t0", "tau": 0.0, "positions": t0_positions})

    # t1: 同位置，前端显示分配线
    t1_positions = [{"pid": pl["pid"], "xy": pid_to_pos_xy[pl["pid"]], "type": pl["type"]}
                    for pl in platforms]
    animation_frames.append({"phase": "t1", "tau": 0.0, "positions": t1_positions})

    # 为插值建立 route 索引
    route_map: Dict[str, Dict] = {r["pid"]: r for r in routes}

    # t2: N_ANIM_STEPS 帧线性插值
    for i in range(1, N_ANIM_STEPS + 1):
        tau = i / N_ANIM_STEPS
        positions = []
        for pl in platforms:
            r = route_map.get(pl["pid"])
            if r and len(r["waypoints"]) >= 2:
                xy = _interpolate_route(
                    [tuple(w) for w in r["waypoints"]], tau)
            else:
                xy = tuple(pid_to_pos_xy[pl["pid"]])
            positions.append({
                "pid": pl["pid"],
                "xy": [xy[0], xy[1]],
                "type": pl["type"],
            })
        animation_frames.append({
            "phase": "t2", "tau": tau, "positions": positions,
        })

    # t3: 全部到位
    t3_positions = []
    for pl in platforms:
        r = route_map.get(pl["pid"])
        if r and len(r["waypoints"]) >= 1:
            final = r["waypoints"][-1]
        else:
            final = pid_to_pos_xy[pl["pid"]]
        t3_positions.append({
            "pid": pl["pid"],
            "xy": final,
            "type": pl["type"],
        })
    animation_frames.append({"phase": "t3", "tau": 1.0, "positions": t3_positions})

    return {
        "meta": meta,
        "platforms": platforms,
        "grids": grids,
        "targets": targets,
        "recon_assignments": recon_assignments,
        "strike_assignments": strike_assignments,
        "routes": routes,
        "animation_frames": animation_frames,
    }


# ── 多 AOI 可视化状态构建器 ───────────────────────────────

def build_multi_aoi_visualization_state(
    input_data: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    将多 AOI 输入 + controller.run() 输出转为前端可消费的 viz dict。

    Args:
        input_data: 传入 MultiAOIController.run() 的原始输入 dict
        result:     run() 返回的输出 dict

    Returns:
        可序列化的 viz dict，含 "aois"、"aoi_route" 等多 AOI 专用字段
    """
    # ── 解析 AOI 列表 ──────────────────────────────────
    aois_raw = input_data.get("aois", [])
    aoi_route_state = result.get("aoi_route_state", {})
    aoi_sequence = aoi_route_state.get("aoi_sequence", [])
    current_aoi = aoi_route_state.get("current_aoi")
    current_idx = aoi_route_state.get("current_aoi_index", 0)

    # 建立 AOI id → raw info 映射
    aoi_map: Dict[str, Dict] = {a["id"]: a for a in aois_raw}

    # 构建多 AOI 可视化数据
    aois_viz: List[Dict[str, Any]] = []
    aoi_centers: List[List[float]] = []

    all_finished = result.get("status") == "ALL_AOI_FINISHED"

    for order, aoi_id in enumerate(aoi_sequence, start=1):
        info = aoi_map.get(aoi_id, {})
        row = int(info.get("row", 1))
        col = int(info.get("col", 1))
        x0 = (col - 1) * AOI_SIZE_KM
        y0 = (row - 1) * AOI_SIZE_KM
        x1 = x0 + AOI_SIZE_KM
        y1 = y0 + AOI_SIZE_KM
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        bbox = [x0, y0, x1, y1]
        center = [cx, cy]
        aoi_centers.append(center)

        # 判断状态
        if all_finished:
            status = "FINISHED"
        elif aoi_id == current_aoi:
            status = "CURRENT"
        elif order <= current_idx:
            status = "FINISHED"
        else:
            status = "PENDING"

        aois_viz.append({
            "id": aoi_id,
            "row": row,
            "col": col,
            "bbox": bbox,
            "center": center,
            "status": status,
            "order": order,
            "priority": info.get("priority", 0),
            "target_value": info.get("target_value", 0),
            "target_threat": info.get("target_threat", 0),
            "target_prior": info.get("target_prior", 0),
        })

    # ── 计算有效出发点（AOI 间不返回集结区）──────────────
    # 若为第一个 AOI → 使用原始 staging；否则 → 使用上一个 AOI 的中心
    original_staging = input_data.get("staging_position", [150.0, -50.0])
    if current_idx > 0 and current_idx < len(aoi_sequence):
        prev_aoi_id = aoi_sequence[current_idx - 1]
        prev_info = aoi_map.get(prev_aoi_id, {})
        if prev_info:
            prv_row = int(prev_info.get("row", 1))
            prv_col = int(prev_info.get("col", 1))
            effective_staging = [
                (prv_col - 0.5) * AOI_SIZE_KM,
                (prv_row - 0.5) * AOI_SIZE_KM,
            ]
        else:
            effective_staging = original_staging
    else:
        effective_staging = original_staging

    # ── 解析平台（从有效出发点出发）───────────────────────
    from task_interface import _parse_platforms_from_dict
    import numpy as np
    platforms_raw = input_data.get("platforms", {})
    platforms_parsed = _parse_platforms_from_dict(platforms_raw, effective_staging)
    # 后续 AOI：覆盖平台位置为上一个 AOI 中心
    if current_idx > 0:
        for p in platforms_parsed:
            p.pos = np.array(effective_staging, dtype=np.float64)

    # ── 解析当前 AOI 的 MILP 求解计划（含实际搭载传感器） ──
    current_aoi_plan = result.get("current_aoi_plan") or {}
    solved_mounted = current_aoi_plan.get("mounted_sensors", {})

    pid_to_pos_xy: Dict[str, List[float]] = {}
    platforms_viz = []
    for p in platforms_parsed:
        xy = effective_staging if current_idx > 0 else (
            p.pos.tolist() if hasattr(p.pos, 'tolist') else list(p.pos)
        )
        pid_to_pos_xy[p.pid] = xy
        platforms_viz.append({
            "pid": p.pid, "type": p.type, "start": xy,
            "sensors_mounted": solved_mounted.get(p.pid, list(p.sensors_mounted)),
        })

    # ── 解析当前 AOI 的栅格 ──────────────────────────────
    from core.snapshot import generate_aoi_grids
    current_aoi_info = aoi_map.get(current_aoi, {})
    if current_aoi_info:
        grids_parsed = generate_aoi_grids(
            aoi_row=int(current_aoi_info.get("row", 1)),
            aoi_col=int(current_aoi_info.get("col", 1)),
        )
    else:
        grids_parsed = []

    cell_center: Dict[str, List[float]] = {}
    grids_viz = []
    for g in grids_parsed:
        center = g.center.tolist() if hasattr(g.center, 'tolist') else list(g.center)
        cell_center[g.cell_id] = center
        grids_viz.append({
            "cell_id": g.cell_id,
            "center": center,
            "w": g.width_km,
            "h": g.height_km,
            "weather_w": g.weather_w,
        })

    # ── 解析目标 ─────────────────────────────────────────
    from task_interface import _parse_targets_from_dict
    targets_raw = input_data.get("targets", [])
    targets_parsed = _parse_targets_from_dict(targets_raw)

    target_pos: Dict[str, List[float]] = {}
    targets_viz = []
    for t in targets_parsed:
        pos = t.pos_est.tolist() if hasattr(t.pos_est, 'tolist') else list(t.pos_est)
        target_pos[t.tid] = pos
        targets_viz.append({
            "tid": t.tid,
            "type": t.type,
            "pos": pos,
            "value": t.value,
            "threat": t.threat,
        })

    # ── 解析任务分配 ─────────────────────────────────────
    tasks = current_aoi_plan.get("tasks", [])

    recon_assignments = []
    strike_assignments = []
    for task in tasks:
        if task.get("task_type") == "recon":
            recon_assignments.append({
                "pid": task["platform"],
                "sensor": task.get("sensor", ""),
                "sensors_mounted": task.get("sensors_mounted", []),
                "cell": task.get("cell", ""),
                "role": task.get("role", ""),
            })
        elif task.get("task_type") == "strike":
            strike_assignments.append({
                "pid": task["platform"],
                "target": task.get("target", ""),
                "munition": task.get("munition", ""),
                "qty": task.get("qty", 0),
                "role": task.get("role", ""),
                "sensors_mounted": task.get("sensors_mounted", []),
            })

    # ── routes: 平台 → 当前 AOI 内分配点 ─────────────────
    uav_cells: Dict[str, List[str]] = {}
    for ra in recon_assignments:
        pid = ra["pid"]
        if pid not in uav_cells:
            uav_cells[pid] = []
        if ra["cell"] not in uav_cells[pid]:
            uav_cells[pid].append(ra["cell"])

    heli_targets: Dict[str, List[str]] = {}
    for sa in strike_assignments:
        pid = sa["pid"]
        if pid not in heli_targets:
            heli_targets[pid] = []
        if sa["target"] not in heli_targets[pid]:
            heli_targets[pid].append(sa["target"])

    routes_viz: List[Dict[str, Any]] = []
    for p in platforms_parsed:
        pid = p.pid
        ptype = p.type
        start_pos = tuple(pid_to_pos_xy[pid])
        waypoints: List[Tuple[float, float]] = [start_pos]
        labels: List[str] = ["start"]

        if ptype == "UAV" and pid in uav_cells:
            cell_coords = [tuple(cell_center[c]) for c in uav_cells[pid] if c in cell_center]
            ordered = _order_waypoints(start_pos, cell_coords)
            for coord in ordered:
                waypoints.append(coord)
                for cid, cxy in cell_center.items():
                    if abs(cxy[0] - coord[0]) < 0.01 and abs(cxy[1] - coord[1]) < 0.01:
                        labels.append(cid)
                        break
                else:
                    labels.append("?")
        elif ptype == "HELI" and pid in heli_targets:
            tgt_coords = [tuple(target_pos[t]) for t in heli_targets[pid] if t in target_pos]
            ordered = _order_waypoints(start_pos, tgt_coords)
            for coord in ordered:
                waypoints.append(coord)
                for tid, txy in target_pos.items():
                    if abs(txy[0] - coord[0]) < 0.01 and abs(txy[1] - coord[1]) < 0.01:
                        labels.append(tid)
                        break
                else:
                    labels.append("?")

        routes_viz.append({
            "pid": pid, "type": ptype,
            "waypoints": [list(w) for w in waypoints],
            "labels": labels,
        })

    # ── animation_frames ────────────────────────────────
    animation_frames: List[Dict[str, Any]] = []

    t0_positions = [{"pid": pl["pid"], "xy": pid_to_pos_xy[pl["pid"]], "type": pl["type"]}
                    for pl in platforms_viz]
    animation_frames.append({"phase": "t0", "tau": 0.0, "positions": t0_positions})

    t1_positions = [{"pid": pl["pid"], "xy": pid_to_pos_xy[pl["pid"]], "type": pl["type"]}
                    for pl in platforms_viz]
    animation_frames.append({"phase": "t1", "tau": 0.0, "positions": t1_positions})

    route_map: Dict[str, Dict] = {r["pid"]: r for r in routes_viz}
    for i in range(1, N_ANIM_STEPS + 1):
        tau = i / N_ANIM_STEPS
        positions = []
        for pl in platforms_viz:
            r = route_map.get(pl["pid"])
            if r and len(r["waypoints"]) >= 2:
                xy = _interpolate_route([tuple(w) for w in r["waypoints"]], tau)
            else:
                xy = tuple(pid_to_pos_xy[pl["pid"]])
            positions.append({"pid": pl["pid"], "xy": [xy[0], xy[1]], "type": pl["type"]})
        animation_frames.append({"phase": "t2", "tau": tau, "positions": positions})

    t3_positions = []
    for pl in platforms_viz:
        r = route_map.get(pl["pid"])
        if r and len(r["waypoints"]) >= 1:
            final = r["waypoints"][-1]
        else:
            final = pid_to_pos_xy[pl["pid"]]
        t3_positions.append({"pid": pl["pid"], "xy": final, "type": pl["type"]})
    animation_frames.append({"phase": "t3", "tau": 1.0, "positions": t3_positions})

    # ── meta ────────────────────────────────────────────
    meta = {
        "map_size_km": 300.0,
        "commander_aoi": current_aoi or "",
        "aoi_bbox": None,  # 多 AOI 模式下不使用单 AOI bbox
        "staging": effective_staging,
        "status": current_aoi_plan.get("solve_status", result.get("status", "N/A")),
        "objective": current_aoi_plan.get("objective", 0.0),
        "solve_time_ms": current_aoi_plan.get("solve_time_ms", 0.0),
        "solver_used": "",
        "current_aoi": current_aoi,
        "route_status": aoi_route_state.get("route_status", "RUNNING"),
        "all_aois_finished": all_finished,
    }

    return {
        "meta": meta,
        "platforms": platforms_viz,
        "grids": grids_viz,
        "targets": targets_viz,
        "recon_assignments": recon_assignments,
        "strike_assignments": strike_assignments,
        "routes": routes_viz,
        "animation_frames": animation_frames,
        "aois": aois_viz,
        "aoi_route": {
            "sequence": aoi_sequence,
            "centers": aoi_centers,
            "current_aoi": current_aoi,
            "current_index": current_idx,
        },
    }
