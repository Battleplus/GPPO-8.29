"""
战场地图绘制器 —— 使用 Plotly graph_objects 绘制多图层战场可视化。

图层（自底向上）:
  1. 任务区 300×300 km 外框
  2. 6×6 AOI 网格线（黑）
  3. 指挥官 AOI 高亮填充矩形
  4. 内部栅格 c0~c4（黑色边框，c0 虚线，c1~c4 实线 + 中心点标注）
  5. 集结区标记（黑色方块）
  6. 目标标记（红色叉号）
  7. 分配连线（t1 及以后，虚线）
  8. 平台标记（每帧更新，UAV 蓝圆、HELI 橙三角）
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional

import plotly.graph_objects as go

from visualization.constants import (
    MAP_SIZE_KM, AOI_SIZE_KM, GRID_LINE_INTERVAL,
    COLOR_UAV, COLOR_HELI, COLOR_TARGET,
    COLOR_STAGING, COLOR_AOI_FILL, COLOR_AOI_BORDER,
    COLOR_UAV_LINE, COLOR_HELI_LINE, COLOR_ALLOCATION_LINE,
    SYMBOL_UAV, SYMBOL_HELI, SYMBOL_TARGET, SYMBOL_STAGING,
    MARKER_SIZE_PLATFORM, MARKER_SIZE_TARGET, MARKER_SIZE_STAGING, MARKER_SIZE_GRID,
    LINE_WIDTH_ALLOCATION, LINE_WIDTH_GRID, LINE_WIDTH_AOI_BORDER,
    LINE_DASH_ALLOCATION, LINE_DASH_GRID_C0,
    COLOR_AOI_CURRENT_FILL, COLOR_AOI_CURRENT_BORDER,
    COLOR_AOI_FINISHED_FILL, COLOR_AOI_FINISHED_BORDER,
    COLOR_AOI_PENDING_FILL, COLOR_AOI_PENDING_BORDER,
    COLOR_AOI_ROUTE_LINE, COLOR_AOI_ROUTE_ARROW,
)


def _add_grid_layer(fig: go.Figure) -> None:
    """添加 6×6 AOI 网格线（黑）与 AOI 标签 A_r_c。"""
    # 垂直线
    for i in range(7):
        x = i * GRID_LINE_INTERVAL
        fig.add_shape(type="line", x0=x, y0=0, x1=x, y1=MAP_SIZE_KM,
                      line=dict(color="black", width=LINE_WIDTH_GRID),
                      layer="below")
    # 水平线
    for i in range(7):
        y = i * GRID_LINE_INTERVAL
        fig.add_shape(type="line", x0=0, y0=y, x1=MAP_SIZE_KM, y1=y,
                      line=dict(color="black", width=LINE_WIDTH_GRID),
                      layer="below")
    # AOI 中心标签
    for r in range(1, 7):
        for c in range(1, 7):
            cx = (c - 1) * AOI_SIZE_KM + AOI_SIZE_KM / 2
            cy = (r - 1) * AOI_SIZE_KM + AOI_SIZE_KM / 2
            fig.add_annotation(x=cx, y=cy, text=f"A_{r}_{c}",
                               showarrow=False,
                               font=dict(size=9, color="black"),
                               opacity=0.6)


def _add_aoi_highlight(fig: go.Figure, aoi_bbox: Optional[List[float]]) -> None:
    """高亮指挥官 AOI 区域（半透明填充 + 粗描边）。"""
    if aoi_bbox is None or len(aoi_bbox) != 4:
        return
    x0, y0, x1, y1 = aoi_bbox
    fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                  fillcolor=COLOR_AOI_FILL,
                  line=dict(color=COLOR_AOI_BORDER, width=LINE_WIDTH_AOI_BORDER),
                  layer="below")


def _weather_label(weather_w: float) -> str:
    """将天气系数转为中文描述。"""
    if weather_w <= 0.2:
        return f"晴好({weather_w:.2f})"
    elif weather_w <= 0.4:
        return f"轻雾({weather_w:.2f})"
    elif weather_w <= 0.6:
        return f"多云({weather_w:.2f})"
    else:
        return f"厚雾({weather_w:.2f})"


def _add_grid_cells(fig: go.Figure, grids: List[Dict[str, Any]]) -> None:
    """绘制 AOI 内部侦察栅格 c0~c4（含天气标注）。"""
    for g in grids:
        cid = g["cell_id"]
        cx, cy = g["center"]
        w, h = g["w"], g["h"]
        weather_w = g.get("weather_w", 0.0)
        x0, y0 = cx - w / 2, cy - h / 2
        x1, y1 = cx + w / 2, cy + h / 2

        if cid == "c0":
            # 巡逻区用虚线框
            fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                          line=dict(color="black", width=LINE_WIDTH_GRID,
                                    dash=LINE_DASH_GRID_C0),
                          fillcolor="rgba(0,0,0,0)")
        else:
            fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                          line=dict(color="black", width=LINE_WIDTH_GRID),
                          fillcolor="rgba(0,0,0,0)")

        # 中心点 + cell_id + 天气文字
        weather_text = _weather_label(weather_w)
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy],
            mode="markers+text",
            marker=dict(color="black", size=MARKER_SIZE_GRID,
                        symbol="circle"),
            text=[f"{cid}<br>{weather_text}"],
            textposition="top center",
            textfont=dict(size=9, color="black"),
            showlegend=False,
        ))


def _add_staging(fig: go.Figure, staging: List[float]) -> None:
    """绘制集结区标记（黑色方块 + 文本）。"""
    sx, sy = staging
    fig.add_trace(go.Scatter(
        x=[sx], y=[sy],
        mode="markers+text",
        marker=dict(color=COLOR_STAGING, size=MARKER_SIZE_STAGING,
                    symbol=SYMBOL_STAGING),
        text=["集结区"],
        textposition="bottom center",
        textfont=dict(size=11, color=COLOR_STAGING),
        name="集结区",
        showlegend=True,
    ))


def _add_targets(fig: go.Figure, targets: List[Dict[str, Any]]) -> None:
    """绘制目标标记（红色叉号 + 类型标签）。"""
    if not targets:
        return
    xs = [t["pos"][0] for t in targets]
    ys = [t["pos"][1] for t in targets]
    texts = [f"{t['tid']}<br>{t['type']}<br>价值{t['value']:.1f} 威胁{t['threat']:.1f}"
             for t in targets]
    labels = [t["tid"] for t in targets]

    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        marker=dict(color=COLOR_TARGET, size=MARKER_SIZE_TARGET,
                    symbol=SYMBOL_TARGET, line=dict(width=2, color=COLOR_TARGET)),
        text=labels,
        textposition="top right",
        textfont=dict(size=10, color=COLOR_TARGET),
        hovertext=texts,
        hoverinfo="text",
        name="目标",
        showlegend=True,
    ))


def _add_allocation_lines(fig: go.Figure, viz: Dict[str, Any]) -> None:
    """添加分配连线（平台当前位置 → 栅格/目标 虚线），t1 及以后显示。"""
    platforms = viz.get("platforms", [])
    recon = viz.get("recon_assignments", [])
    strike = viz.get("strike_assignments", [])

    grids = viz.get("grids", [])
    grid_map = {g["cell_id"]: g["center"] for g in grids}
    targets = viz.get("targets", [])
    target_map = {t["tid"]: t["pos"] for t in targets}

    pid_start = {pl["pid"]: pl["start"] for pl in platforms}

    # 侦察分配连线：平台当前位置 → cell center（UAV 蓝）
    for ra in recon:
        if ra["cell"] not in grid_map or ra["pid"] not in pid_start:
            continue
        start_xy = pid_start[ra["pid"]]
        cell_xy = grid_map[ra["cell"]]
        mounted = ra.get("sensors_mounted", [])
        mounted_str = "/".join(mounted) if mounted else "?"
        fig.add_trace(go.Scatter(
            x=[start_xy[0], cell_xy[0]], y=[start_xy[1], cell_xy[1]],
            mode="lines",
            line=dict(color=COLOR_UAV_LINE, width=LINE_WIDTH_ALLOCATION,
                      dash=LINE_DASH_ALLOCATION),
            showlegend=False,
            hoverinfo="text",
            hovertext=f"{ra['pid']}→{ra['cell']} | 搭载:[{mounted_str}] 使用:{ra.get('sensor', '?')}",
        ))

    # 打击分配连线：平台当前位置 → target（HELI 橙）
    for sa in strike:
        if sa["target"] not in target_map or sa["pid"] not in pid_start:
            continue
        start_xy = pid_start[sa["pid"]]
        tgt_xy = target_map[sa["target"]]
        mounted = sa.get("sensors_mounted", [])
        mounted_str = "/".join(mounted) if mounted else "?"
        mun = sa.get("munition", "")
        qty = sa.get("qty", 0)
        if mun and qty > 0:
            weapon_str = f"{mun}×{qty}"
        elif mun:
            weapon_str = f"{mun}×0(支援)"
        else:
            weapon_str = "支援(不发射)"
        fig.add_trace(go.Scatter(
            x=[start_xy[0], tgt_xy[0]], y=[start_xy[1], tgt_xy[1]],
            mode="lines",
            line=dict(color=COLOR_HELI_LINE, width=LINE_WIDTH_ALLOCATION,
                      dash=LINE_DASH_ALLOCATION),
            showlegend=False,
            hoverinfo="text",
            hovertext=f"{sa['pid']}→{sa['target']} | 搭载:[{mounted_str}] {weapon_str}",
        ))


def _add_platform_markers(fig: go.Figure, positions: List[Dict[str, Any]],
                         platforms_info: Optional[List[Dict[str, Any]]] = None) -> None:
    """在当前帧渲染平台标记（UAV 蓝圆、HELI 橙三角），hover 显示搭载传感器。"""
    # 建立 pid → sensors_mounted 映射
    pid_sensors: Dict[str, List[str]] = {}
    if platforms_info:
        for p in platforms_info:
            pid_sensors[p["pid"]] = p.get("sensors_mounted", [])

    uav_x, uav_y, uav_text, uav_hover = [], [], [], []
    heli_x, heli_y, heli_text, heli_hover = [], [], [], []

    for pos in positions:
        pid = pos["pid"]
        xy = pos["xy"]
        ptype = pos["type"]
        sensors = pid_sensors.get(pid, [])
        sensors_str = "/".join(sensors) if sensors else ""
        if ptype == "UAV":
            uav_x.append(xy[0]); uav_y.append(xy[1])
            uav_text.append(pid)
            uav_hover.append(f"{pid} 搭载:[{sensors_str}]" if sensors_str else pid)
        else:
            heli_x.append(xy[0]); heli_y.append(xy[1])
            heli_text.append(pid)
            heli_hover.append(f"{pid} 搭载:[{sensors_str}]" if sensors_str else pid)

    if uav_x:
        fig.add_trace(go.Scatter(
            x=uav_x, y=uav_y,
            mode="markers+text",
            marker=dict(color=COLOR_UAV, size=MARKER_SIZE_PLATFORM,
                        symbol=SYMBOL_UAV,
                        line=dict(width=1, color="white")),
            text=uav_text,
            textposition="middle right",
            textfont=dict(size=10, color=COLOR_UAV),
            hovertext=uav_hover,
            hoverinfo="text",
            name="UAV",
            showlegend=True,
        ))
    if heli_x:
        fig.add_trace(go.Scatter(
            x=heli_x, y=heli_y,
            mode="markers+text",
            marker=dict(color=COLOR_HELI, size=MARKER_SIZE_PLATFORM,
                        symbol=SYMBOL_HELI,
                        line=dict(width=1, color="white")),
            text=heli_text,
            textposition="middle right",
            textfont=dict(size=10, color=COLOR_HELI),
            hovertext=heli_hover,
            hoverinfo="text",
            name="HELI",
            showlegend=True,
        ))


def _add_multi_aoi_highlights(fig: go.Figure, aois: List[Dict[str, Any]]) -> None:
    """绘制所有候选 AOI，按状态着色。"""
    for a in aois:
        x0, y0, x1, y1 = a["bbox"]
        status = a["status"]
        order = a.get("order", 0)

        if status == "CURRENT":
            fill, border, dash = COLOR_AOI_CURRENT_FILL, COLOR_AOI_CURRENT_BORDER, None
            lw = 3.0
        elif status == "FINISHED":
            fill, border, dash = COLOR_AOI_FINISHED_FILL, COLOR_AOI_FINISHED_BORDER, None
            lw = 2.0
        else:  # PENDING
            fill, border, dash = COLOR_AOI_PENDING_FILL, COLOR_AOI_PENDING_BORDER, "dot"
            lw = 1.5

        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      fillcolor=fill, line=dict(color=border, width=lw, dash=dash) if dash else dict(color=border, width=lw),
                      layer="below")

        # 顺序数字标注 + AOI ID
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        label_parts = [f"AOI #{order}", a["id"]]
        if status == "CURRENT":
            label_parts.append("◀ 当前")
        elif status == "FINISHED":
            label_parts.append("✓ 已完成")

        fig.add_annotation(x=cx, y=cy, text="<br>".join(label_parts),
                           showarrow=False,
                           font=dict(size=10, color=border),
                           bgcolor="rgba(255,255,255,0.75)",
                           borderpad=3)


def _add_aoi_route_path(fig: go.Figure,
                         aoi_centers: List[List[float]],
                         staging: List[float],
                         current_idx: int) -> None:
    """绘制 AOI 间的排序路径：staging → AOI#1 → AOI#2 → ..."""
    all_pts = [staging] + aoi_centers
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]

    # 灰色虚线路径
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines+markers+text",
        line=dict(color=COLOR_AOI_ROUTE_LINE, width=2.0, dash="dashdot"),
        marker=dict(color=[COLOR_AOI_ROUTE_ARROW] * len(all_pts), size=6, symbol="circle"),
        text=["集结区"] + [f"#{i+1}" for i in range(len(aoi_centers))],
        textposition=["bottom center"] + ["top center"] * len(aoi_centers),
        textfont=dict(size=10, color=COLOR_AOI_ROUTE_ARROW),
        showlegend=False,
        hoverinfo="text",
        hovertext=["集结区"] + [f"AOI #{i+1} (执行顺序)" for i in range(len(aoi_centers))],
    ))


def _add_targets_in_aoi_markers(fig: go.Figure, targets: List[Dict], current_aoi_bbox: List[float]) -> None:
    """高亮落在当前 AOI 内的目标（和普通目标叠加显示时保持清晰）。"""
    if not targets or not current_aoi_bbox:
        return
    x0, y0, x1, y1 = current_aoi_bbox
    in_aoi = [t for t in targets
              if x0 <= t["pos"][0] <= x1 and y0 <= t["pos"][1] <= y1]
    if not in_aoi:
        return
    xs = [t["pos"][0] for t in in_aoi]
    ys = [t["pos"][1] for t in in_aoi]
    texts = [f"{t['tid']} (当前AOI内)" for t in in_aoi]
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker=dict(color=COLOR_AOI_CURRENT_BORDER, size=MARKER_SIZE_TARGET + 2,
                    symbol="circle-open", line=dict(width=3, color=COLOR_AOI_CURRENT_BORDER)),
        name="AOI 内目标",
        hovertext=texts,
        hoverinfo="text",
        showlegend=True,
    ))


def render_figure(viz: Dict[str, Any], frame_idx: int = 0,
                  multi_aoi_data: Optional[Dict[str, Any]] = None) -> go.Figure:
    """
    根据 viz dict 和帧索引渲染 Plotly 战场地图。

    Args:
        viz: build_visualization_state 的输出 dict
        frame_idx: 动画帧索引（0-based，对应 animation_frames 数组）

    Returns:
        plotly.graph_objects.Figure
    """
    meta = viz.get("meta", {})
    staging = meta.get("staging", [150.0, -50.0])
    frames = viz.get("animation_frames", [])
    grids = viz.get("grids", [])
    targets = viz.get("targets", [])

    fig = go.Figure()

    # ── Layer 1: 任务区外框 ────────────────────────────
    fig.add_shape(type="rect", x0=0, y0=0, x1=MAP_SIZE_KM, y1=MAP_SIZE_KM,
                  line=dict(color="black", width=1.5), fillcolor="rgba(0,0,0,0)")

    # ── Layer 2: AOI 网格 ──────────────────────────────
    _add_grid_layer(fig)

    # ── Layer 3: AOI 高亮（多 AOI 或单 AOI） ──────────
    if multi_aoi_data and multi_aoi_data.get("aois"):
        _add_multi_aoi_highlights(fig, multi_aoi_data["aois"])
    else:
        _add_aoi_highlight(fig, meta.get("aoi_bbox"))

    # ── Layer 3.5: AOI 路径（仅多 AOI） ────────────────
    if multi_aoi_data and multi_aoi_data.get("aoi_route"):
        route = multi_aoi_data["aoi_route"]
        _add_aoi_route_path(fig, route["centers"], staging, route.get("current_index", 0))

    # ── Layer 4: 栅格 c0~c4 ────────────────────────────
    _add_grid_cells(fig, grids)

    # ── Layer 5: 集结区 ────────────────────────────────
    _add_staging(fig, staging)

    # ── Layer 6: 目标 ──────────────────────────────────
    _add_targets(fig, targets)

    # ── Layer 6.5: AOI 内目标高亮（仅多 AOI） ─────────
    if multi_aoi_data and multi_aoi_data.get("aois") and targets:
        current_aoi = next((a for a in multi_aoi_data["aois"] if a["status"] == "CURRENT"), None)
        if current_aoi:
            _add_targets_in_aoi_markers(fig, targets, current_aoi["bbox"])

    # ── 确定当前帧 ─────────────────────────────────────
    if frames and 0 <= frame_idx < len(frames):
        frame = frames[frame_idx]
    elif frames:
        frame = frames[0]
    else:
        frame = None

    phase = frame["phase"] if frame else "t0"
    positions = frame.get("positions", []) if frame else []

    # ── Layer 7: 分配连线（t1 及以后） ─────────────────
    if phase in ("t1", "t2", "t3"):
        _add_allocation_lines(fig, viz)

    # ── Layer 8: 平台标记（含传感器信息） ──────────────
    _add_platform_markers(fig, positions, platforms_info=viz.get("platforms"))

    # ── 标题（多 AOI 时附加当前 AOI 信息） ───────────────
    title_text = f"UAV–HELI 协同侦察打击 · 任务分配可视化  [{phase.upper()}]"
    if multi_aoi_data and multi_aoi_data.get("meta", {}).get("current_aoi"):
        ma_meta = multi_aoi_data["meta"]
        title_text += f"  |  当前 AOI: {ma_meta['current_aoi']}"
        if ma_meta.get("all_aois_finished"):
            title_text += "  |  全部完成 ✓"

    # ── 布局配置 ───────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=16),
        ),
        xaxis=dict(
            title="X (km)", range=[-10, MAP_SIZE_KM + 10],
            scaleanchor="y", scaleratio=1, constrain="domain",
            showgrid=True, gridcolor="rgba(200,200,200,0.3)",
        ),
        yaxis=dict(
            title="Y (km)", range=[-50, MAP_SIZE_KM + 20],
            showgrid=True, gridcolor="rgba(200,200,200,0.3)",
        ),
        width=900,
        height=900,
        margin=dict(l=60, r=30, t=60, b=60),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        hovermode="closest",
    )

    # 确保坐标轴等比例
    fig.update_xaxes(scaleanchor="y", scaleratio=1)

    return fig
