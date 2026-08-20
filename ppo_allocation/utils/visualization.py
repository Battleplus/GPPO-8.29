"""可视化模块。

提供场景状态绘图、before/after 静态对比图和动画生成功能。
使用 matplotlib 作为渲染后端。
"""

from pathlib import Path
import copy
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter
from config import (
    AREA_SIZE,
    REGION_BOUNDS,
    Weather,
    SensorType,
    TargetType,
    TaskType,
    NO_UAV,
)

# =========================
# 全局样式配置
# =========================

UAV_COLORS = {
    0: "#E74C3C",
    1: "#3498DB",
    2: "#2ECC71",
    3: "#9B59B6",
}

REGION_SUNNY_COLOR = "#FFF8E1"
REGION_RAINY_COLOR = "#E3F2FD"

TARGET_COLORS = {
    "tracked": "#E74C3C",
    "destroyed": "#95A5A6",
    "discovered": "#F39C12",
    "undiscovered": "#B0B0B0",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "figure.facecolor": "white",
})


# =========================
# 辅助标签
# =========================

def _weather_str(weather):
    return "Sunny" if weather == Weather.SUNNY else "Rainy"


def _sensor_label(sensor):
    return "EO" if sensor == SensorType.EO else "SAR"


def _task_label(task):
    if task == TaskType.SEARCH:
        return "SEARCH"
    if task == TaskType.TRACK:
        return "TRACK"
    return "IDLE"


def _target_type_label(tt):
    return "CAR" if tt == TargetType.CAR else "CMD"


# =========================
# 单帧绘图
# =========================

def draw_state(ax, snapshot, title: str, show_legend: bool = True):
    """在 matplotlib Axes 上绘制一帧场景状态。"""
    ax.set_xlim(-3, AREA_SIZE + 3)
    ax.set_ylim(-3, AREA_SIZE + 3)
    ax.set_aspect("equal")
    ax.set_title(title, fontweight="bold", pad=16, fontsize=15)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.grid(True, alpha=0.12, linestyle="--", linewidth=0.5)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    regions_data = snapshot["regions"]
    uavs_data = snapshot["uavs"]
    targets_data = snapshot["targets"]

    # ---- 第 1 层：区域底色 ----
    for rid, (xmin, xmax, ymin, ymax) in REGION_BOUNDS.items():
        r = regions_data[str(rid)]
        weather = Weather(r["weather"])
        face_color = REGION_SUNNY_COLOR if weather == Weather.SUNNY else REGION_RAINY_COLOR
        rect = mpatches.FancyBboxPatch(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            boxstyle="round,pad=0.4",
            facecolor=face_color,
            edgecolor="#CCCCCC",
            linewidth=1.5,
            zorder=0,
        )
        ax.add_patch(rect)

    # ---- 第 2 层：区域标签 ----
    for rid, (xmin, xmax, ymin, ymax) in REGION_BOUNDS.items():
        r = regions_data[str(rid)]
        weather = Weather(r["weather"])
        assigned = r["assigned_uav"]

        # 区域编号 + 天气 — 左上角
        ax.text(xmin + 2.5, ymax - 4, f"R{rid}",
                fontsize=14, fontweight="bold", color="#333333",
                ha="left", va="top", zorder=3)
        w_str = _weather_str(weather)
        w_color = "#E67E22" if weather == Weather.SUNNY else "#2980B9"
        ax.text(xmin + 2.5, ymax - 8, w_str,
                fontsize=10, color=w_color,
                ha="left", va="top", zorder=3)

        # 分配信息 — 区域正中偏下，给已摧毁无人机标记留出上方空间
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        if assigned != NO_UAV:
            uid_color = UAV_COLORS.get(assigned, "#333333")
            ax.text(cx, cy, f"U{assigned}",
                    fontsize=13, fontweight="bold", color=uid_color,
                    ha="center", va="center", zorder=3)
        else:
            rect_highlight = mpatches.Rectangle(
                (xmin + 3, ymin + 3), xmax - xmin - 6, ymax - ymin - 6,
                fill=False, edgecolor="#E74C3C", linewidth=3,
                linestyle="--", zorder=2, alpha=0.8
            )
            ax.add_patch(rect_highlight)
            ax.text(cx, cy,
                    "VACANT",
                    fontsize=12, color="#E74C3C", style="italic",
                    fontweight="bold",
                    ha="center", va="center", zorder=3)

    # ---- 第 3 层：已摧毁无人机标记（在区域标签之后、分配连线之前） ----
    # 绘制在区域角落，避免与中心标签重叠
    for uid, u in uavs_data.items():
        uid_int = int(uid)
        if u["alive"]:
            continue
        # 将摧毁标记偏移到其所在区域的右下角
        rid = _find_nearest_region(u["x"], u["y"])
        if rid is not None:
            xmin, xmax, ymin, ymax = REGION_BOUNDS[rid]
            dx, dy = xmax - 5, ymin + 5
        else:
            dx, dy = u["x"], u["y"]

        ax.scatter(dx, dy, s=100, marker="X",
                   color="#BDC3C7", edgecolors="#95A5A6",
                   linewidths=2.0, zorder=4, alpha=0.7)
        ax.text(dx - 3, dy + 2, f"U{uid} [DESTROYED]",
                fontsize=8, color="#95A5A6",
                fontweight="bold", ha="right", va="bottom", zorder=4)

    # ---- 第 4 层：分配连线 ----
    for uid, u in uavs_data.items():
        uid_int = int(uid)
        if not u["alive"]:
            continue
        color = UAV_COLORS.get(uid_int, "#333333")
        for rid in u.get("regions", []):
            xmin, xmax, ymin, ymax = REGION_BOUNDS[rid]
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            ax.plot([u["x"], cx], [u["y"], cy],
                    color=color, linewidth=2, linestyle="--",
                    alpha=0.45, zorder=1)

    # ---- 第 5 层：存活无人机图标 ----
    for uid, u in uavs_data.items():
        uid_int = int(uid)
        if not u["alive"]:
            continue
        color = UAV_COLORS.get(uid_int, "#333333")
        sensor = SensorType(u["sensor"])
        task = TaskType(u["task"])

        ax.scatter(u["x"], u["y"], s=280, marker="o",
                   facecolor=color, edgecolors="white",
                   linewidths=2.5, zorder=5)
        ax.scatter(u["x"], u["y"], s=80, marker="o",
                   facecolor="white", edgecolors="none", zorder=6)

        label_x = u["x"] + 3.0
        label_y = u["y"] + 3.0
        ax.text(label_x, label_y, f"U{uid}",
                fontsize=11, fontweight="bold", color=color,
                ha="left", va="bottom", zorder=5)
        ax.text(label_x, label_y - 2.5,
                f"{_sensor_label(sensor)}  {_task_label(task)}",
                fontsize=9, color="#555555",
                ha="left", va="bottom", zorder=5)

        # 负载徽章
        n_regions = len(u.get("regions", []))
        if n_regions > 0:
            ax.text(u["x"] - 1.5, u["y"] + 2.5,
                    str(n_regions),
                    fontsize=10, fontweight="bold", color="white",
                    ha="center", va="center", zorder=7,
                    bbox=dict(boxstyle="circle,pad=0.2",
                              facecolor=color, edgecolor="none", alpha=0.9))

    # ---- 第 6 层：目标图标 ----
    for tid, t in targets_data.items():
        tx, ty = t["x"], t["y"]
        ttype = _target_type_label(t["target_type"])

        if t["destroyed"]:
            marker_color = TARGET_COLORS["destroyed"]
            marker, status_text = "s", "DESTROYED"
        elif t["tracked"]:
            marker_color = TARGET_COLORS["tracked"]
            marker, status_text = "^", "TRACKED"
        elif t["discovered"]:
            marker_color = TARGET_COLORS["discovered"]
            marker, status_text = "^", "DISC"
        else:
            # 未发现目标以半透明虚线圆绘制，表示"疑似存在"
            ax.scatter(tx, ty, s=120, marker="o",
                       facecolor="none", edgecolors=TARGET_COLORS["undiscovered"],
                       linewidths=1.5, linestyle="--", zorder=3, alpha=0.6)
            ax.text(tx + 2.2, ty + 2.2,
                    f"T{tid}?",
                    fontsize=9, color="#999999", style="italic",
                    ha="left", va="bottom", zorder=3)
            continue

        ax.scatter(tx, ty, s=180, marker=marker,
                   facecolor=marker_color, edgecolors="white",
                   linewidths=2, zorder=4)
        ax.text(tx + 2.8, ty + 2.8,
                f"T{tid} ({ttype})",
                fontsize=10, fontweight="bold", color=marker_color,
                ha="left", va="bottom")
        ax.text(tx + 2.8, ty + 0.6,
                status_text,
                fontsize=8, color="#777777",
                ha="left", va="bottom")

    # ---- 第 7 层：图例 ----
    if show_legend:
        legend_elements = []
        # 存活无人机
        for uid, u in uavs_data.items():
            uid_int = int(uid)
            if not u["alive"]:
                continue
            color = UAV_COLORS.get(uid_int, "#333333")
            sensor = _sensor_label(SensorType(u["sensor"]))
            task = _task_label(TaskType(u["task"]))
            n_regions = len(u.get("regions", []))
            label_str = f"U{uid}  {sensor}  {task}  [{n_regions}区域]"
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=color, markersize=12,
                       label=label_str)
            )
        # 已摧毁无人机
        for uid, u in uavs_data.items():
            if u["alive"]:
                continue
            label_str = f"U{uid}  DESTROYED"
            legend_elements.append(
                Line2D([0], [0], marker="X", color="w",
                       markerfacecolor="#BDC3C7", markersize=10,
                       markeredgecolor="#95A5A6", markeredgewidth=1.5,
                       label=label_str)
            )
        ax.legend(handles=legend_elements, loc="lower left",
                  fontsize=10, ncol=2, framealpha=0.9,
                  edgecolor="#CCCCCC", handletextpad=0.5,
                  columnspacing=0.6)


def _find_nearest_region(x: float, y: float):
    """根据坐标找到最近的区域编号。"""
    best_rid, best_dist = None, float("inf")
    for rid, (xmin, xmax, ymin, ymax) in REGION_BOUNDS.items():
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        d = (x - cx) ** 2 + (y - cy) ** 2
        if d < best_dist:
            best_dist = d
            best_rid = rid
    return best_rid


# =========================
# 静态对比图
# =========================

def save_comparison_figure(before, after, event_text: str, action_summary: str, output_path: str):
    """生成 PPO 重分配前后的静态对比 PNG。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(24, 14))

    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 5.0],
                          hspace=0.3, wspace=0.22,
                          left=0.04, right=0.98, top=0.96, bottom=0.04)

    # ---- 上方信息面板 ----
    ax_info = fig.add_subplot(gs[0, :])
    ax_info.axis("off")

    ax_info.text(0.02, 0.95, "TRIGGER EVENT",
                 fontsize=14, fontweight="bold", color="#C0392B",
                 ha="left", va="top")
    ax_info.text(0.02, 0.55, event_text,
                 fontsize=13, color="#333333",
                 ha="left", va="center",
                 bbox=dict(boxstyle="round,pad=0.6",
                           facecolor="#FFF5F5", edgecolor="#E8B4B4",
                           linewidth=1.5))

    ax_info.text(0.52, 0.95, "PPO REALLOCATION ACTION",
                 fontsize=14, fontweight="bold", color="#2980B9",
                 ha="left", va="top")
    ax_info.text(0.52, 0.55, action_summary,
                 fontsize=10.5, color="#333333",
                 ha="left", va="center", family="monospace",
                 linespacing=1.8,
                 bbox=dict(boxstyle="round,pad=0.6",
                           facecolor="#F0F7FB", edgecolor="#B0C8E0",
                           linewidth=1.5))

    # ---- 下方地图 ----
    ax_left = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    draw_state(ax_left, before, "Before — Initial State", show_legend=True)
    draw_state(ax_right, after, "After — PPO Local Reallocation", show_legend=False)

    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    return str(output_path)


# =========================
# GIF 动画
# =========================

def save_before_after_animation(before, after, event_text: str, action_summary: str, output_path: str):
    """生成 PPO 重分配前后的对比 GIF 动画。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 13))

    frames = [
        (before, "BEFORE: Initial State"),
        (before, "BEFORE: Initial State"),
        (before, "BEFORE: Initial State"),
        (after,  "AFTER: PPO Local Reallocation"),
        (after,  "AFTER: PPO Local Reallocation"),
        (after,  "AFTER: PPO Local Reallocation"),
    ]

    def update(i):
        ax.clear()
        snapshot, title = frames[i]
        draw_state(ax, snapshot, title, show_legend=True)

        summary_first_line = action_summary.split("\n")[0] if action_summary else ""
        ax.text(0.5, -0.09,
                f"Event: {event_text}",
                transform=ax.transAxes,
                fontsize=13, color="#C0392B",
                ha="center", va="top", style="italic",
                fontweight="bold")
        ax.text(0.5, -0.16,
                summary_first_line,
                transform=ax.transAxes,
                fontsize=11, color="#555555",
                ha="center", va="top")

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1800, repeat=True)
    anim.save(output_path, writer=PillowWriter(fps=0.56), dpi=130)
    plt.close(fig)
    return str(output_path)


# =========================
# 循环测试总览图
# =========================

def save_loop_overview_figure(initial_snapshot: dict, round_records: list, output_path: str):
    """生成循环测试全流程总览 PNG。

    布局：3 列 × 2 行，共 6 面板。
      [初始态势]  [第1轮: 事件 → PPO]  [第2轮: 事件 → PPO]
      [第3轮... ]  [第4轮...         ]  [第5轮...         ]

    Args:
        initial_snapshot: 初始场景快照
        round_records:    [{"round": 1, "event": "...", "snapshot": {...}}, ...]
        output_path:      输出 PNG 路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panels = [("Initial State", initial_snapshot)]
    for rec in round_records:
        r = rec["round"]
        event_short = rec["event"] if len(rec["event"]) <= 55 else rec["event"][:52] + "..."
        title = f"Round {r}: {event_short}"
        panels.append((title, rec["snapshot"]))

    n = len(panels)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12 * cols, 11 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for idx, (title, snap) in enumerate(panels):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        draw_state(ax, snap, title, show_legend=(idx == 0))

    # 隐藏多余子图
    for idx in range(len(panels), rows * cols):
        r, c = idx // cols, idx % cols
        axes[r][c].axis("off")

    fig.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    return str(output_path)


# =========================
# 循环测试 HTML 动画
# =========================

_ANIM_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UAV PPO Task Allocation — Loop Test Animation</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#f0f2f5;color:#2c3e50;font-family:'Segoe UI','Microsoft YaHei',sans-serif;
  display:flex;flex-direction:column;align-items:center;min-height:100vh;padding-bottom:30px}
h1{font-size:1.35em;margin:18px 0 4px;color:#2c3e50}
#subtitle{font-size:.82em;color:#7f8c8d;margin-bottom:12px;max-width:760px;text-align:center}
#container{position:relative;background:#fff;border-radius:10px;
  box-shadow:0 2px 12px rgba(0,0,0,.08);padding:6px}
canvas{display:block;border-radius:6px}
#info-panel{width:760px;background:#fff;border-radius:10px;
  margin-top:10px;padding:14px 18px;box-shadow:0 2px 12px rgba(0,0,0,.08);
  border-left:4px solid #f39c12}
#info-panel .round{font-size:1.05em;font-weight:bold;color:#e67e22;margin-bottom:4px}
#info-panel .event{font-size:.9em;color:#c0392b;margin-bottom:6px}
#info-panel .action{font-size:.8em;color:#555;font-family:'Cascadia Code','Consolas',monospace;
  white-space:pre-line;line-height:1.45}
#controls{display:flex;align-items:center;gap:10px;margin-top:10px;width:760px}
#controls button{background:#fff;color:#555;border:1px solid #d5d8dc;
  padding:7px 14px;border-radius:6px;cursor:pointer;font-size:.85em;transition:all .15s}
#controls button:hover{background:#f0f2f5;border-color:#b0b8c0}
#controls button:active{transform:scale(.96)}
#controls button.active{background:#e74c3c;color:#fff;border-color:#c0392b}
#progress{flex:1;display:flex;align-items:center;gap:8px}
#slider{flex:1;accent-color:#e67e22}
#frame-label{font-size:.82em;color:#888;min-width:64px;text-align:center}
#legend{display:flex;flex-wrap:wrap;gap:12px;margin-top:8px;width:760px;align-items:center}
.leg-item{display:flex;align-items:center;gap:5px;font-size:.76em;color:#666}
.leg-drone{width:22px;height:22px;position:relative}
.leg-drone canvas{position:absolute;top:0;left:0}
</style>
</head>
<body>
<h1>UAV PPO Task Allocation — Loop Test</h1>
<div id="subtitle">__SUBTITLE__</div>
<div id="container"><canvas id="map"></canvas></div>
<div id="info-panel">
  <div class="round" id="round-title">Initial State</div>
  <div class="event" id="event-text"></div>
  <div class="action" id="action-text"></div>
</div>
<div id="controls">
  <button id="btn-prev" title="Previous frame">◀ Prev</button>
  <button id="btn-play" title="Play / Pause">▶ Play</button>
  <button id="btn-next" title="Next frame">Next ▶</button>
  <div id="progress">
    <input type="range" id="slider" min="0" max="0" value="0" step="1">
    <span id="frame-label">0 / 0</span>
  </div>
</div>
<div id="legend"></div>

<script>
// ========== DATA ==========
var ALL_FRAMES = __FRAMES_JSON__;

// ========== CONSTANTS ==========
var AREA_SIZE = 50;
var PAD = 50;
var SCALE = 13.2;
var CANVAS_W = PAD * 2 + AREA_SIZE * SCALE;
var CANVAS_H = PAD * 2 + AREA_SIZE * SCALE;

var UAV_COLORS = {0:"#E74C3C", 1:"#3498DB", 2:"#2ECC71", 3:"#9B59B6"};
var UAV_LIGHT  = {0:"#FADBD8", 1:"#D6EAF8", 2:"#D5F5E3", 3:"#E8DAEF"};
var REGION_BOUNDS = {0:[0,25,25,50], 1:[25,50,25,50], 2:[0,25,0,25], 3:[25,50,0,25]};
var TARGET_META = __TARGET_META_JSON__;

// ========== COORDINATE HELPER ==========
function tx(x){return PAD + x * SCALE;}
function ty(y){return PAD + (AREA_SIZE - y) * SCALE;}

// ========== CANVAS SETUP ==========
var canvas = document.getElementById("map");
canvas.width = CANVAS_W;
canvas.height = CANVAS_H;
var ctx = canvas.getContext("2d");

// ========== DRAWING HELPERS ==========
function roundRect(x,y,w,h,r){
  ctx.beginPath();
  ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);
  ctx.quadraticCurveTo(x+w,y,x+w,y+r);
  ctx.lineTo(x+w,y+h-r);
  ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);
  ctx.lineTo(x+r,y+h);
  ctx.quadraticCurveTo(x,y+h,x,y+h-r);
  ctx.lineTo(x,y+r);
  ctx.quadraticCurveTo(x,y,x+r,y);
  ctx.closePath();
}

function drawArrowHead(fromX,fromY,toX,toY,color){
  var dx=toX-fromX, dy=toY-fromY;
  var len=Math.sqrt(dx*dx+dy*dy);
  if(len<1)return;
  var ux=dx/len, uy=dy/len;
  var perpX=-uy, perpY=ux;
  var tipX=toX-ux*16, tipY=toY-uy*16;
  ctx.fillStyle=color;
  ctx.beginPath();
  ctx.moveTo(tipX+ux*10, tipY+uy*10);
  ctx.lineTo(tipX+perpX*6, tipY+perpY*6);
  ctx.lineTo(tipX-perpX*6, tipY-perpY*6);
  ctx.closePath();
  ctx.fill();
}

// ========== WEATHER ICONS ==========
function drawSunIcon(cx,cy){
  ctx.fillStyle="#f39c12";
  ctx.beginPath();ctx.arc(cx,cy,5.5,0,Math.PI*2);ctx.fill();
  ctx.strokeStyle="#f39c12";ctx.lineWidth=2;
  for(var a=0;a<8;a++){
    var angle=a*Math.PI/4;
    ctx.beginPath();
    ctx.moveTo(cx+Math.cos(angle)*8, cy+Math.sin(angle)*8);
    ctx.lineTo(cx+Math.cos(angle)*11, cy+Math.sin(angle)*11);
    ctx.stroke();
  }
}

function drawRainIcon(cx,cy){
  // cloud
  ctx.fillStyle="#85a5c8";
  ctx.beginPath();ctx.arc(cx-3,cy-1,4.5,0,Math.PI*2);ctx.fill();
  ctx.beginPath();ctx.arc(cx+3,cy-2,5.5,0,Math.PI*2);ctx.fill();
  ctx.beginPath();ctx.arc(cx,cy-4,4,0,Math.PI*2);ctx.fill();
  // rain drops
  ctx.strokeStyle="#5090d0";ctx.lineWidth=1.5;
  for(var i=0;i<3;i++){
    var rx=cx-4+i*4;
    ctx.beginPath();
    ctx.moveTo(rx,cy+2);ctx.lineTo(rx-1,cy+7);ctx.stroke();
  }
}

// ========== DRONE ICON ==========
function drawDrone(x,y,color,lightColor,scale){
  var s=scale||1;
  // four rotors
  var arms=[[-7,-5],[-7,5],[7,-5],[7,5]];
  for(var i=0;i<4;i++){
    ctx.fillStyle=color;
    ctx.globalAlpha=0.3;
    ctx.beginPath();
    ctx.arc(x+arms[i][0]*s, y+arms[i][1]*s, 4.5*s, 0, Math.PI*2);
    ctx.fill();
    ctx.globalAlpha=1;
    ctx.strokeStyle=color;ctx.lineWidth=1.2*s;
    ctx.beginPath();
    ctx.arc(x+arms[i][0]*s, y+arms[i][1]*s, 2.8*s, 0, Math.PI*2);
    ctx.stroke();
  }
  // arms
  ctx.strokeStyle=color;ctx.lineWidth=1.8*s;
  ctx.beginPath();ctx.moveTo(x-7*s,y-5*s);ctx.lineTo(x,y);ctx.stroke();
  ctx.beginPath();ctx.moveTo(x-7*s,y+5*s);ctx.lineTo(x,y);ctx.stroke();
  ctx.beginPath();ctx.moveTo(x+7*s,y-5*s);ctx.lineTo(x,y);ctx.stroke();
  ctx.beginPath();ctx.moveTo(x+7*s,y+5*s);ctx.lineTo(x,y);ctx.stroke();
  // central body
  ctx.fillStyle=color;
  roundRect(x-3.5*s, y-2.8*s, 7*s, 5.6*s, 2*s);
  ctx.fill();
  ctx.fillStyle="#fff";ctx.globalAlpha=0.7;
  roundRect(x-1.8*s, y-1.2*s, 3.6*s, 2.4*s, 1.2*s);
  ctx.fill();
  ctx.globalAlpha=1;
}

function drawDestroyedMark(x,y){
  ctx.strokeStyle="#bdc3c7";ctx.lineWidth=2.5;ctx.globalAlpha=0.6;
  ctx.beginPath();ctx.moveTo(x-7,y-7);ctx.lineTo(x+7,y+7);ctx.stroke();
  ctx.beginPath();ctx.moveTo(x+7,y-7);ctx.lineTo(x-7,y+7);ctx.stroke();
  ctx.globalAlpha=1;
}

// ========== GRID ==========
function drawGrid(){
  // grid lines
  ctx.strokeStyle="#e8e8e8";ctx.lineWidth=0.5;
  for(var i=0;i<=AREA_SIZE;i+=5){
    ctx.beginPath();ctx.moveTo(tx(i),ty(0));ctx.lineTo(tx(i),ty(AREA_SIZE));ctx.stroke();
    ctx.beginPath();ctx.moveTo(tx(0),ty(i));ctx.lineTo(tx(AREA_SIZE),ty(i));ctx.stroke();
  }
}

// ========== REGIONS ==========
function drawRegion(rid){
  var b=REGION_BOUNDS[rid], xmin=b[0],xmax=b[1],ymin=b[2],ymax=b[3];
  var rx=tx(xmin),ry=ty(ymax),rw=(xmax-xmin)*SCALE,rh=(ymax-ymin)*SCALE;
  var r=ALL_FRAMES[currentFrame].snapshot.regions[String(rid)];
  var isSunny=r.weather===0;

  // fill
  ctx.fillStyle=isSunny?"#FFF3E0":"#E3F2FD";
  roundRect(rx,ry,rw,rh,6);ctx.fill();

  // inner border only (not the outer edges shared with map boundary)
  ctx.strokeStyle="#d5d8dc";ctx.lineWidth=1;
  var innerPad=1.5;
  // vertical midline
  if(xmin>0){
    ctx.beginPath();ctx.moveTo(rx,ry+innerPad);ctx.lineTo(rx,ry+rh-innerPad);ctx.stroke();
  }
  // horizontal midline
  if(ymin>0){
    ctx.beginPath();ctx.moveTo(rx+innerPad,ty(ymin));ctx.lineTo(rx+rw-innerPad,ty(ymin));ctx.stroke();
  }

  // weather icon — top-left
  if(isSunny){drawSunIcon(rx+14,ry+14);}
  else{drawRainIcon(rx+14,ry+14);}

  // region label
  ctx.fillStyle="#7f8c8d";
  ctx.font="bold 11px 'Segoe UI','Microsoft YaHei'";
  ctx.textAlign="left";ctx.textBaseline="top";
  ctx.fillText("R"+rid, rx+26, ry+6);

  // assigned UAV or VACANT — center
  var assigned=r.assigned_uav;
  var cx=rx+rw/2, cy=ry+rh/2;
  if(assigned!==-1){
    var acolor=UAV_COLORS[assigned]||"#888";
    var alight=UAV_LIGHT[assigned]||"#eee";
    var apos=UAV_DRAW_POS[assigned];
    var dcx=apos?apos.ux:cx;
    var dcy=apos?apos.uy:cy;
    drawDrone(dcx, dcy-2, acolor, alight, 0.85);
    ctx.fillStyle=acolor;
    ctx.font="bold 11px 'Segoe UI','Microsoft YaHei'";
    ctx.textAlign="center";ctx.textBaseline="top";
    ctx.fillText("U"+assigned, dcx, dcy+12);
  } else {
    ctx.strokeStyle="#E74C3C";ctx.lineWidth=2;ctx.setLineDash([5,4]);
    ctx.strokeRect(rx+8,ry+8,rw-16,rh-16);ctx.setLineDash([]);
    ctx.fillStyle="#E74C3C";
    ctx.font="italic bold 12px 'Segoe UI','Microsoft YaHei'";
    ctx.textAlign="center";ctx.textBaseline="middle";
    ctx.fillText("VACANT", cx, cy);
  }
}

// ========== ASSIGNMENT LINES + ARROWS ==========
function drawAssignmentLines(){
  var uavs=ALL_FRAMES[currentFrame].snapshot.uavs;
  for(var uid in uavs){
    var u=uavs[uid];
    if(!u.alive)continue;
    var color=UAV_COLORS[uid]||"#888";
    var regions=u.regions||[];
    for(var i=0;i<regions.length;i++){
      var rid=regions[i];
      var b=REGION_BOUNDS[rid];
      var rcx=(b[0]+b[1])/2, rcy=(b[2]+b[3])/2;
      var pos=UAV_DRAW_POS[uid];
      var fromX=pos?pos.ux:tx(u.x);
      var fromY=pos?pos.uy:ty(u.y);
      var toX=tx(rcx), toY=ty(rcy);
      ctx.strokeStyle=color;ctx.lineWidth=2;
      ctx.setLineDash([7,5]);ctx.globalAlpha=0.5;
      ctx.beginPath();ctx.moveTo(fromX,fromY);ctx.lineTo(toX,toY);ctx.stroke();
      ctx.setLineDash([]);ctx.globalAlpha=1;
      drawArrowHead(fromX,fromY,toX,toY,color);
    }
  }
}

var UAV_DRAW_POS = {};  // {uid: {ux, uy}} — final on-screen position for each alive UAV

function computeRegionLayout(){
  UAV_DRAW_POS = {};
  var uavs = ALL_FRAMES[currentFrame].snapshot.uavs;
  var regions = ALL_FRAMES[currentFrame].snapshot.regions;

  // For each region, collect UAVs that belong there:
  //   a) UAV physically at/near this region center
  //   b) UAV assigned to search this region
  var regionUAVs = {0:[], 1:[], 2:[], 3:[]};
  var seen = {};  // uid -> true, avoid duplicates

  for(var rid=0; rid<4; rid++){
    var b = REGION_BOUNDS[rid];
    var rcx = (b[0]+b[1])/2, rcy = (b[2]+b[3])/2;  // data coords

    // a) UAVs whose physical position is near this region center
    for(var uid in uavs){
      var u = uavs[uid];
      if(!u.alive) continue;
      var d = Math.sqrt(Math.pow(u.x-rcx,2) + Math.pow(u.y-rcy,2));
      if(d < 8){  // within 8 data-units of region center
        regionUAVs[rid].push(parseInt(uid));
        seen[uid] = true;
      }
    }

    // b) Assigned search UAV for this region (if not already in list)
    var r = regions[String(rid)];
    var assigned = r.assigned_uav;
    if(assigned !== -1 && !seen[assigned]){
      // Check this UAV is alive and not already placed elsewhere
      if(uavs[String(assigned)] && uavs[String(assigned)].alive){
        regionUAVs[rid].push(assigned);
        seen[assigned] = true;
      }
    }
  }

  // Assign draw positions within each region (spread around center)
  for(var rid=0; rid<4; rid++){
    var list = regionUAVs[rid];
    var b = REGION_BOUNDS[rid];
    var cx = tx((b[0]+b[1])/2), cy = ty((b[2]+b[3])/2);
    var n = list.length;
    if(n === 0) continue;
    if(n === 1){
      UAV_DRAW_POS[list[0]] = {ux: cx, uy: cy};
    } else {
      var radius = 18;
      for(var i=0; i<n; i++){
        var angle = (2*Math.PI*i)/n - Math.PI/2;
        UAV_DRAW_POS[list[i]] = {
          ux: cx + Math.cos(angle)*radius,
          uy: cy + Math.sin(angle)*radius
        };
      }
    }
  }
}

// ========== UAVs ==========
function drawUAVs(){
  var uavs=ALL_FRAMES[currentFrame].snapshot.uavs;
  for(var uid in uavs){
    var u=uavs[uid];
    var color=UAV_COLORS[uid]||"#888";
    var light=UAV_LIGHT[uid]||"#eee";

    // use computed layout position (shared with region drones)
    var pos=UAV_DRAW_POS[uid];
    var ux=pos?pos.ux:tx(u.x);
    var uy=pos?pos.uy:ty(u.y);

    if(!u.alive){
      var rid=nearestRegion(u.x,u.y);
      if(rid!==null){
        var b=REGION_BOUNDS[rid];
        ux=tx(b[1]-4); uy=ty(b[2]+4);
        // dead UAVs with same region: offset vertically
        var deadInSameRegion = 0;
        for(var uid2 in uavs){
          if(uid2===uid)break;
          var u2=uavs[uid2];
          if(!u2.alive && nearestRegion(u2.x,u2.y)===rid) deadInSameRegion++;
        }
        uy += deadInSameRegion * 14;
      }
      drawDestroyedMark(ux,uy);
      ctx.fillStyle="#aaa";ctx.font="9px 'Segoe UI','Microsoft YaHei'";
      ctx.textAlign="right";ctx.textBaseline="top";
      ctx.fillText("U"+uid+" [DEAD]", ux-5, uy+8);
      continue;
    }

    // drone icon
    drawDrone(ux, uy, color, light, 1.05);

    // label
    var labelX=ux+26, labelY=uy-14;
    ctx.fillStyle=color;
    ctx.font="bold 11px 'Segoe UI','Microsoft YaHei'";
    ctx.textAlign="left";ctx.textBaseline="bottom";
    ctx.fillText("U"+uid, labelX, labelY);
    var sensor=u.sensor===0?"EO":"SAR";
    var task=u.task===0?"SEARCH":(u.task===1?"TRACK":"IDLE");
    ctx.fillStyle="#888";
    ctx.font="9px 'Segoe UI','Microsoft YaHei'";
    ctx.fillText(sensor+" | "+task, labelX, labelY+13);

    // load badge
    var n=u.regions?u.regions.length:0;
    if(n>0){
      ctx.beginPath();ctx.arc(ux-10,uy-11,8.5,0,Math.PI*2);
      ctx.fillStyle=color;ctx.globalAlpha=0.92;ctx.fill();
      ctx.globalAlpha=1;
      ctx.fillStyle="#fff";ctx.font="bold 10px 'Segoe UI','Microsoft YaHei'";
      ctx.textAlign="center";ctx.textBaseline="middle";
      ctx.fillText(String(n), ux-10, uy-11);
    }
  }
}

// ========== TARGETS ==========
function drawTargets(){
  var targets=ALL_FRAMES[currentFrame].snapshot.targets;
  for(var tid in targets){
    var t=targets[tid];
    var ttx=tx(t.x), tty=ty(t.y);
    var meta=TARGET_META[tid]||{type:"?",movable:false};
    var labelX=ttx+16, labelY=tty-7;

    if(t.destroyed){
      ctx.fillStyle="#bdc3c7";ctx.fillRect(ttx-7,tty-7,14,14);
      ctx.strokeStyle="#fff";ctx.lineWidth=2;ctx.strokeRect(ttx-7,tty-7,14,14);
      ctx.fillStyle="#888";ctx.font="bold 9px 'Segoe UI','Microsoft YaHei'";
      ctx.textAlign="left";ctx.textBaseline="bottom";
      ctx.fillText("T"+tid+" ("+meta.type+")",labelX,labelY);
      ctx.fillStyle="#aaa";ctx.font="7px 'Segoe UI','Microsoft YaHei'";
      ctx.fillText("DESTROYED",labelX,labelY+11);
    } else if(t.tracked){
      drawTriangle(ttx,tty,9,"#E74C3C","#fff");
      ctx.fillStyle="#c0392b";ctx.font="bold 9px 'Segoe UI','Microsoft YaHei'";
      ctx.textAlign="left";ctx.textBaseline="bottom";
      ctx.fillText("T"+tid+" ("+meta.type+")",labelX,labelY);
      ctx.fillStyle="#e74c3c";ctx.font="7px 'Segoe UI','Microsoft YaHei'";
      ctx.fillText("TRACKED",labelX,labelY+11);
    } else if(t.discovered){
      drawTriangle(ttx,tty,9,"#F39C12","#fff");
      ctx.fillStyle="#e67e22";ctx.font="bold 9px 'Segoe UI','Microsoft YaHei'";
      ctx.textAlign="left";ctx.textBaseline="bottom";
      ctx.fillText("T"+tid+" ("+meta.type+")",labelX,labelY);
      ctx.fillStyle="#f39c12";ctx.font="7px 'Segoe UI','Microsoft YaHei'";
      ctx.fillText("DISC",labelX,labelY+11);
    } else {
      ctx.beginPath();ctx.arc(ttx,tty,7,0,Math.PI*2);
      ctx.strokeStyle="#bbb";ctx.lineWidth=1.2;ctx.globalAlpha=0.6;
      ctx.setLineDash([3,3]);ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;
      ctx.fillStyle="#aaa";ctx.font="italic 8px 'Segoe UI','Microsoft YaHei'";
      ctx.textAlign="left";ctx.textBaseline="bottom";
      ctx.fillText("T"+tid+"?",labelX,labelY);
    }
  }
}

function drawTriangle(x,y,size,fillColor,strokeColor){
  ctx.beginPath();
  ctx.moveTo(x,y-size);
  ctx.lineTo(x+size*0.87,y+size*0.5);
  ctx.lineTo(x-size*0.87,y+size*0.5);
  ctx.closePath();
  ctx.fillStyle=fillColor;ctx.fill();
  ctx.strokeStyle=strokeColor;ctx.lineWidth=1.5;ctx.stroke();
}

function nearestRegion(x,y){
  var best=null,bestD=Infinity;
  for(var rid in REGION_BOUNDS){
    var b=REGION_BOUNDS[rid];
    var cx=(b[0]+b[1])/2, cy=(b[2]+b[3])/2;
    var d=(x-cx)*(x-cx)+(y-cy)*(y-cy);
    if(d<bestD){bestD=d;best=parseInt(rid);}
  }
  return best;
}

// ========== RENDER ==========
var currentFrame=0;
var totalFrames=ALL_FRAMES.length;
var slider=document.getElementById("slider");
slider.max=totalFrames-1;

function render(){
  ctx.clearRect(0,0,CANVAS_W,CANVAS_H);
  // canvas bg
  ctx.fillStyle="#fcfcfc";ctx.fillRect(0,0,CANVAS_W,CANVAS_H);

  computeRegionLayout();
  drawGrid();
  for(var rid=0;rid<4;rid++)drawRegion(rid);
  drawAssignmentLines();
  drawTargets();
  drawUAVs();

  // map border
  ctx.strokeStyle="#b0b8c0";ctx.lineWidth=1.8;
  ctx.strokeRect(tx(0),ty(AREA_SIZE),AREA_SIZE*SCALE,AREA_SIZE*SCALE);

  // axis labels
  ctx.fillStyle="#999";ctx.font="10px 'Segoe UI','Microsoft YaHei'";
  ctx.textAlign="center";ctx.textBaseline="top";
  for(var v=0;v<=50;v+=10){
    ctx.fillText(String(v),tx(v),ty(0)+3);
  }
  ctx.textAlign="right";ctx.textBaseline="middle";
  for(var v=0;v<=50;v+=10){
    ctx.fillText(String(v),tx(0)-6,ty(v));
  }

  // axis titles
  ctx.fillStyle="#666";ctx.font="bold 11px 'Segoe UI','Microsoft YaHei'";
  ctx.textAlign="center";ctx.fillText("X", CANVAS_W/2, CANVAS_H-10);
  ctx.save();ctx.translate(14,CANVAS_H/2);ctx.rotate(-Math.PI/2);
  ctx.fillText("Y",0,0);ctx.restore();

  // update info panel
  var frame=ALL_FRAMES[currentFrame];
  document.getElementById("round-title").textContent=frame.title;
  document.getElementById("event-text").textContent=frame.event||"";
  document.getElementById("action-text").textContent=frame.action||"";
  document.getElementById("frame-label").textContent=(currentFrame+1)+" / "+totalFrames;
  slider.value=currentFrame;

  // update legend dynamically
  buildLegend();
}

function goToFrame(idx){
  currentFrame=Math.max(0,Math.min(totalFrames-1,idx));
  render();
}

// ========== CONTROLS ==========
var playing=false;
var playTimer=null;

document.getElementById("btn-play").onclick=function(){
  var btn=document.getElementById("btn-play");
  if(playing){
    playing=false;clearInterval(playTimer);
    btn.textContent="▶ Play";btn.classList.remove("active");
  } else {
    playing=true;
    btn.textContent="⏸ Pause";btn.classList.add("active");
    playTimer=setInterval(function(){
      if(currentFrame>=totalFrames-1)goToFrame(0);
      else goToFrame(currentFrame+1);
    }, 2000);
  }
};

document.getElementById("btn-prev").onclick=function(){goToFrame(currentFrame-1);};
document.getElementById("btn-next").onclick=function(){goToFrame(currentFrame+1);};
slider.oninput=function(){goToFrame(parseInt(slider.value));};

document.onkeydown=function(e){
  if(e.key==="ArrowLeft"){e.preventDefault();goToFrame(currentFrame-1);}
  if(e.key==="ArrowRight"){e.preventDefault();goToFrame(currentFrame+1);}
  if(e.key===" "){e.preventDefault();document.getElementById("btn-play").click();}
};

// ========== LEGEND (dynamic) ==========
function buildLegend(){
  var container=document.getElementById("legend");
  var uavs=ALL_FRAMES[currentFrame].snapshot.uavs;
  var html='';
  // alive UAVs first
  for(var uid in uavs){
    var u=uavs[uid];
    if(!u.alive)continue;
    var sensor=u.sensor===0?"EO":"SAR";
    var task=u.task===0?"SEARCH":(u.task===1?"TRACK":"IDLE");
    var n=u.regions?u.regions.length:0;
    html+='<div class="leg-item">'
      +'<span style="display:inline-block;width:13px;height:13px;border-radius:50%;background:'+UAV_COLORS[uid]+';border:2px solid '+UAV_COLORS[uid]+'"></span>'
      +' U'+uid+' '+sensor+' '+task+' ['+n+'区域]</div>';
  }
  // dead UAVs
  for(var uid in uavs){
    var u=uavs[uid];
    if(u.alive)continue;
    html+='<div class="leg-item"><span style="color:#bbb;font-size:1.1em">✕</span> U'+uid+' DESTROYED</div>';
  }
  html+='<div class="leg-item" style="margin-left:6px;color:#aaa">|</div>';
  html+='<div class="leg-item"><span style="color:#bbb">○</span> Undiscovered</div>';
  html+='<div class="leg-item"><span style="color:#E74C3C">▲</span> Tracked</div>';
  html+='<div class="leg-item"><span style="color:#f39c12">☀</span> Sunny</div>';
  html+='<div class="leg-item"><span style="color:#5090d0">☁</span> Rainy</div>';
  container.innerHTML=html;
}

// ========== INIT ==========
goToFrame(0);
</script>
</body>
</html>"""


def save_loop_animation_html(initial_snapshot: dict, round_records: list, output_path: str,
                              subtitle: str = ""):
    """生成循环测试 HTML 动画页面。

    将所有帧数据嵌入 HTML，使用 Canvas + JavaScript 实现交互动画：
      - Play/Pause 自动播放（2 秒/帧）
      - Prev/Next 逐帧切换
      - 滑块跳转
      - 键盘 ← → 空格 控制

    Args:
        initial_snapshot: 初始场景快照
        round_records:    [{"round": 1, "event": "...", "action_summary": "...",
                            "snapshot": {...}}, ...]
        output_path:      输出 HTML 路径
        subtitle:         场景描述文本（根据当前态势自动生成）
    """
    # 根据初始快照构建 target meta
    targets_data = initial_snapshot.get("targets", {})
    target_meta = {}
    for tid, t in targets_data.items():
        ttype = "CMD" if t.get("target_type") == 1 else "CAR"
        target_meta[int(tid)] = {"type": ttype, "movable": ttype == "CAR"}

    # 构建场景描述文本
    if not subtitle:
        subtitle = _build_scenario_subtitle(initial_snapshot)

    frames = [{
        "title": "Initial State",
        "event": "",
        "action": subtitle,
        "snapshot": _strip_snapshot(initial_snapshot),
    }]

    for rec in round_records:
        r = rec["round"]
        frames.append({
            "title": f"Round {r} — After PPO Reallocation",
            "event": f"Event: {rec['event']}",
            "action": rec.get("action_summary", ""),
            "snapshot": _strip_snapshot(rec["snapshot"]),
        })

    frames_json = json.dumps(frames, ensure_ascii=False, indent=2)
    target_meta_json = json.dumps(target_meta, ensure_ascii=False)

    html = _ANIM_HTML_TEMPLATE.replace("__FRAMES_JSON__", frames_json)
    html = html.replace("__SUBTITLE__", subtitle)
    html = html.replace("__TARGET_META_JSON__", target_meta_json)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return str(output_path)


def _build_scenario_subtitle(snapshot: dict) -> str:
    """根据初始快照自动生成场景描述文本。"""
    uavs = snapshot.get("uavs", {})
    regions = snapshot.get("regions", {})
    targets = snapshot.get("targets", {})

    alive_uavs = []
    dead_uavs = []
    for uid, u in uavs.items():
        sensor = "EO" if u.get("sensor") == 0 else "SAR"
        if u.get("alive"):
            regs = [f"R{r}" for r in sorted(u.get("regions", []))]
            alive_uavs.append(f"U{uid}({sensor})→{'&'.join(regs) if regs else 'IDLE'}")
        else:
            dead_uavs.append(f"U{uid}")

    parts = []
    if alive_uavs:
        parts.append(" | ".join(alive_uavs))
    else:
        parts.append("No alive UAVs")
    if dead_uavs:
        parts.append(f"[Destroyed: {', '.join(dead_uavs)}]")

    n_targets = len(targets)
    n_discovered = sum(1 for t in targets.values() if t.get("discovered"))
    parts.append(f"Targets: {n_targets} ({n_discovered} discovered)")

    return "Scenario: " + "  |  ".join(parts)


def _strip_snapshot(snap: dict) -> dict:
    """精简快照，只保留可视化需要的字段。"""
    s = copy.deepcopy(snap)
    for uid in s.get("uavs", {}):
        u = s["uavs"][uid]
        u.pop("sensor_failed", None)
    for tid in s.get("targets", {}):
        t = s["targets"][tid]
        t.pop("movable", None)
        t.pop("tracker_id", None)
    s.pop("event", None)
    return s
