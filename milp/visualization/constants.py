"""
可视化常量 —— 颜色、图标、地图尺寸等集中管理。
所有 Plotly 绘图和 Streamlit 前端统一引用此处常量。
"""

# ── 地图尺寸 ──────────────────────────────────────────
MAP_SIZE_KM = 300.0       # 任务区边长 (km)
AOI_SIZE_KM = 50.0        # 单个 AOI 边长 (km)
SUB_SIZE_KM = 25.0        # 子区边长 (km)
GRID_LINE_INTERVAL = 50.0 # AOI 网格线间距 (km)

# ── 平台 / 目标颜色 ──────────────────────────────────
COLOR_UAV        = "#1f77b4"   # 蓝
COLOR_HELI       = "#ff7f0e"   # 橙
COLOR_TARGET     = "#d62728"   # 红
COLOR_GRID       = "#888888"   # 灰
COLOR_STAGING    = "#000000"   # 黑
COLOR_AOI_FILL   = "rgba(255, 255, 0, 0.15)"  # 淡黄半透明
COLOR_AOI_BORDER = "#ffcc00"   # 金黄
COLOR_UAV_LINE   = "#1f77b4"   # UAV 路径线 (蓝)
COLOR_HELI_LINE  = "#ff7f0e"   # HELI 路径线 (橙)
COLOR_ALLOCATION_LINE = "#aaaaaa"  # 分配连线 (浅灰)

# ── 多 AOI 状态颜色 ────────────────────────────────────
COLOR_AOI_CURRENT_FILL   = "rgba(255, 255, 0, 0.18)"   # 当前 AOI 淡黄
COLOR_AOI_CURRENT_BORDER = "#ffcc00"                    # 当前 AOI 金黄边框
COLOR_AOI_FINISHED_FILL  = "rgba(0, 200, 0, 0.12)"     # 已完成 AOI 淡绿
COLOR_AOI_FINISHED_BORDER = "#2ca02c"                   # 已完成 AOI 绿边框
COLOR_AOI_PENDING_FILL   = "rgba(180, 180, 180, 0.08)" # 待执行 AOI 浅灰
COLOR_AOI_PENDING_BORDER = "#999999"                    # 待执行 AOI 灰边框
COLOR_AOI_ROUTE_LINE     = "#555555"                    # AOI 路线深灰
COLOR_AOI_ROUTE_ARROW    = "#333333"                    # AOI 路线箭头

# ── Plotly marker 符号 ───────────────────────────────
SYMBOL_UAV    = "circle"
SYMBOL_HELI   = "triangle-up"
SYMBOL_TARGET = "x"
SYMBOL_STAGING = "square"

# ── 标记大小 ──────────────────────────────────────────
MARKER_SIZE_PLATFORM = 12
MARKER_SIZE_TARGET   = 14
MARKER_SIZE_STAGING  = 16
MARKER_SIZE_GRID     = 8

# ── 线条样式 ──────────────────────────────────────────
LINE_WIDTH_ROUTE       = 2.5
LINE_WIDTH_ALLOCATION  = 1.2
LINE_WIDTH_GRID        = 0.5
LINE_WIDTH_AOI_BORDER  = 2.0
LINE_DASH_ALLOCATION   = "dot"
LINE_DASH_GRID_C0      = "dash"

# ── 动画参数 ──────────────────────────────────────────
N_ANIM_STEPS = 20          # t2 阶段帧数
