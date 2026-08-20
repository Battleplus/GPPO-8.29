#!/bin/bash
# SAR无人机搜索路径规划 — 运行脚本
# 用法: bash run.sh
#
# 修改下面这行数字即可切换示例 (1~6):
#   1 = 跑道形搜索 (racetrack)
#   2 = 多边形搜索 (sar_polygon)
#   3 = 圆角多边形搜索 (sar_rounded)
#   4 = 8字形搜索 (figure_eight)
#   5 = 同区域对比四种路径模式
#   6 = 四机田字格协同搜索 (25×25km ×4)
#   7 = 单机多区域循环调度（任务重分配）
#   8 = 分阶段任务：先四机各搜一格，两圈后重分配（2机各管两区）
# QL_SAR_VIZ_RUN_SECONDS=1800 bash run.sh 

ACTIVE_EXAMPLE=${ACTIVE_EXAMPLE:-7}
export SAR_SHOW_PATHS="${SAR_SHOW_PATHS:-1}"  # 1=显示轨迹, 0=隐藏

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
set -e
cd "$SCRIPT_DIR/.."

echo "=============================================="
echo "  SAR 无人机搜索路径规划"
echo "  当前示例: ${ACTIVE_EXAMPLE}"
echo "=============================================="
echo ""

export ACTIVE_EXAMPLE

MISSION_PLATFORM="${SAR_MISSION_PLATFORM:-Blue_CH4_Recon}"
export SAR_MISSION_PLATFORM="$MISSION_PLATFORM"

python -c "
from sar_search_planner import PlannerConfig, plan, plan_mission, compute_start_near_area
import json, dataclasses, os

active = int(os.environ.get('ACTIVE_EXAMPLE', '2'))
platform_id = os.environ.get('SAR_MISSION_PLATFORM', 'Blue_Quad_Recon_1')
config = None

if active == 1:
    print('【示例1】跑道形搜索 (racetrack)')
    print('  区域: 中心(60,-30)km，20km×15km')
    config = PlannerConfig(
        area_center_km=(60, -30),
        area_width_km=25,
        area_height_km=25,
        pattern='racetrack',
        angle_deg=30,
        altitude_agl_m=6000,
        racetrack_length_km=14,
        racetrack_width_km=10,
    )

elif active == 2:
    print('【示例2】多边形搜索 (sar_polygon)')
    print('  区域: 中心(60,-30)km，20km×15km')
    config = PlannerConfig(
        area_center_km=(60, -30),
        area_width_km=25,
        area_height_km=25,
        pattern='sar_polygon',
        sar_radius_km=8,
        sar_sides=6,
        sar_loops=1,
        altitude_agl_m=5000,
    )

elif active == 3:
    print('【示例3】圆角多边形搜索 (sar_rounded)')
    print('  区域: 中心(60,-30)km，20km×15km')
    config = PlannerConfig(
        area_center_km=(60, -30),
        area_width_km=25,
        area_height_km=25,
        pattern='sar_rounded',
        sar_radius_km=8,
        sar_sides=6,
        sar_turn_radius_km=4,
        sar_loops=1,
        altitude_agl_m=5000,
    )

elif active == 4:
    print('【示例4】8字形搜索 (figure_eight)')
    print('  区域: 中心(60,-30)km，20km×15km')
    config = PlannerConfig(
        area_center_km=(60, -30),
        area_width_km=25,
        area_height_km=25,
        pattern='figure_eight',
        eight_radius_km=5,
        eight_line_km=14,
        eight_loops=1,
        angle_deg=30,
        altitude_agl_m=5000,
    )

elif active == 5:
    print('【示例5】同区域对比四种路径模式（可视化叠加显示）')
    print('  区域: 中心(0,0)，30km×30km')
    configs = []
    for pattern in ['racetrack', 'sar_polygon', 'sar_rounded', 'figure_eight']:
        c = PlannerConfig(
            area_center_km=(0, 0),
            area_width_km=25,
            area_height_km=25,
            pattern=pattern,
            altitude_agl_m=6000,
        )
        r = plan(c)
        print(f'  {pattern:15s} -> {len(r.waypoints):3d} 航点,  {r.stats[\"path_length_km\"]:6.1f} km,  {r.stats[\"mountains_in_area\"]} 座山')
        configs.append(dataclasses.asdict(c))
    print()

    with open('sar_search_planner/_active_config.json', 'w') as f:
        json.dump(configs, f, indent=2)
    print('==============================================')
    print('  规划完成（4种路径将叠加显示）')
    print('==============================================')
    exit(0)

elif active == 6:
    # ── 田字格中心坐标──
    GRID_CX = 60.0
    GRID_CY = 30.0
    HALF = 12.5  # 半格宽度 25km / 2

    print('【示例6】四机田字格协同搜索')
    print(f'  整体: 50×50km，中心({GRID_CX:.0f},{GRID_CY:.0f})km，分4格各25×25km')
    print()

    quadrants: list[dict] = [
        {
            '_platform_id': 'Blue_CH4_Recon',
            'area_center_km': (GRID_CX - HALF, GRID_CY + HALF),
            'area_width_km': 25, 'area_height_km': 25,
            'pattern': 'racetrack',
        },
        {
            '_platform_id': 'Blue_CH4_Recon_2',
            'area_center_km': (GRID_CX + HALF, GRID_CY + HALF),
            'area_width_km': 25, 'area_height_km': 25,
            'pattern': 'sar_polygon',
        },
        {
            '_platform_id': 'Blue_CH4_StrikeRecon',
            'area_center_km': (GRID_CX - HALF, GRID_CY - HALF),
            'area_width_km': 25, 'area_height_km': 25,
            'pattern': 'sar_rounded',
        },
        {
            '_platform_id': 'Blue_CH4_StrikeRecon_2',
            'area_center_km': (GRID_CX + HALF, GRID_CY - HALF),
            'area_width_km': 25, 'area_height_km': 25,
            'pattern': 'figure_eight',
        },
    ]

    for q in quadrants:
        pid = q.pop('_platform_id')
        cfg_keys = {k: v for k, v in q.items()}
        c = PlannerConfig(**cfg_keys)
        r = plan(c)
        q['_platform_id'] = pid  # restore for JSON
        print(f'  {pid:25s} | {q[\"pattern\"]:15s} | '
              f'中心({q[\"area_center_km\"][0]:.0f},{q[\"area_center_km\"][1]:.0f}) | '
              f'{len(r.waypoints):3d} 航点 | {r.stats[\"path_length_km\"]:5.1f} km | '
              f'{r.stats[\"mountains_in_area\"]} 座山')

    print()
    with open('sar_search_planner/_active_config.json', 'w') as f:
        json.dump(quadrants, f, indent=2)
    print('==============================================')
    print('  规划完成（四机田字格协同搜索）')
    print('==============================================')
    exit(0)

elif active == 7:
    # ── 田字格中心坐标──
    GRID_CX = 50.0
    GRID_CY = 30.0
    HALF = 12.5

    print('【示例7】任务重分配：Blue_CH4_Recon接管NW+NE，Blue_CH4_StrikeRecon接管SW+SE')
    print(f'  田字格中心({GRID_CX:.0f},{GRID_CY:.0f})，4格各25×25km')
    print(f'  Blue_CH4_Recon_2, Blue_CH4_StrikeRecon_2 → 待命（模拟损毁）')
    print()

    quadrants: list[dict] = [
        {
            '_platform_id': 'Blue_CH4_Recon',
            'area_center_km': (GRID_CX - HALF, GRID_CY + HALF),
            'area_width_km': 25, 'area_height_km': 25,
            'pattern': 'racetrack',
        },
        {
            '_platform_id': 'Blue_CH4_Recon_2',
            'area_center_km': (GRID_CX + HALF, GRID_CY + HALF),
            'area_width_km': 25, 'area_height_km': 25,
            'pattern': 'racetrack',
        },
        {
            '_platform_id': 'Blue_CH4_StrikeRecon',
            'area_center_km': (GRID_CX - HALF, GRID_CY - HALF),
            'area_width_km': 25, 'area_height_km': 25,
            'pattern': 'figure_eight',
        },
        {
            '_platform_id': 'Blue_CH4_StrikeRecon',
            'area_center_km': (GRID_CX + HALF, GRID_CY - HALF),
            'area_width_km': 25, 'area_height_km': 25,
            'pattern': 'figure_eight',
        },
    ]

    for q in quadrants:
        pid = q.pop('_platform_id')
        cfg_keys = {k: v for k, v in q.items()}
        c = PlannerConfig(**cfg_keys)
        r = plan(c)
        q['_platform_id'] = pid
        print(f'  {pid:25s} | {q[\"pattern\"]:15s} | '
              f'中心({q[\"area_center_km\"][0]:.0f},{q[\"area_center_km\"][1]:.0f}) | '
              f'{len(r.waypoints):3d} 航点 | {r.stats[\"path_length_km\"]:5.1f} km')

    print()
    with open('sar_search_planner/_active_config.json', 'w') as f:
        json.dump(quadrants, f, indent=2)
    print('==============================================')
    print('  规划完成（2机各负责2区，循环搜索）')
    print('==============================================')
    exit(0)

elif active == 8:
    # ── 田字格中心坐标──
    GRID_CX = 50.0
    GRID_CY = 30.0
    HALF = 12.5

    print('【示例8】分阶段任务：4机各搜一格 → 两圈后 2机各管两区')
    print(f'  田字格中心({GRID_CX:.0f},{GRID_CY:.0f})，4格各25×25km')
    print()

    # Phase 1: normal 4-drone search (same as example 6)
    phase1: list[dict] = [
        {'_platform_id': 'Blue_CH4_Recon',        'area_center_km': (GRID_CX - HALF, GRID_CY + HALF), 'area_width_km': 25, 'area_height_km': 25, 'pattern': 'racetrack'},
        {'_platform_id': 'Blue_CH4_Recon_2',      'area_center_km': (GRID_CX + HALF, GRID_CY + HALF), 'area_width_km': 25, 'area_height_km': 25, 'pattern': 'racetrack'},
        {'_platform_id': 'Blue_CH4_StrikeRecon',   'area_center_km': (GRID_CX - HALF, GRID_CY - HALF), 'area_width_km': 25, 'area_height_km': 25, 'pattern': 'figure_eight'},
        {'_platform_id': 'Blue_CH4_StrikeRecon_2', 'area_center_km': (GRID_CX + HALF, GRID_CY - HALF), 'area_width_km': 25, 'area_height_km': 25, 'pattern': 'figure_eight'},
    ]

    # Phase 2: reassignment (same as example 7 but with user's config)
    phase2: list[dict] = [
        {'_platform_id': 'Blue_CH4_Recon',        'area_center_km': (GRID_CX - HALF, GRID_CY + HALF), 'area_width_km': 25, 'area_height_km': 25, 'pattern': 'racetrack'},
        {'_platform_id': 'Blue_CH4_Recon_2',        'area_center_km': (GRID_CX + HALF, GRID_CY + HALF), 'area_width_km': 25, 'area_height_km': 25, 'pattern': 'racetrack'},
        {'_platform_id': 'Blue_CH4_StrikeRecon',   'area_center_km': (GRID_CX - HALF, GRID_CY - HALF), 'area_width_km': 25, 'area_height_km': 25, 'pattern': 'figure_eight'},
        {'_platform_id': 'Blue_CH4_StrikeRecon',   'area_center_km': (GRID_CX + HALF, GRID_CY - HALF), 'area_width_km': 25, 'area_height_km': 25, 'pattern': 'figure_eight'},
    ]

    for label, phase in [(\"Phase1\", phase1), (\"Phase2\", phase2)]:
        print(f'  {label}:')
        for q in phase:
            pid = q['_platform_id']
            pm = q['pattern']
            cx, cy = q['area_center_km']
            print(f'    {pid:25s} | {pm:15s} | 中心({cx:.0f},{cy:.0f})')
    print()
    config_out = [
        {\"phase_label\": \"4-drone search\", \"phase_laps\": 5, \"quadrants\": phase1},
        {\"phase_label\": \"reassign 2 drones\", \"phase_laps\": -1, \"quadrants\": phase2},
    ]
    with open('sar_search_planner/_active_config.json', 'w') as f:
        json.dump(config_out, f, indent=2)
    print('==============================================')
    print('  规划完成（分阶段：Phase1×2圈 → Phase2×持续）')
    print('==============================================')
    exit(0)

else:
    print(f'无效示例号: {active}，请设置 ACTIVE_EXAMPLE=1~8')
    exit(1)

result = plan(config)
print(f'  航点数: {len(result.waypoints)}')
print(f'  路径总长: {result.stats[\"path_length_km\"]:.1f} km')
print(f'  区域内山体: {len(result.mountains_in_area)} 座')
for m in result.mountains_in_area:
    print(f'    - {m.obstacle_id} (r={m.radius_units:.0f}u, h={m.height_units:.1f}u)')
print(f'  碰撞(前/后): {result.stats[\"collision_count_before\"]}/{result.stats[\"collision_count_after\"]}')

# ── Mission planning ──────────────────────────
print()
print('--- 任务规划 ---')
print(f'  平台: {platform_id}')
sx, sy, sz = compute_start_near_area(
    result.search_area,
    offset_km=15.0,
    direction_deg=225.0,
    altitude_agl_m=config.altitude_agl_m,
    meters_per_unit=config.meters_per_unit,
)
print(f'  初始位置: ({sx:.0f}, {sy:.0f}, {sz:.1f}) 场景单位 (搜索区域西南方15km外)')
mission = plan_mission(result, sx, sy, sz, platform_id)
print(f'  最佳切入点: 航点 #{mission.entry_index} (共{len(result.waypoints)}个)')
print(f'  过渡距离: {mission.transit_distance_km:.1f} km')
print(f'  任务总长: {mission.transit_distance_km + result.stats[\"path_length_km\"]:.1f} km')
print(f'  切入点坐标: ({mission.search_waypoints[0].x:.0f}, {mission.search_waypoints[0].y:.0f})')

# 保存配置和任务数据
with open('sar_search_planner/_active_config.json', 'w') as f:
    json.dump(dataclasses.asdict(config), f, indent=2)

mission_data = {
    'platform_id': platform_id,
    'start_x': sx, 'start_y': sy, 'start_z': sz,
    'entry_index': mission.entry_index,
    'transit_distance_km': mission.transit_distance_km,
    'mission_total_km': mission.transit_distance_km + result.stats['path_length_km'],
}
with open('sar_search_planner/_active_mission.json', 'w') as f:
    json.dump(mission_data, f, indent=2)

print()
print('==============================================')
print('  规划完成')
print('==============================================')
"

echo ""
echo "Starting simple phased 3D SAR visualization..."
python "$SCRIPT_DIR/run_multi_drone_simple_viz.py"
# /home/isaac/isaacsim/python.sh "$SCRIPT_DIR/run_mission_viz.py"
