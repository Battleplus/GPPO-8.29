"""
端到端测试: 构造 SituationSnapshot → MILPAllocator.solve() → AllocationPlan。

运行方式:
    cd 代码-v2
    python -m pytest tests/test_end_to_end.py -v
    或
    python tests/test_end_to_end.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import GlobalSettings, SolverType
from core.snapshot import (
    SituationSnapshot, TargetInfo, PlatformInfo, SensorParams,
    generate_aoi_grids,
)
from core.allocation import AllocationPlan, ReconAssignment, StrikeAssignment
from allocation.milp_allocator import MILPAllocator
from task_interface import load_snapshot_from_json, load_all_scenarios


def build_sample_snapshot(with_targets: bool = True) -> SituationSnapshot:
    """
    构造一个包含 5 架 UAV、2 架 HELI、4 个目标和 5 个栅格的态势快照。
    场景: AOI A_3_4（第 3 行第 4 列, 中心约 175,125），平台从集结区出发。
    """
    # 使用 AOI 网格生成函数
    grids = generate_aoi_grids(aoi_row=3, aoi_col=4)

    # 覆盖各子区天气以测试多传感器择优
    grids[1].weather_w = 0.15  # c1: 晴好, EO 占优
    grids[2].weather_w = 0.40  # c2: 轻雾
    grids[3].weather_w = 0.55  # c3: 多云, SAR 占优
    grids[4].weather_w = 0.70  # c4: 厚雾, SAR 最优

    staging = np.array([150.0, -50.0])  # 集结区: 任务区南侧外 50 km

    targets = []
    if with_targets:
        # 目标分散在 300×300 任务区的不同 AOI 区域
        targets = [
            TargetInfo(
                tid="g1", type="RADAR",
                pos_est=np.array([270.0, 260.0]),  # A_6_6 东北角
                pos_cov=np.eye(2) * 0.1,
                velocity_est=np.zeros(2),
                confirmed=True, alive=True,
                value=1.0, threat=0.9
            ),
            TargetInfo(
                tid="g2", type="CP",
                pos_est=np.array([35.0, 230.0]),   # A_5_1 西北侧
                pos_cov=np.eye(2) * 0.1,
                velocity_est=np.zeros(2),
                confirmed=True, alive=True,
                value=0.85, threat=0.6
            ),
            TargetInfo(
                tid="g3", type="AV",
                pos_est=np.array([220.0, 45.0]),   # A_1_5 东南侧
                pos_cov=np.eye(2) * 0.2,
                velocity_est=np.array([0.02, 0.01]),
                confirmed=True, alive=True,
                value=0.7, threat=0.5
            ),
            TargetInfo(
                tid="g4", type="AV",
                pos_est=np.array([60.0, 80.0]),    # A_2_2 西南侧
                pos_cov=np.eye(2) * 0.2,
                velocity_est=np.array([-0.01, 0.02]),
                confirmed=True, alive=True,
                value=0.7, threat=0.5
            ),
        ]

    platforms = [
        PlatformInfo(pid="U1", type="UAV", pos=staging.copy(),
                     alt=2.0, lost=False,
                     sensors_mounted=["EO", "SAR", "ESM"],
                     munitions={"HF": 0, "RKT": 0, "GUN": 0}),
        PlatformInfo(pid="U2", type="UAV", pos=staging.copy(),
                     alt=2.0, lost=False,
                     sensors_mounted=["EO", "SAR", "ESM"],
                     munitions={"HF": 0, "RKT": 0, "GUN": 0}),
        PlatformInfo(pid="U3", type="UAV", pos=staging.copy(),
                     alt=2.0, lost=False,
                     sensors_mounted=["EO", "SAR", "ESM"],
                     munitions={"HF": 0, "RKT": 0, "GUN": 0}),
        PlatformInfo(pid="U4", type="UAV", pos=staging.copy(),
                     alt=2.0, lost=False,
                     sensors_mounted=["EO", "SAR", "ESM"],
                     munitions={"HF": 0, "RKT": 0, "GUN": 0}),
        PlatformInfo(pid="U5", type="UAV", pos=staging.copy(),
                     alt=2.0, lost=False,
                     sensors_mounted=["EO", "SAR", "ESM"],
                     munitions={"HF": 0, "RKT": 0, "GUN": 0}),
        PlatformInfo(pid="H1", type="HELI", pos=staging.copy(),
                     alt=3.0, lost=False,
                     sensors_mounted=["MMW", "EOIR"],
                     munitions={"HF": 16, "RKT": 76, "GUN": 1200}),
        PlatformInfo(pid="H2", type="HELI", pos=staging.copy(),
                     alt=3.0, lost=False,
                     sensors_mounted=["MMW", "EOIR"],
                     munitions={"HF": 16, "RKT": 76, "GUN": 1200}),
    ]

    sensor_params = [
        SensorParams(name="EO", P0=0.85, R=15.0, weather_sensitive=True),
        SensorParams(name="SAR", P0=0.90, R=50.0, weather_sensitive=False),
        SensorParams(name="ESM", P0=0.80, R=100.0, weather_sensitive=False),
    ]

    # LOS/occlusion: all 1s
    N_H = 2
    N_G = len(targets) if with_targets else 0
    los_matrix = np.ones((N_H, max(N_G, 1)))
    occlusion_matrix = np.ones((N_H, max(N_G, 1)))

    return SituationSnapshot(
        cycle_id=0,
        timestamp=0.0,
        grids=grids,
        targets=targets,
        platforms=platforms,
        sensor_params=sensor_params,
        commander_AOI=["A_3_4"],
        staging_position=staging,
        los_matrix=los_matrix if N_G > 0 else None,
        occlusion_matrix=occlusion_matrix if N_G > 0 else None,
    )


def test_e2e_with_targets():
    """端到端测试: 有确认存活目标时，应同时产出侦察与打击分配。"""
    settings = GlobalSettings(
        active_solver=SolverType.CBC,
        solver_time_limit_s=5.0,
        solver_mip_gap=1e-3,
    )
    snapshot = build_sample_snapshot(with_targets=True)
    allocator = MILPAllocator(settings)
    plan = allocator.solve(snapshot)

    assert isinstance(plan, AllocationPlan)
    assert plan.cycle_id == 0
    assert plan.status in ("OPTIMAL", "FEASIBLE", "TIME_LIMIT"), \
        f"Unexpected status: {plan.status}"
    assert plan.solve_time_ms > 0
    assert plan.solver_used == "CBC (python-mip)"

    print(f"\n=== 端到端测试（有目标）===")
    print(f"  状态: {plan.status}")
    print(f"  求解时间: {plan.solve_time_ms:.1f} ms")
    print(f"  目标值: {plan.objective:.2f}")
    print(f"  MIP Gap: {plan.mip_gap:.4f}")
    print(f"  侦察分配 ({len(plan.recon_assignments)} 条):")
    for ra in plan.recon_assignments:
        print(f"    {ra.pid} → {ra.sensor_used} → {ra.cell} ({ra.role})")
    print(f"  打击分配 ({len(plan.strike_assignments)} 条):")
    for sa in plan.strike_assignments:
        print(f"    {sa.pid} → {sa.target} ({sa.munition}×{sa.qty}, {sa.role})")

    # 验证侦察分配规则
    esm_assignments = [ra for ra in plan.recon_assignments if ra.sensor_used == "ESM"]
    eo_sar_assignments = [ra for ra in plan.recon_assignments if ra.sensor_used in ("EO", "SAR")]

    # 恰好 1 架 ESM UAV，覆盖全部 5 个栅格
    esm_uavs = set(ra.pid for ra in esm_assignments)
    assert len(esm_uavs) == 1, f"应有恰好 1 架 ESM UAV，实际: {esm_uavs}"
    esm_cells = set(ra.cell for ra in esm_assignments)
    assert esm_cells == {"c0", "c1", "c2", "c3", "c4"}, \
        f"ESM UAV 应覆盖全部 5 个栅格，实际覆盖: {esm_cells}"

    # 其余 UAV 使用 EO 或 SAR（新模型允许每架 UAV 同时挂 EO+SAR，各 1 格）
    eo_sar_uavs = set(ra.pid for ra in eo_sar_assignments)
    assert len(eo_sar_uavs) >= 4, \
        f"应有至少 4 架 EO/SAR UAV，实际: {eo_sar_uavs}"
    for pid in eo_sar_uavs:
        uav_assignments = [ra for ra in eo_sar_assignments if ra.pid == pid]
        assert len(uav_assignments) <= 2, \
            f"EO/SAR UAV {pid} 每架最多 2 个传感器分配（EO+SAR），实际: {len(uav_assignments)}"

    # 每个子区域 (c1-c4) 至少 1 架 EO/SAR
    eo_sar_cells = set(ra.cell for ra in eo_sar_assignments)
    assert eo_sar_cells == {"c1", "c2", "c3", "c4"}, \
        f"EO/SAR 应覆盖 c1-c4，实际覆盖: {eo_sar_cells}"

    _ = plan  # suppress return warning


def test_e2e_no_targets():
    """端到端测试: 无目标时，纯侦察模式不应崩溃，且无打击分配。"""
    settings = GlobalSettings(
        active_solver=SolverType.CBC,
        solver_time_limit_s=5.0,
    )
    snapshot = build_sample_snapshot(with_targets=False)
    allocator = MILPAllocator(settings)
    plan = allocator.solve(snapshot)

    assert isinstance(plan, AllocationPlan)
    assert plan.status in ("OPTIMAL", "FEASIBLE", "TIME_LIMIT"), \
        f"Unexpected status: {plan.status}"
    assert len(plan.strike_assignments) == 0, \
        "纯侦察模式不应有打击分配"

    print(f"\n=== 端到端测试（纯侦察）===")
    print(f"  状态: {plan.status}")
    print(f"  求解时间: {plan.solve_time_ms:.1f} ms")
    print(f"  侦察分配 ({len(plan.recon_assignments)} 条):")
    for ra in plan.recon_assignments:
        print(f"    {ra.pid} → {ra.sensor_used} → {ra.cell} ({ra.role})")

    # 验证侦察分配规则（纯侦察模式）
    esm_assignments = [ra for ra in plan.recon_assignments if ra.sensor_used == "ESM"]
    esm_uavs = set(ra.pid for ra in esm_assignments)
    assert len(esm_uavs) == 1, f"纯侦察: 应有恰好 1 架 ESM UAV，实际: {esm_uavs}"
    esm_cells = set(ra.cell for ra in esm_assignments)
    assert esm_cells == {"c0", "c1", "c2", "c3", "c4"}, \
        f"纯侦察: ESM UAV 应覆盖全部 5 个栅格，实际: {esm_cells}"

    eo_sar_assignments = [ra for ra in plan.recon_assignments if ra.sensor_used in ("EO", "SAR")]
    eo_sar_cells = set(ra.cell for ra in eo_sar_assignments)
    assert eo_sar_cells == {"c1", "c2", "c3", "c4"}, \
        f"纯侦察: EO/SAR 应覆盖 c1-c4，实际: {eo_sar_cells}"

    _ = plan  # suppress return warning


def test_snapshot_interface():
    """测试 SituationSnapshot 辅助方法。"""
    snapshot = build_sample_snapshot(with_targets=True)

    active = snapshot.get_active_targets()
    assert len(active) == 4, f"应有 4 个活跃目标，实际 {len(active)}"

    uavs = snapshot.get_uav_platforms()
    assert len(uavs) == 5, f"应有 5 架 UAV，实际 {len(uavs)}"

    helis = snapshot.get_heli_platforms()
    assert len(helis) == 2, f"应有 2 架 HELI，实际 {len(helis)}"

    uav_list = snapshot.get_platform_by_type("UAV")
    heli_list = snapshot.get_platform_by_type("HELI")
    assert len(uav_list) == 5
    assert len(heli_list) == 2

    print("\n=== SituationSnapshot 接口测试通过 ===")


def test_allocator_reuse():
    """测试 MILPAllocator 可重复调用，缓存热启动解。"""
    settings = GlobalSettings(
        active_solver=SolverType.CBC,
        solver_time_limit_s=5.0,
    )
    snapshot = build_sample_snapshot(with_targets=True)
    allocator = MILPAllocator(settings)

    plan1 = allocator.solve(snapshot)
    plan2 = allocator.solve(snapshot)

    assert plan1.status in ("OPTIMAL", "FEASIBLE")
    assert plan2.status in ("OPTIMAL", "FEASIBLE")

    print(f"\n=== 重复调用测试 ===")
    print(f"  Plan 1 目标值: {plan1.objective:.2f}, 时间: {plan1.solve_time_ms:.1f} ms")
    print(f"  Plan 2 目标值: {plan2.objective:.2f}, 时间: {plan2.solve_time_ms:.1f} ms")


# ── JSON 场景批量测试 ─────────────────────────────────

def test_json_scenario_default():
    """从 JSON 文件加载默认场景并求解。"""
    settings = GlobalSettings(
        active_solver=SolverType.CBC,
        solver_time_limit_s=5.0,
    )
    json_path = os.path.join(os.path.dirname(__file__), "..", "scenarios", "default.json")
    snapshot = load_snapshot_from_json(json_path)
    allocator = MILPAllocator(settings)
    plan = allocator.solve(snapshot)

    assert plan.status in ("OPTIMAL", "FEASIBLE", "TIME_LIMIT"), \
        f"JSON 场景 status={plan.status}"
    assert len(plan.recon_assignments) > 0, "应有侦察分配"

    print(f"\n=== JSON 默认场景 ===")
    print(f"  状态: {plan.status}  目标值: {plan.objective:.2f}  耗时: {plan.solve_time_ms:.1f} ms")
    print(f"  侦察: {len(plan.recon_assignments)} 条  打击: {len(plan.strike_assignments)} 条")
    for ra in plan.recon_assignments:
        print(f"    {ra.pid} → {ra.sensor_used} → {ra.cell} ({ra.role})")
    for sa in plan.strike_assignments:
        print(f"    {sa.pid} → {sa.target} ({sa.munition}×{sa.qty}, {sa.role})")


def test_all_json_scenarios():
    """批量运行 scenarios/ 目录下全部 JSON 场景。"""
    settings = GlobalSettings(
        active_solver=SolverType.CBC,
        solver_time_limit_s=5.0,
    )
    scenarios_dir = os.path.join(os.path.dirname(__file__), "..", "scenarios")
    snapshots = load_all_scenarios(scenarios_dir)

    assert len(snapshots) > 0, f"未找到 JSON 场景文件于 {scenarios_dir}"

    allocator = MILPAllocator(settings)

    print(f"\n=== 批量多场景测试 ({len(snapshots)} 个) ===")
    for i, snap in enumerate(snapshots):
        plan = allocator.solve(snap)
        aoi = snap.commander_AOI[0] if snap.commander_AOI else "?"
        n_targets = len(snap.get_active_targets())
        n_uav = len(snap.get_uav_platforms())
        n_heli = len(snap.get_heli_platforms())
        print(
            f"  [{i+1}] {aoi} | {n_uav}U+{n_heli}H | {n_targets}目标 | "
            f"{plan.status} | obj={plan.objective:.2f} | {plan.solve_time_ms:.0f}ms"
        )
        assert plan.status in ("OPTIMAL", "FEASIBLE", "TIME_LIMIT"), \
            f"场景 {i+1} ({aoi}) status={plan.status}"

    print(f"  全部 {len(snapshots)} 个场景通过!")


def test_json_output():
    """验证 AllocationPlan.to_json() 可正确序列化，下游可 json.load 回读。"""
    import tempfile

    settings = GlobalSettings(
        active_solver=SolverType.CBC,
        solver_time_limit_s=5.0,
    )
    snapshot = build_sample_snapshot(with_targets=True)
    allocator = MILPAllocator(settings)
    plan = allocator.solve(snapshot)

    # 写 JSON
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        fpath = f.name
    try:
        plan.to_json(fpath)

        # 回读
        import json
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["cycle_id"] == plan.cycle_id
        assert data["status"] == plan.status
        assert abs(data["objective"] - plan.objective) < 1e-6
        assert data["solve_time_ms"] == plan.solve_time_ms
        assert data["solver_used"] == plan.solver_used
        assert len(data["recon_assignments"]) == len(plan.recon_assignments)
        assert len(data["strike_assignments"]) == len(plan.strike_assignments)

        # 校验侦察条目
        for i, ra in enumerate(plan.recon_assignments):
            d = data["recon_assignments"][i]
            assert d["platform"] == ra.pid
            assert d["sensor_used"] == ra.sensor_used
            assert d["sensors_mounted"] == ra.sensors_mounted
            assert d["cell"] == ra.cell
            assert d["role"] == ra.role

        # 校验打击条目
        for i, sa in enumerate(plan.strike_assignments):
            d = data["strike_assignments"][i]
            assert d["platform"] == sa.pid
            assert d["target"] == sa.target
            assert d["munition"] == sa.munition
            assert d["qty"] == sa.qty
            assert d["role"] == sa.role

        print(f"\n=== JSON 输出测试 ===")
        print(f"  写入: {fpath}")
        print(f"  JSON 字段: {list(data.keys())}")
        print(f"  侦察条目: {len(data['recon_assignments'])} 条")
        print(f"  打击条目: {len(data['strike_assignments'])} 条")
        print(f"  回读校验通过!")
    finally:
        os.unlink(fpath)


if __name__ == "__main__":
    print("=" * 60)
    print("无人/有人机协同侦察-打击任务分配 —— 端到端测试")
    print("=" * 60)

    test_snapshot_interface()
    test_e2e_no_targets()
    test_e2e_with_targets()
    test_allocator_reuse()
    test_json_scenario_default()
    test_all_json_scenarios()
    test_json_output()

    print("\n" + "=" * 60)
    print("全部端到端测试通过!")
    print("=" * 60)
