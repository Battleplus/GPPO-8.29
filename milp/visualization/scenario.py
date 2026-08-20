"""
默认场景加载器 —— 复用测试样例保证「前端看到的」与「测试验证的」是同一个场景。

默认优先从 tests.test_end_to_end.build_sample_snapshot 导入，
若因任何原因导入失败则回退到本地等价副本 build_demo_snapshot。
"""

import os
import sys
import numpy as np

# 确保项目根在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_build_sample_snapshot = None
try:
    from tests.test_end_to_end import build_sample_snapshot as _build_sample_snapshot
except ImportError:
    pass


def _build_demo_snapshot(with_targets: bool = True):
    """本地等价副本 —— 仅在无法导入测试函数时使用。"""
    from core.snapshot import (
        SituationSnapshot, TargetInfo, PlatformInfo, SensorParams,
        generate_aoi_grids,
    )

    grids = generate_aoi_grids(aoi_row=3, aoi_col=4)
    grids[1].weather_w = 0.15
    grids[2].weather_w = 0.40
    grids[3].weather_w = 0.55
    grids[4].weather_w = 0.70

    staging = np.array([150.0, -50.0])

    targets = []
    if with_targets:
        targets = [
            TargetInfo(tid="g1", type="RADAR",
                       pos_est=np.array([270.0, 260.0]),  # A_6_6 东北角
                       pos_cov=np.eye(2) * 0.1, velocity_est=np.zeros(2),
                       confirmed=True, alive=True, value=1.0, threat=0.9),
            TargetInfo(tid="g2", type="CP",
                       pos_est=np.array([35.0, 230.0]),   # A_5_1 西北侧
                       pos_cov=np.eye(2) * 0.1, velocity_est=np.zeros(2),
                       confirmed=True, alive=True, value=0.85, threat=0.6),
            TargetInfo(tid="g3", type="AV",
                       pos_est=np.array([220.0, 45.0]),   # A_1_5 东南侧
                       pos_cov=np.eye(2) * 0.2,
                       velocity_est=np.array([0.02, 0.01]),
                       confirmed=True, alive=True, value=0.7, threat=0.5),
            TargetInfo(tid="g4", type="AV",
                       pos_est=np.array([60.0, 80.0]),    # A_2_2 西南侧
                       pos_cov=np.eye(2) * 0.2,
                       velocity_est=np.array([-0.01, 0.02]),
                       confirmed=True, alive=True, value=0.7, threat=0.5),
        ]

    platforms = []
    for i in range(1, 6):
        platforms.append(PlatformInfo(
            pid=f"U{i}", type="UAV", pos=staging.copy(), alt=2.0, lost=False,
            sensors_mounted=["EO", "SAR", "ESM"],
            munitions={"HF": 0, "RKT": 0, "GUN": 0},
        ))
    for i in range(1, 3):
        platforms.append(PlatformInfo(
            pid=f"H{i}", type="HELI", pos=staging.copy(), alt=3.0, lost=False,
            sensors_mounted=["MMW", "EOIR"],
            munitions={"HF": 16, "RKT": 76, "GUN": 1200},
        ))

    sensor_params = [
        SensorParams(name="EO", P0=0.85, R=15.0, weather_sensitive=True),
        SensorParams(name="SAR", P0=0.90, R=50.0, weather_sensitive=False),
        SensorParams(name="ESM", P0=0.80, R=100.0, weather_sensitive=False),
    ]

    N_H, N_G = 2, len(targets) if with_targets else 0
    los = np.ones((N_H, max(N_G, 1)))
    occ = np.ones((N_H, max(N_G, 1)))

    return SituationSnapshot(
        cycle_id=0, timestamp=0.0,
        grids=grids, targets=targets, platforms=platforms,
        sensor_params=sensor_params,
        commander_AOI=["A_3_4"], staging_position=staging,
        los_matrix=los if N_G > 0 else None,
        occlusion_matrix=occ if N_G > 0 else None,
    )


def load_default_snapshot(with_targets: bool = True):
    """
    默认演示场景：5 UAV + 2 HELI + AOI A_3_4 + staging=[150,-50] + g1~g4。

    优先复用 tests/test_end_to_end.py::build_sample_snapshot()，
    确保「前端看到的」与「测试验证的」是同一个场景。
    """
    if _build_sample_snapshot is not None:
        return _build_sample_snapshot(with_targets=with_targets)
    return _build_demo_snapshot(with_targets=with_targets)


def load_scenario_by_name(name: str):
    """
    按名称加载 scenarios/ 目录下的 JSON 场景文件。

    Parameters
    ----------
    name : str
        场景文件名（不含 .json 后缀），如 "default"、"pure_recon"

    Returns
    -------
    SituationSnapshot
    """
    from task_interface import load_snapshot_from_json

    scenarios_root = os.path.join(os.path.dirname(__file__), "..", "scenarios")
    fpath = os.path.join(scenarios_root, f"{name}.json")
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"场景文件不存在: {fpath}")
    return load_snapshot_from_json(fpath)


def list_scenario_names() -> list:
    """列出 scenarios/ 目录下所有可用的 JSON 场景名称（不含后缀）。"""
    scenarios_root = os.path.join(os.path.dirname(__file__), "..", "scenarios")
    if not os.path.isdir(scenarios_root):
        return []
    names = []
    for fname in sorted(os.listdir(scenarios_root)):
        if fname.endswith(".json"):
            names.append(fname[:-5])  # 去掉 .json
    return names
