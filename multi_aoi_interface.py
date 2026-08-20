"""
多 AOI 场景任务分配模块 —— 对外顶层接口。

使用示例::

    from multi_aoi_interface import MultiAOITaskAllocator, load_multi_aoi_request_from_json

    # 1. 创建分配器（全局复用）
    allocator = MultiAOITaskAllocator(solver="cbc", verbose=1)

    # 2a. 从 JSON 文件加载
    input_data = load_multi_aoi_request_from_json("scenarios/multi_aoi_example.json")

    # 2b. 或直接构造字典
    input_data = {
        "aois": [...],
        "platforms": {...},
        "targets": [...],
        "aoi_route_state": None,
        "execution_feedback": None,
    }

    # 3. 首次调用
    result = allocator.run(input_data)

    # 4. 持久化 AOI 状态，下次带回
    input_data["aoi_route_state"] = result["aoi_route_state"]
    input_data["execution_feedback"] = {
        "aoi_id": result["aoi_route_state"]["current_aoi"],
        "aoi_status": "FINISHED",
    }

    # 5. 第二次调用
    result = allocator.run(input_data)

输入字段说明::

    aois                — 候选 AOI 列表（必填）
    platforms           — 平台配置（与单 AOI JSON 格式相同）
    targets             — 目标列表（与单 AOI JSON 格式相同）
    sensor_params       — 传感器参数（可选，有默认值）
    staging_position    — 出发点坐标 [x, y] km（可选，默认 [150, -50]）
    cycle_id            — 当前分配轮次（可选，默认 0）
    grid_weather        — 各子区天气覆盖（可选）
    aoi_route_state     — AOI 执行状态（首次调用传 null，后续带回）
    execution_feedback  — 底层规控反馈（可选）

输出字段说明::

    status              — "AOI_PLAN_READY" | "ALL_AOI_FINISHED"
    aoi_route_state     — 最新 AOI 执行状态（上层必须保存并带回）
    current_aoi_plan    — 当前 AOI 任务分配结果（ALL_AOI_FINISHED 时为 null）
      ├── aoi           — 当前 AOI ID
      ├── tasks         — 任务列表（与单 AOI make_execution_order 格式兼容）
      ├── solve_status  — MILP 求解状态
      ├── objective     — 目标函数值
      └── solve_time_ms — 求解耗时 (ms)
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from task_interface import TaskAllocator
from aoi import MultiAOIController

__all__ = [
    "MultiAOITaskAllocator",
    "load_multi_aoi_request_from_json",
    "save_multi_aoi_result_to_json",
]


class MultiAOITaskAllocator:
    """
    多 AOI 场景的任务分配器入口。

    单实例跨轮次复用；内部持有 TaskAllocator（复用 MILP 热启动）。

    Args:
        solver:       求解器名称，同 TaskAllocator，默认 "cbc"
        time_limit_s: 单次 MILP 求解时间上限（秒），默认 3.0
        mip_gap:      MIP gap 阈值，默认 1e-3
        verbose:      终端输出级别 0/1/2，默认 0
        grid_size_km: AOI 边长（km），默认 50.0
    """

    def __init__(
        self,
        solver: str = "cbc",
        time_limit_s: float = 3.0,
        mip_gap: float = 1e-3,
        verbose: int = 0,
        grid_size_km: float = 50.0,
    ):
        self._task_allocator = TaskAllocator(
            solver=solver,
            time_limit_s=time_limit_s,
            mip_gap=mip_gap,
            verbose=verbose,
        )
        self._controller = MultiAOIController(
            allocator=self._task_allocator,
            grid_size_km=grid_size_km,
        )

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行多 AOI 任务分配（一次调用处理一个 AOI）。

        Args:
            input_data: 输入字典，字段说明见模块文档

        Returns:
            输出字典，字段说明见模块文档
        """
        return self._controller.run(input_data)


# ── 工具函数 ─────────────────────────────────────────────

def load_multi_aoi_request_from_json(filepath: str) -> Dict[str, Any]:
    """
    从 JSON 文件加载多 AOI 任务请求。

    Args:
        filepath: JSON 文件路径

    Returns:
        可直接传入 MultiAOITaskAllocator.run() 的字典
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_multi_aoi_result_to_json(
    result: Dict[str, Any],
    filepath: str,
    indent: int = 2,
) -> None:
    """
    将 run() 的输出字典写入 JSON 文件。

    Args:
        result:   run() 返回的输出字典
        filepath: 目标文件路径
        indent:   JSON 缩进空格数
    """
    out_dir = os.path.dirname(filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=indent)
