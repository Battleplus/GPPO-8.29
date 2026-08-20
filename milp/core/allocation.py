"""
分配方案数据结构 —— 任务分配模块的标准化输出接口。

职责:
  1. 定义 ReconAssignment / StrikeAssignment 数据类
  2. 定义 AllocationPlan 顶层容器

对外接口:
  - ReconAssignment   — 单条侦察分配条目
  - StrikeAssignment  — 单条打击分配条目
  - AllocationPlan    — 完整分配方案

参考:
  设计方案 §3.2; 实施方案 §2.4
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ReconAssignment:
    """
    单条侦察分配。

    属性:
        pid:             平台标识 (如 "U1")
        sensors_mounted: 该平台搭载的全部传感器
        sensor_used:     本任务使用的传感器 (如 "ESM")
        cell:            侦察栅格 (如 "c1", "c0" 为巡逻区)
        role:            角色 (如 "area_scan")
    """
    pid: str                # 平台 ID
    sensors_mounted: List[str] = field(default_factory=list)
    sensor_used: str = ""   # 传感器名称
    cell: str = ""          # 栅格 ID
    role: str = ""          # 侦察角色

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.pid,
            "sensors_mounted": self.sensors_mounted,
            "sensor_used": self.sensor_used,
            "cell": self.cell,
            "role": self.role,
        }


@dataclass
class StrikeAssignment:
    """
    单条打击分配。

    属性:
        pid:      平台标识 (如 "H1")
        target:   目标标识 (如 "g1")
        munition: 弹药类型 (如 "HF")，若为空字符串则表示为支援角色（提供平台数）
        qty:      发射数量，若为 0 则表示为支援角色（无开火）
        role:     角色 (lead / wing / striker / *_support)

    NOTE: munition="" 且 qty=0 时表示该直升机被分配到此目标以满足
          req_plat 火力需求，但自身不开火（支援角色）。
    """
    pid: str        # 平台 ID
    target: str     # 目标 ID
    munition: str   # 弹药类型（"" = 支援无开火）
    qty: int        # 弹药数量（0 = 支援无开火）
    role: str       # 打击角色

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.pid,
            "target": self.target,
            "munition": self.munition,
            "qty": self.qty,
            "role": self.role,
        }


@dataclass
class AllocationPlan:
    """
    完整分配方案 —— 任务分配模块唯一输出。

    包含求解元数据与两类分配列表。

    属性:
        cycle_id:           当前分配轮次
        solve_time_ms:      求解耗时 (ms)
        solver_used:        求解器标识名
        recon_assignments:  侦察分配列表
        strike_assignments: 打击分配列表
        objective:          目标函数值
        mip_gap:            MIP 相对 gap
        status:             求解状态 (OPTIMAL / FEASIBLE / INFEASIBLE / TIME_LIMIT)

    参考:
        设计方案 §3.2
    """
    cycle_id: int
    solve_time_ms: float
    solver_used: str = ""
    recon_assignments: List[ReconAssignment] = field(default_factory=list)
    strike_assignments: List[StrikeAssignment] = field(default_factory=list)
    mounted_sensors: Dict[str, List[str]] = field(default_factory=dict)
    objective: float = 0.0
    mip_gap: float = 0.0
    status: str = "UNSOLVED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "status": self.status,
            "objective": self.objective,
            "mip_gap": self.mip_gap,
            "solve_time_ms": self.solve_time_ms,
            "solver_used": self.solver_used,
            "recon_assignments": [r.to_dict() for r in self.recon_assignments],
            "strike_assignments": [s.to_dict() for s in self.strike_assignments],
        }

    def to_json(self, filepath: str, indent: int = 2) -> None:
        import os

        # 若未指定目录，默认写入 results/
        if os.sep not in filepath and "/" not in filepath and "\\" not in filepath:
            os.makedirs("results", exist_ok=True)
            filepath = os.path.join("results", filepath)

        out_dir = os.path.dirname(filepath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=indent)
