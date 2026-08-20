"""
执行输出接口 —— 任务分配模块 → 底层规控 的标准化数据契约。

================================================================================
使用方：底层规控（执行层）的开发者
作用：  接收任务分配模块产出的任务清单，按序执行

核心设计原则：
  1. 扁平任务列表 —— 执行单元只需顺序遍历，无需解析层级结构
  2. 先侦察后打击 —— 任务已按依赖关系排序
  3. 每个任务自包含 —— 含平台、目标/栅格、传感器/弹药、角色
  4. 可序列化 —— JSON 格式，支持跨进程/跨语言传输
================================================================================

=== 快速模板 ===

    from execution_output_interface import (
        ExecutionOrder, ReconTask, StrikeTask, TaskStatus,
        build_execution_order, to_json, from_json,
    )

    # ── 从分配方案构建执行清单 ──
    eo = build_execution_order(plan, snapshot)

    # ── 校验 ──
    errors = eo.validate()
    if errors:
        raise ValueError("\\n".join(errors))

    # ── 执行方遍历任务清单（伪代码）──
    for task in eo.tasks:
        if task.task_type == "recon":
            # 派发 UAV task.platform 用 task.sensor_used 侦察 task.cell
            result = execute_recon(task)
        elif task.task_type == "strike":
            # 派发 HELI task.platform 用 task.munition×task.qty 打击 task.target
            result = execute_strike(task)
        # 收集 result → 反馈给态势理解模块

    # ── 序列化输出 ──
    json_str = to_json(eo)
    with open("execution_order.json", "w") as f:
        f.write(json_str)

================================================================================
字段速查表
================================================================================

ExecutionOrder（顶层容器，一次 MILP 求解的完整输出）
├── cycle_id: int                        分配轮次
├── timestamp: float                     仿真时间戳
├── aoi_id: str                          当前 AOI 标识
├── solve_status: str                    求解状态，"OPTIMAL"/"FEASIBLE"/"TIME_LIMIT"
├── objective: float                     目标函数值
├── solve_time_ms: float                 求解耗时 (ms)
├── tasks: List[Task]                    有序任务列表（先侦察后打击）
│
│   ReconTask（task_type == "recon"）
│   ├── platform: str                    执行平台，如 "U3"
│   ├── task_type: str                   "recon"
│   ├── sensor: str                      使用传感器，"EO"/"SAR"/"ESM"
│   ├── cell: str                        目标栅格，"c0"~"c4"
│   ├── role: str                        角色（normal / esm_patrol）
│   └── aoi: str                         所属 AOI
│
│   StrikeTask（task_type == "strike"）
│   ├── platform: str                    执行平台，如 "H1"
│   ├── task_type: str                   "strike"
│   ├── target: str                      目标标识，如 "g1"
│   ├── munition: str                    弹药类型，"HF"/"RKT"/"GUN"
│   ├── qty: int                         发射数量（0 表示编队支援,不发射）
│   ├── role: str                        角色（lead / striker / wing / *_support）
│   └── aoi: str                         所属 AOI

TaskStatus（执行状态枚举，供规控反馈使用）
    PENDING → IN_PROGRESS → SUCCESS / FAILED / ABORTED
================================================================================
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Union
from enum import Enum


# ============================================================================
# 枚举
# ============================================================================

class TaskStatus(str, Enum):
    """任务执行状态枚举。

    状态转移:
        PENDING → IN_PROGRESS → SUCCESS
        PENDING → IN_PROGRESS → FAILED      (可重试)
        PENDING → ABORTED                   (上级取消)
    """
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


# ============================================================================
# 任务数据类
# ============================================================================

@dataclass
class ReconTask:
    """侦察任务 —— 指派一个平台用指定传感器侦察一个栅格。

    Attributes:
        platform:         执行平台 PID，如 "U3"
        sensors_mounted:  该平台搭载的全部传感器，如 ["EO", "SAR"]
        sensor_used:      本任务使用的传感器: "EO" / "SAR" / "ESM"
        cell:             目标栅格: "c0"(巡逻区) / "c1"~"c4"(子区)
        role:             角色: "normal"(标准侦察) / "esm_patrol"(ESM 广域巡逻)
        aoi:              所属 AOI 标识，如 "A_3_4"
    """
    platform: str
    sensors_mounted: List[str] = field(default_factory=list)
    sensor_used: str = ""
    cell: str = ""
    role: str = "normal"
    aoi: str = ""

    @property
    def task_type(self) -> str:
        return "recon"

    def validate(self) -> List[str]:
        errs = []
        if not self.platform:
            errs.append("ReconTask: platform 为空")
        if self.sensor_used not in ("EO", "SAR", "ESM", "MMW", "EOIR"):
            errs.append(f"ReconTask: sensor_used='{self.sensor_used}' 不在已知传感器列表")
        if not self.cell:
            errs.append("ReconTask: cell 为空")
        return errs


@dataclass
class StrikeTask:
    """打击任务 —— 指派一个平台用指定弹药打击一个目标。

    Attributes:
        platform:         执行平台 PID，如 "H1"
        sensors_mounted:  该平台搭载的全部传感器
        target:           目标标识，如 "g1"
        munition:         弹药类型: "HF"(地狱火) / "RKT"(火箭弹) / "GUN"(机炮)
        qty:              发射数量。0 表示编队支援角色（提供平台数但不发射）
        role:             角色:
                            "lead"          — 主攻（长机）
                            "striker"       — 主攻（独立打击）
                            "wing"          — 僚机
                            "lead_support"  — 长机支援（qty=0）
                            "striker_support"— 打击支援（qty=0）
        aoi:              所属 AOI 标识
    """
    platform: str
    sensors_mounted: List[str] = field(default_factory=list)
    target: str = ""
    munition: str = ""
    qty: int = 0
    role: str = "striker"
    aoi: str = ""

    @property
    def task_type(self) -> str:
        return "strike"

    @property
    def is_support(self) -> bool:
        """是否为纯支援角色（不实际发射弹药）。"""
        return self.qty == 0

    @property
    def is_effective_strike(self) -> bool:
        """是否为有效打击（实际发射弹药）。"""
        return self.qty > 0

    def validate(self) -> List[str]:
        errs = []
        if not self.platform:
            errs.append("StrikeTask: platform 为空")
        if not self.target:
            errs.append("StrikeTask: target 为空")
        # 有效打击 (qty>0) 必须指定弹药；支援任务 (qty=0) 可无弹药
        if self.qty > 0 and self.munition not in ("HF", "RKT", "GUN"):
            errs.append(f"StrikeTask: munition='{self.munition}' 不在已知弹药列表 {{HF, RKT, GUN}}")
        if self.qty < 0:
            errs.append(f"StrikeTask: qty={self.qty} 不能为负数")
        return errs


# 任务联合类型
Task = Union[ReconTask, StrikeTask]


# ============================================================================
# 执行清单
# ============================================================================

@dataclass
class ExecutionOrder:
    """执行清单 —— 一次 MILP 求解后对底层规控的完整输出。

    任务已按依赖关系排序：同一 AOI 内，侦察任务在前，打击任务在后。
    执行单元只需遍历 tasks 列表，调用 execute(task) 即可。

    每个任务自带 sensors_mounted（搭载清单）和 sensor_used（当前使用）。
    不再单独列出 platforms 摘要。

    Attributes:
        cycle_id:      分配轮次编号
        timestamp:     仿真时间戳
        aoi_id:        当前 AOI 标识
        solve_status:  MILP 求解状态
        objective:     目标函数值
        solve_time_ms: 求解耗时 (ms)
        tasks:         有序任务列表（List[ReconTask | StrikeTask]）
    """
    cycle_id: int
    timestamp: float
    aoi_id: str
    solve_status: str
    objective: float
    solve_time_ms: float
    tasks: List[Task] = field(default_factory=list)

    @property
    def recon_tasks(self) -> List[ReconTask]:
        """筛选侦察任务。"""
        return [t for t in self.tasks if t.task_type == "recon"]

    @property
    def strike_tasks(self) -> List[StrikeTask]:
        """筛选打击任务。"""
        return [t for t in self.tasks if t.task_type == "strike"]

    @property
    def effective_strikes(self) -> List[StrikeTask]:
        """筛选有效打击（qty > 0）。"""
        return [t for t in self.strike_tasks if t.is_effective_strike]

    @property
    def support_tasks(self) -> List[StrikeTask]:
        """筛选纯支援任务（qty == 0）。"""
        return [t for t in self.strike_tasks if t.is_support]

    def summary(self) -> str:
        """单行摘要字符串。"""
        n_r = len(self.recon_tasks)
        n_s = len(self.strike_tasks)
        n_eff = len(self.effective_strikes)
        return (
            f"[Cycle {self.cycle_id}] {self.aoi_id} | "
            f"{self.solve_status} | obj={self.objective:.2f} | "
            f"{self.solve_time_ms:.0f}ms | "
            f"recon:{n_r} strike:{n_s}(eff:{n_eff})"
        )

    def validate(self) -> List[str]:
        """校验整个执行清单。"""
        errs = []
        if self.cycle_id < 0:
            errs.append(f"cycle_id={self.cycle_id} 不能为负数")
        if not self.aoi_id:
            errs.append("aoi_id 为空")
        if self.solve_status not in ("OPTIMAL", "FEASIBLE", "TIME_LIMIT", "INFEASIBLE"):
            errs.append(f"solve_status='{self.solve_status}' 不在已知状态列表")
        for i, task in enumerate(self.tasks):
            task_errs = task.validate()
            for e in task_errs:
                errs.append(f"tasks[{i}] {e}")
        return errs


# ============================================================================
# 从分配方案构建执行清单（核心适配函数）
# ============================================================================

def build_execution_order(plan, snapshot) -> ExecutionOrder:
    """将 MILP 求解结果转为标准化执行清单。

    这是任务分配模块与底层规控之间的核心适配函数。
    其他模块开发者可直接调用，无需理解 MILP 内部结构。

    Args:
        plan:     TaskAllocator.solve() 返回的 AllocationPlan
        snapshot: 当前态势快照 SituationSnapshot

    Returns:
        ExecutionOrder，含扁平任务列表

    Example:
        >>> plan = allocator.solve(snap)
        >>> eo = build_execution_order(plan, snap)
        >>> for task in eo.tasks:
        ...     if task.task_type == "recon":
        ...         dispatch_uav(task.platform, task.sensor_used, task.cell)
        ...     else:
        ...         dispatch_heli(task.platform, task.target, task.munition, task.qty)
    """
    aoi = snapshot.commander_AOI[0] if snapshot.commander_AOI else ""

    tasks: List[Task] = []

    for ra in plan.recon_assignments:
        tasks.append(ReconTask(
            platform=ra.pid,
            sensors_mounted=ra.sensors_mounted,
            sensor_used=ra.sensor_used,
            cell=ra.cell,
            role=ra.role,
            aoi=aoi,
        ))

    for sa in plan.strike_assignments:
        tasks.append(StrikeTask(
            platform=sa.pid,
            sensors_mounted=plan.mounted_sensors.get(sa.pid, []),
            target=sa.target,
            munition=sa.munition if sa.munition else "",
            qty=sa.qty,
            role=sa.role,
            aoi=aoi,
        ))

    return ExecutionOrder(
        cycle_id=plan.cycle_id,
        timestamp=snapshot.timestamp,
        aoi_id=aoi,
        solve_status=plan.status,
        objective=plan.objective,
        solve_time_ms=plan.solve_time_ms,
        tasks=tasks,
    )


# ============================================================================
# JSON 序列化/反序列化
# ============================================================================

def _task_to_dict(task: Task) -> Dict[str, Any]:
    """将单个任务转为字典。"""
    d: Dict[str, Any] = {
        "platform": task.platform,
        "task_type": task.task_type,
        "role": task.role,
        "aoi": task.aoi,
        "sensors_mounted": task.sensors_mounted,
    }
    if task.task_type == "recon":
        d["sensor_used"] = task.sensor_used
        d["cell"] = task.cell
    elif task.task_type == "strike":
        d["target"] = task.target
        d["munition"] = task.munition
        d["qty"] = task.qty
    return d


def _dict_to_task(d: Dict[str, Any]) -> Task:
    """从字典恢复单个任务。"""
    task_type = d["task_type"]
    if task_type == "recon":
        return ReconTask(
            platform=d["platform"],
            sensors_mounted=d.get("sensors_mounted", d.get("sensors", [])),
            sensor_used=d.get("sensor_used", d.get("sensor", "")),
            cell=d["cell"],
            role=d.get("role", "normal"),
            aoi=d.get("aoi", ""),
        )
    elif task_type == "strike":
        return StrikeTask(
            platform=d["platform"],
            sensors_mounted=d.get("sensors_mounted", []),
            target=d["target"],
            munition=d.get("munition", ""),
            qty=d.get("qty", 0),
            role=d.get("role", "striker"),
            aoi=d.get("aoi", ""),
        )
    else:
        raise ValueError(f"未知 task_type: {task_type}")


def to_json(eo: ExecutionOrder, indent: int = 2) -> str:
    """将 ExecutionOrder 序列化为 JSON 字符串。

    输出格式与 task_interface.save_execution_order 完全兼容。

    Example:
        >>> json_str = to_json(eo)
        >>> with open("execution_order.json", "w", encoding="utf-8") as f:
        ...     f.write(json_str)
    """
    return json.dumps({
        "cycle_id": eo.cycle_id,
        "timestamp": eo.timestamp,
        "aoi_id": eo.aoi_id,
        "solve_status": eo.solve_status,
        "objective": eo.objective,
        "solve_time_ms": eo.solve_time_ms,
        "tasks": [_task_to_dict(t) for t in eo.tasks],
    }, ensure_ascii=False, indent=indent)


def from_json(data: Union[str, Dict]) -> ExecutionOrder:
    """从 JSON 字符串或字典反序列化为 ExecutionOrder。

    Example:
        >>> eo = from_json('{"cycle_id": 0, ...}')
        >>> eo = from_json({"cycle_id": 0, ...})
    """
    if isinstance(data, str):
        data = json.loads(data)

    tasks = [_dict_to_task(t) for t in data.get("tasks", [])]

    return ExecutionOrder(
        cycle_id=data.get("cycle_id", 0),
        timestamp=data.get("timestamp", 0.0),
        aoi_id=data.get("aoi_id", data.get("aoi", "")),
        solve_status=data.get("solve_status", "FEASIBLE"),
        objective=data.get("objective", 0.0),
        solve_time_ms=data.get("solve_time_ms", 0.0),
        tasks=tasks,
    )


# ============================================================================
# 使用示例（可独立运行验证）
# ============================================================================

def _demo():
    """演示输入→求解→输出的完整数据流。"""
    print("=" * 60)
    print("  执行输出接口 —— 端到端数据流演示")
    print("=" * 60)

    # Step 1: 构建态势输入
    from situation_input_interface import (
        build_situation_input, validate_situation_input, convert_to_snapshot,
    )

    si = build_situation_input(
        aoi_row=3, aoi_col=4,
        targets=[
            {"tid": "g1", "type": "RADAR", "pos": [162, 112],
             "confirmed": True, "value": 0.97, "threat": 0.91},
            {"tid": "g2", "type": "CP",    "pos": [188, 112],
             "confirmed": True, "value": 0.85, "threat": 0.60},
        ],
    )
    errors = validate_situation_input(si)
    assert not errors, f"输入校验失败: {errors}"

    # Step 2: 求解
    from task_interface import TaskAllocator
    snap = convert_to_snapshot(si)
    plan = TaskAllocator(solver="cbc", time_limit_s=5.0, verbose=0).solve(snap)
    print(f"\n求解状态: {plan.status}  目标值: {plan.objective:.2f}")

    # Step 3: 构建执行清单
    eo = build_execution_order(plan, snap)
    errors = eo.validate()
    assert not errors, f"输出校验失败: {errors}"

    print(f"\n{'-'*50}")
    print(f"执行清单: {eo.summary()}")
    print(f"{'-'*50}")

    # Step 4: 遍历输出
    for i, task in enumerate(eo.tasks):
        if task.task_type == "recon":
            print(f"  [{i}] {task.platform}: {task.sensor_used} → {task.cell} "
                  f"({task.role})")
        else:
            if task.is_effective_strike:
                print(f"  [{i}] {task.platform}: {task.munition}×{task.qty} → "
                      f"{task.target} ({task.role})  *** 发射 ***")
            else:
                print(f"  [{i}] {task.platform}: → {task.target} "
                      f"({task.role})  [支援,不发射]")

    # Step 5: JSON 序列化
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False, encoding="utf-8") as f:
        f.write(to_json(eo))
        tmp = f.name

    eo2 = from_json(json.loads(to_json(eo)))
    assert eo2.cycle_id == eo.cycle_id
    assert len(eo2.tasks) == len(eo.tasks)

    os.unlink(tmp)
    print(f"\n[OK] JSON 序列化/反序列化校验通过")
    print(f"[OK] 接口数据流: SituationInput → MILP → ExecutionOrder → JSON")


if __name__ == "__main__":
    _demo()
