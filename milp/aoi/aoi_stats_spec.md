  # AOI 统计指标计算规范 —— 供态势理解模块参考

## 接口定义

```python
def compute_aoi_stats(
    aoi: dict,        # {"id": "A_3_4", "row": 3, "col": 4, ...}
    targets: list,    # [{"tid": "g1", "pos": [x,y], "confirmed": bool, "value": float, "threat": float}, ...]
) -> tuple[float, float, float]:
    """
    返回 (target_prior, target_value, target_threat)
    """
```

## 三个指标的计算方法

### 1. `target_prior` — 目标存在先验概率

**含义**：该 AOI 内存在"有价值目标"的估计概率，∈ [0, 1]。

**计算逻辑**：

```
已确认目标数 = count( targets in AOI, confirmed=True  )
未确认线索数 = count( external_intel_hints in AOI )      ← 态势理解自身的情报

target_prior = clamp( (已确认目标数 × 0.3 + 未确认线索数 × 0.15) , 0.1, 0.95 )
```

**说明**：
- 每个已确认目标贡献 0.3（确定性信号，权重大）
- 每条未确认情报线索贡献 0.15（弱信号）
- 下界 0.1：即使没有任何信号，也有 10% 的基础概率（战场总可能存在未知目标）
- 上界 0.95：避免绝对确定

**特例**：如果态势理解没有额外情报来源，可简化为基础值 + 已确认目标贡献：

```python
target_prior = min(0.1 + confirmed_count * 0.3, 0.95)
```

---

### 2. `target_value` — 目标平均作战价值

**含义**：该 AOI 内目标的预期作战价值，∈ [0, 1]。

**计算逻辑**：

```
已确认目标的价值列表 = [t.value for t in targets_in_aoi if t.confirmed]

如果有已确认目标:
    target_value = mean(已确认目标的价值列表)
否则:
    target_value = 0.5   ← 默认中等价值（无信息时的中性估计）
```

**说明**：
- 价值来自态势融合对目标类型（RADAR/CP/AV）的评估
- 无目标时用中性默认值，不影响排序（会被 target_prior 的低值压制）

---

### 3. `target_threat` — 目标平均威胁度

**含义**：该 AOI 内目标的预期威胁等级，∈ [0, 1]。

**计算逻辑**：与 target_value 完全对称：

```
已确认目标的威胁列表 = [t.threat for t in targets_in_aoi if t.confirmed]

如果有已确认目标:
    target_threat = mean(已确认目标的威胁列表)
否则:
    target_threat = 0.5
```

---

## 配合 AOI 排序使用

AOI 综合价值公式（在 `aoi_router.py` 中）：

```
V(a) = 0.40 × priority      ← 指挥员优先级（人工指定）
     + 0.20 × target_value  ← 本函数计算
     + 0.20 × target_threat ← 本函数计算
     + 0.20 × target_prior  ← 本函数计算
```

**权重设计意图**：
- `priority` 占 40%：指挥员意图始终是最强信号
- 其余三项各占 20%：客观情报的辅助判断

---

## 计算示例

假设场景：

```python
aoi = {"id": "A_5_6", "row": 5, "col": 6, "priority": 0.8}

targets = [
    {"tid": "g1", "pos": [252, 238], "confirmed": True,  "value": 0.97, "threat": 0.91},
    {"tid": "g4", "pos": [275, 220], "confirmed": True,  "value": 0.60, "threat": 0.30},
    # g2, g3 在其他 AOI 或未确认，不参与计算
]
```

计算过程：

```
A_5_6 内已确认目标: g1, g4 (2 个)

target_prior  = min(0.1 + 2 × 0.3, 0.95) = 0.70
target_value  = (0.97 + 0.60) / 2         = 0.785
target_threat = (0.91 + 0.30) / 2         = 0.605

V(A_5_6) = 0.40 × 0.8 + 0.20 × 0.785 + 0.20 × 0.605 + 0.20 × 0.70
         = 0.32 + 0.157 + 0.121 + 0.14
         = 0.738
```

---

## 边界情况

| 情况 | target_prior | target_value | target_threat |
|------|-------------|--------------|---------------|
| AOI 内无任何目标、无情报 | 0.10 | 0.50 | 0.50 |
| AOI 内有 1 个已确认目标 | 0.40 | 该目标 value | 该目标 threat |
| AOI 内有 3+ 个已确认目标 | 0.95 (封顶) | 均值 | 均值 |
| AOI 内只有未确认线索 | 线索数 × 0.15 + 0.1 | 0.50 | 0.50 |
