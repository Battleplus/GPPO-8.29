# 原始 `ppo_allocation` 随机事件基线审计

## 1. 审计结论

原始模块是一个 **4 架 UAV、4 个搜索 Region、3 个 Target** 的局部重分配原型。它采用 165 维扁平状态、`MultiDiscrete([6,6,6,6])` 区域动作、严格动作掩码和 `MaskablePPO(MlpPolicy)`。目标发现后，规则层把发现者从 `SEARCH` 切换到 `TRACK`；PPO 只负责补齐搜索区域。

这份代码可以作为后续随机事件 GPPO 的 **业务状态、动作合法性、C++/JSON 接口和 MLP-PPO 对照组**，但不能直接作为“连续随机事件下已经验证有效”的基线。核心原因不是 PPO 训练轮数，而是当前训练环境通常在首个事件后一步结束：一般事件的 mask 会强制为每个受影响区域选择一架合法 UAV，随后 `all_valid=True` 立即终止。因此 500 个配对实例中，PPO、Nearest Legal 和 Masked Random 都达到 100% 成功率、100% 区域分配率和平均 1 步；PPO 与 Nearest Legal 的奖励也完全相同。现有结果主要证明 mask/规则能生成合法分配，不能证明策略学会了多事件恢复。

## 2. 审计范围与证据

本次只审计原始实现，没有修改 `ppo_allocation` 源码。主要证据如下：

| 主题 | 权威实现 |
|---|---|
| 场景规模、枚举、观测/动作维度、奖励权重 | `ppo_allocation/config.py` |
| reset、step、状态编码、事件生成、终止 | `ppo_allocation/env/uav_env.py` |
| 事件对象 | `ppo_allocation/env/event.py` |
| 动作 mask 与动作修复 | `ppo_allocation/policy/action_mask.py`、`action_repair.py` |
| 奖励 | `ppo_allocation/utils/reward.py` |
| PPO 模型与超参数 | `ppo_allocation/policy/ppo_agent.py`、`train.py` |
| 评估指标 | `ppo_allocation/evaluate.py`、`utils/metrics.py` |
| 外部事件接口 | `ppo_allocation/reallocation_service.py`、`cpp_bridge.py` |
| 单次/循环演示 | `ppo_allocation/apply.py`、`test_loop.py` |
| 已保存模型 | `ppo_allocation/results/models/run_20260605_210049/maskable_ppo_uav_task_allocation.zip` |

原始 ZIP 已由主任务另行保留；本审计基于其独立解压工作副本。已保存模型的 SHA-256 为 `5A9BE7153D33532CE99C61F13C8151549CC6CC919EC75FAD150D05BC78DEC5DA`。

## 3. 环境和任务背景

### 3.1 实体与业务边界

- UAV 数量为 4；Region 数量为 4；Target 数量为 3；地图为 `50×50`，均分为四个 `25×25` 区域。
- UAV 任务状态是 `SEARCH / TRACK / IDLE`。全部 UAV 初始化为 SAR，天气逻辑已注释；当前合法性主要检查存活、非 TRACK、传感器未故障。
- 初始分配固定为 `Ui → Ri`。目标位置和所属区域由 RNG 生成；未发现目标的位置/区域在观测中置零。
- Target 发现和跟踪属于规则逻辑：发现者进入 TRACK 并释放原搜索区域；PPO 不选择跟踪目标。

这与后续实验要求的任务背景一致，但原始环境没有 workload、能量、通信质量、事件观测延迟、事件队列和版本控制。

### 3.2 Gymnasium 生命周期

`reset(seed=...)` 初始化实体和一一分配后，立即调用 `_generate_next_event()`，也就是 episode 开始时已经发生一次事件。`step(action)` 的顺序是：

1. 记录旧分配；
2. 根据当前 mask 修复动作并执行；
3. 判断当前事件是否解决；
4. 判断 `all_valid` 和是否还有可搜索 UAV；
5. 计算奖励；
6. 仅在未结束且距离上次事件达到 10 个决策步时再生成事件。

最大决策步为 50，但这只是截断上限，并不代表每个 episode 真有 50 步。`random_event_mode` 被构造函数保存，却没有被 `reset()` 或 `_generate_next_event()` 用来分支；服务层之所以能用 `False`，是因为它手动加载状态并直接注入事件，避免走随机 reset 流程。

## 4. 状态、动作和 mask

### 4.1 165 维扁平状态

状态是单个 `Box(shape=(165,), low=-1, high=1)` 向量，而不是图：

| 部分 | 维度 | 内容摘要 |
|---|---:|---|
| 4 个 UAV | `4×17=68` | alive、位置、传感器、任务 one-hot、区域 multi-hot、目标 one-hot、负载比例 |
| 4 个 Region | `4×10=40` | 中心、天气、是否合法分配、分配者 one-hot |
| 3 个 Target | `3×15=45` | 类型、发现/跟踪/摧毁、已知位置、区域、跟踪者 |
| 当前 Event | 12 | 类型 one-hot、影响区域 multi-hot、相关 UAV one-hot |
| 合计 | **165** | 送入 MLP 的固定长度向量 |

向量中用 one-hot/multi-hot 隐含表达了 UAV–Region、UAV–Target 关系，但没有显式节点、边、邻接、消息传递或图级池化。

### 4.2 动作

动作空间为 `MultiDiscrete([6,6,6,6])`。每个 Region 独立输出：`KEEP / U0 / U1 / U2 / U3 / NO_UAV`。模型实际输出 24 个 logits（四个六分类分布），不是“选择一条 UAV–Region 边”的单一离散动作，也不是对完整 `6^4` 联合动作逐一评分。

### 4.3 一般事件 mask 的真实行为

mask 是 `(4,6)` 后展平为 24 位布尔数组：

- 非受影响 Region：只允许 `KEEP`；
- 受影响 Region 且存在合法 UAV：**只允许合法的 U0–U3，不允许 KEEP，也不允许 NO_UAV**；
- 受影响 Region 且没有合法 UAV：只允许 `NO_UAV`；
- `TARGET_DESTROYED`：所有 Region 允许 KEEP；满足“空缺或原负责人负载大于 1”的 Region 还允许被释放 UAV 接管。

这里存在注释与代码不一致：`action_mask.py` 文档写“受影响区域允许合法 UAV + KEEP”，实际一般事件分支没有打开 KEEP。正是这一点使首个 damage/discovery 事件几乎必然在一步内被补齐。

动作执行前还有 repair 层。它把越过 per-region mask 的动作回退为 KEEP 或首个合法动作，并在 `TARGET_DESTROYED` 中限制被释放 UAV 最多接管一个 Region。正常使用 `MaskablePPO` 时 repair 率应接近 0；它是安全兜底，不是学习能力证据。

## 5. 原始事件生成器审计

### 5.1 代码声明与实际可采样事件

对外服务支持四种业务事件：`UAV_DAMAGE / TARGET_DISCOVERED / TARGET_DESTROYED / REGION_VACANCY`。但训练环境 `_generate_next_event()` 的候选列表实际只有：

1. UAV 损毁；
2. 目标发现导致搜索区域空缺；
3. 已跟踪目标摧毁。

`_event_region_vacancy()` 虽然存在，却没有放入候选列表。`TARGET_DISCOVERED` 没有独立的 `EventType`，被编码为 `REGION_VACANCY`，所以模型无法从事件类型 one-hot 区分“自然区域空缺”和“发现目标导致的空缺”，只能从实体状态间接推断。配置中的 `EVENT_PROBS` 也没有被生成器读取；实际是对三个函数等概率抽取。

### 5.2 条件事件和拒绝采样

生成器最多重试 30 次：随机抽函数，条件不成立则返回 `None` 再抽。若仍失败，回退为空的 `TARGET_DESTROYED` 占位事件。这是拒绝采样，事件实际比例会随当前状态和可行条件改变，并不等于配置概率。

初始状态没有已跟踪目标，因此 `TARGET_DESTROYED` 在 reset 时不可行。可行的首事件通常只有 damage 和 discovery；而直接 `REGION_VACANCY` 永远不会由训练候选抽到。事件对象也只有类型、影响 Region、释放/损毁 UAV 和描述，没有 `event_id`、发生/观测时间、源事件、target、severity、payload、seed 或 state version。

### 5.3 事件时序能力

环境只有一个 `current_event` 指针，没有事件队列或 pending 集合。新事件只可能在当前 episode 未结束并累计 10 个 decision steps 后覆盖 `current_event`。没有 overlap/burst 语义，也没有“处理旧事件过程中又来新事件”的排队、合并、抢占或 stale decision 拒绝流程。

外部真实事件也不是由本模块探测：`ReallocationService.handle_event()` 假设上游已经检测并传入一个事件 dict。代码没有遥测心跳、目标检测器、周期检查或异步消息订阅，因此“事件如何在真实系统中被探查”属于上游感知/通信模块，不应归功于当前 PPO 环境。

## 6. 终止条件与一决策退化

终止表达式是：

```text
terminated = all_regions_valid OR no_uav_available
truncated  = NOT terminated AND decision_step >= 50
```

`all_regions_valid` 只检查四个 Region 是否都有合法负责人且 `need_reassign=False`，没有检查当前事件是否成功、是否还有 pending event，或预定事件是否尚未到达。因此：

- 对 UAV damage / target discovery，一般 mask 强制为受影响 Region 选合法 UAV；一次联合动作后通常立即 `all_valid=True`，episode 结束；
- 第 10 步的后续随机事件通常没有机会出现；
- 对 `TARGET_DESTROYED`，Region 原本就可能全部合法，即使策略全部 KEEP、释放 UAV 未重新投入搜索，`event_success=False` 仍可能因 `all_valid=True` 而结束；
- 最大 50 步和 `EVENT_INTERVAL=10` 不能证明环境支持连续事件，因为常见轨迹根本到不了第二个事件。

因此训练样本主要是大量相互独立的单事件、单决策上下文，PPO 的 `gamma`、GAE 和 Critic 的长期价值能力几乎没有被任务时序真正检验。

## 7. 奖励函数

原始奖励是决策后状态的加权总和，并非 `J(before)-J(after)`：

| 项 | 权重/实现 |
|---|---|
| 合法分配 | 每个合法 Region `+5` |
| 未分配或非法 | 每个 `-4` |
| 变更距离 | 仅对负责人发生变化的 Region，平均 `distance/AREA_SIZE` 后乘 `-10` |
| 负载差 | `-(max load-min load)` |
| 空闲可用 UAV | 每架 `-2` |
| 当前事件成功 | `+10` |
| 仍空缺的受影响 Region | 每个额外 `-4` |
| 切换、非法修复、终止 bonus、传感器匹配 | 当前权重均为 `0` |

这会形成两类解释限制：第一，绝对覆盖奖励每一步重复发放，长 episode 的累计奖励与一步恢复不能直接比较；第二，没有 workload、优先级覆盖、恢复时延、通信量、门控通信、旧任务保持率或突发重切换成本。距离归一化用的是 `AREA_SIZE=50`，不是配置中已定义的地图对角线 `MAX_DISTANCE≈70.71`。

## 8. PPO 模型和训练产物

### 8.1 网络结构

`build_model()` 使用 `sb3_contrib.MaskablePPO(policy="MlpPolicy")`。已保存模型 `policy.pth` 的真实参数形状为：

- Actor MLP：`165 → 64 → 64 → 24 logits`；
- Critic MLP：`165 → 64 → 64 → 1`；
- 总参数量：**31,193**。

Actor 与 Critic 都直接读取扁平向量。没有 AHGNN、Adaptive Gate、图 attention、边打分或图级 Critic；这些都必须作为新 GPPO 实现，而不能在报告中称为原代码已有。

### 8.2 超参数与脚本漂移

保存模型元数据和代码一致：learning rate `3e-4`、`n_steps=512`、batch `128`、每次更新 `10` epoch、`γ=0.99`、GAE `λ=0.95`、clip `0.2`、entropy `0.01`、value coefficient `0.5`、gradient norm `0.5`。模型记录的 `num_timesteps=5120`；这是脚本请求 5000 步而 rollout 长度为 512 导致的向上取整。

`train.py` 顶部注释仍写“总 500,000 步、每 50,000 步保存”，实际变量是 `total_timesteps=5000`、`save_interval=50000`，最终只形成一个训练阶段。这种注释/配置漂移必须在新协议中消除。

模型归档的原训练环境是 Python 3.10.20、SB3 2.8.0、PyTorch 2.6.0+cu124、NumPy 2.2.6、Gymnasium 1.2.3。当前 `requirements.txt` 只有宽松下界，没有锁文件；在 NumPy 1.24 环境加载该模型会报 `ModuleNotFoundError: numpy._core`。因此旧模型复现应优先匹配归档元数据，不能只执行当前宽松 requirements 后假设兼容。

## 9. 原始评估指标

`MetricsTracker` 只统计：

- 平均 Region 分配率；
- 每决策步平均 reward；
- 平均 load gap；
- 非法修复总数；
- NO_UAV 动作总数；
- `event_success=True` 的决策步比例。

最后一项名称是 `ppo_local_reallocation_success_rate`，但没有 event_id 和事件分母，实际上是“成功标志的 step 比例”，不是严格的逐事件恢复成功率。原生 `evaluate.py` 默认仅 10 个 episode，没有固定 test bank、按事件类型分层、多训练 seed、置信区间、显著性检验、推理时间、恢复时延、通信开销、旧任务保持率、累计缺口 AUC、超时率或 oracle gap。

## 10. 已有 500 实例结果及其正确解释

工作区中存在一次此前生成的配对审计结果（不属于原仓库正式评估入口）：

- 脚本：`E:\Z博士\tmp\ppo_review\batch_evaluate.py`，SHA-256 `69E5DED98F0315818376CA645CA03F324B2C3CB1026078834809518CB6C98F93`；
- 汇总：`E:\Z博士\tmp\ppo_review\ppo_allocation\results\batch_evaluation\summary.json`，SHA-256 `305060A197DEE0B5AF85D73A6A009A3E7BD6E5BFB1F6C96B3F4808852D696A3`；
- 明细：同目录 `episodes.csv`，SHA-256 `5752B5D2C4F4DD6981C66174160E58B68A1EF7F6FD734EEB5B3D2712F606DFA5`。

该脚本对三种策略分别使用相同的 500 个 episode seed（base seed `20260813`），但 masked random 自己另有固定 RNG。结果如下：

| 策略 | 平均 reward | reward SD | 成功率 | 分配率 | load gap | 平均步数 | repair rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO | 24.0000 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| Nearest Legal | 24.0000 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| Masked Random | 23.3165 | 0.9738 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |

事件分布只有 `UAV_DAMAGE=253`、`TARGET_DISCOVERED=247`，没有直接 `REGION_VACANCY`，也没有 `TARGET_DESTROYED`。PPO 与 Nearest Legal 完全相同，Masked Random 也总能一步成功，只因部分分配距离较远而 reward 略低。

因此这个结果不支持“PPO 学会复杂连续恢复”，更不支持“GPPO 一定优于 PPO”。它强烈支持以下诊断：**当前 mask 和一步终止条件已经把可行性问题基本解完，学习算法只剩在少量合法 UAV 之间优化距离的狭窄空间。** 后续实验必须先扩展为可重放的多事件 episode，并让成功率、恢复时延、任务保持和通信代价真正可区分。

## 11. 现有可复现入口与边界

以下命令应从 `ppo_allocation` 目录执行，因为源码采用 `from config import ...` 这类顶层导入：

```powershell
# 训练原始 MLP-MaskablePPO
python train.py

# 修改 evaluate.py 中模型路径后，运行默认 10 episode 评估
python evaluate.py

# 单场景 + 单事件推理、JSON 和 before/after 图
python apply.py --scenario scenarios/example_target_discovered.json

# 连续手工注入 5 轮事件的可视化演示
python test_loop.py
```

服务/C++ 接口从仓库根目录使用：

```powershell
python ppo_allocation/cpp_bridge.py --request-file <request.json>
python -m pytest ppo_allocation/tests/test_cpp_interface.py
```

边界和已知复现风险：

- `test_loop.py` 使用无 seed 的 RNG，手工注入事件后直接 `_execute_action()`，绕过 `env.step()` 的 reward、终止和指标；它是演示，不是有效的连续事件评估。
- 根目录名 `54_20-master` 不是合法 Python 包名，仓库根 `__init__.py` 又使用相对导入；当前 pytest 收集可能先在根 `__init__.py` 报 `attempted relative import with no known parent package`。
- 旧模型依赖 NumPy 2.x 序列化模块路径；仅满足 `numpy>=1.24` 不保证可加载。
- `evaluate.py` 的默认模型路径是 `results/models/latest/...`，而仓库中的真实模型位于带时间戳目录，需要手工修改。

这些问题必须通过新实验的锁定环境、统一模块入口和自动化测试解决；不能把手工成功运行一次当作可复现性证明。

## 12. 后续 GPPO 实验必须保留和必须替换的部分

建议保留：

- 4 UAV / 4 Region / 3 Target 业务背景；
- `SEARCH / TRACK / IDLE` 状态机和目标发现后规则转 TRACK；
- 四种对外事件名称与 JSON/C++ 接口兼容；
- 合法性检查、旧任务尽量不动的局部重分配原则；
- 原始 165 维 MLP-MaskablePPO 作为对照组。

必须替换或扩展：

- 单一 `current_event` → 可重放 event tape、队列、pending set 和 graph/state version；
- 拒绝采样 → 对当前有效事件类型做条件概率重归一化；
- 一步 `all_valid` 终止 → 按事件数/物理时间 horizon，并区分暂时与最终不可行；
- 扁平向量 MLP → UAV/Region/Target 异构图、AHGNN、可选 Adaptive Gate；
- 四维区域动作 → 合法 UAV–Region 边加 NOOP 的统一候选动作；
- 绝对状态奖励 → 可追踪组件的 `J(before)-J(after)`；
- step 成功率 → 逐事件恢复、延迟、通信、任务保持、超时、AUC、oracle gap 等配对指标；
- 单次随机演示 → 固定 train/validation/test event banks、多训练 seed 和统计检验。

最终判断：原始代码足以定义“任务是什么”和提供 MLP-PPO 兼容基线，但它目前测到的主要是 **mask 约束下的一步合法补位**。随机事件 GPPO 的第一项有效性门槛，应当是让四类事件都能在同一 episode 中按可重放时序出现，并在旧事件尚未恢复时容许新事件到达；只有在此之后，比较 MLP、图网络和 Adaptive Gate 才具有可解释性。
