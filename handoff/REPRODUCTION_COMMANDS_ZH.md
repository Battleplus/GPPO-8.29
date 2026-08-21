# 复现命令指南

## 1. 环境准备

### 1.1 安装依赖

```bash
# 进入项目目录
cd E:\Z博士\8.20\54_20-master

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r ppo_allocation/requirements.txt
pip install -r ppo_allocation/requirements-random-event-lock.txt
```

### 1.2 验证环境

```bash
# 运行核心测试
cd ppo_allocation
python -m unittest discover -s tests_random_event -v
```

## 2. 生成事件带

### 2.1 Smoke 测试

```bash
cd ppo_allocation
python -m random_event.experiment smoke \
    --tapes-per-mode 20 \
    --events-per-tape 3 \
    --master-seed 20260820
```

### 2.2 Validation Bank

```bash
cd ppo_allocation
python -m random_event.experiment protocol-bank \
    --tier preliminary \
    --split validation \
    --events-per-tape 5
```

### 2.3 Test Bank

```bash
cd ppo_allocation
python -m random_event.experiment protocol-bank \
    --tier preliminary \
    --split test \
    --events-per-tape 5
```

## 3. 训练模型

### 3.1 训练 GPPO-NoGate

```bash
cd ppo_allocation
python -m random_event.experiment train \
    --variants GPPO-NoGate \
    --seeds 1101,2202,3303 \
    --timesteps 300000 \
    --events-per-episode 5
```

### 3.2 训练 GPPO-Adaptive

```bash
cd ppo_allocation
python -m random_event.experiment train \
    --variants GPPO-Adaptive \
    --seeds 1101,2202,3303 \
    --timesteps 300000 \
    --events-per-episode 5
```

### 3.3 训练 Fair PPO-MLP

```bash
cd ppo_allocation
python -m random_event.experiment train \
    --variants PPO-MLP \
    --seeds 1101,2202,3303 \
    --timesteps 300000 \
    --events-per-episode 5
```

### 3.4 训练所有变体

```bash
cd ppo_allocation
python -m random_event.experiment train \
    --variants GPPO-NoGate,GPPO-Adaptive,PPO-MLP \
    --seeds 1101,2202,3303 \
    --timesteps 300000 \
    --events-per-episode 5
```

## 4. 评估模型

### 4.1 评估 Validation Bank

```bash
cd ppo_allocation
python -m random_event.experiment evaluate \
    --manifest results/random_event/tapes/preliminary_validation_protocol/manifest.json \
    --gppo-checkpoint results/random_event/models
```

### 4.2 评估 Test Bank

```bash
cd ppo_allocation
python -m random_event.experiment evaluate \
    --manifest results/random_event/tapes/preliminary_test_protocol/manifest.json \
    --gppo-checkpoint results/random_event/models
```

## 5. Colab Pro 运行

### 5.1 准备 Colab Bundle

```bash
# Colab bundle 已生成在 colab_bundle/ 目录
# 包含:
# - random_event_gppo_preliminary.ipynb
# - requirements.txt
# - README.md
```

### 5.2 在 Colab 上运行

1. 打开 Google Colab
2. 上传 `random_event_gppo_preliminary.ipynb`
3. 启用 GPU 运行时
4. 运行所有单元格

## 6. 生成报告

### 6.1 生成统计报告

```bash
cd ppo_allocation
python -c "
from random_event.metrics import aggregate_tapes, paired_metric_report
import json

# 加载评估结果
with open('results/random_event/test_eval/evaluation_summary.json') as f:
    results = json.load(f)

# 生成配对统计
for algorithm_pair, paired_results in results['paired_statistics'].items():
    print(f'{algorithm_pair}:')
    for metric, report in paired_results.items():
        if 'episode_return' in metric:
            print(f'  {metric}: mean_diff={report[\"mean_difference\"]:.3f}')
"
```

### 6.2 生成图表

```bash
cd ppo_allocation
python -c "
from random_event.plotting import plot_cumulative_vacancy, plot_recovery_delay
import matplotlib.pyplot as plt

# 生成图表 (需要实际数据)
# plot_cumulative_vacancy(results)
# plot_recovery_delay(results)
# plt.savefig('figures/preliminary_results.png')
"
```

## 7. 验证 P0 门禁

```bash
cd ppo_allocation
python -c "
import json
from pathlib import Path

# 检查 P0 门禁
p0_gate_path = Path('../handoff/P0_GATE.json')
if p0_gate_path.exists():
    with open(p0_gate_path) as f:
        p0_gate = json.load(f)
    print(f'P0 Gate Status: {p0_gate[\"training_allowed\"]}')
    print(f'Checks:')
    for check, status in p0_gate['checks'].items():
        print(f'  {check}: {status[\"status\"]}')
else:
    print('P0 gate file not found')
"
```

## 8. 验证冻结协议

```bash
cd ppo_allocation
python -c "
import json
from pathlib import Path

# 检查 seed manifest
with open('../configs/seed_manifest.json') as f:
    manifest = json.load(f)

# 验证 validation 不包含 unseen
validation_modes = list(manifest['preliminary']['validation']['modes'].keys())
print(f'Validation modes: {validation_modes}')
assert 'unseen' not in validation_modes, 'Validation should not contain unseen'

# 验证 test 不参与 checkpoint selection
test_checkpoint_selection = manifest['preliminary']['test']['checkpoint_selection']
print(f'Test checkpoint selection: {test_checkpoint_selection}')
assert not test_checkpoint_selection, 'Test should not participate in checkpoint selection'

print('Freeze protocol verification passed!')
"
```

## 9. 完整复现流程

```bash
# 1. 环境准备
cd E:\Z博士\8.20\54_20-master
python -m venv .venv
.venv\Scripts\activate
pip install -r ppo_allocation/requirements.txt

# 2. 运行测试
cd ppo_allocation
python -m unittest discover -s tests_random_event -v

# 3. 生成事件带
python -m random_event.experiment smoke --tapes-per-mode 20 --events-per-tape 3
python -m random_event.experiment protocol-bank --tier preliminary --split validation
python -m random_event.experiment protocol-bank --tier preliminary --split test

# 4. 训练模型
python -m random_event.experiment train \
    --variants GPPO-NoGate,GPPO-Adaptive,PPO-MLP \
    --seeds 1101,2202,3303 \
    --timesteps 300000

# 5. 评估模型
python -m random_event.experiment evaluate \
    --manifest results/random_event/tapes/preliminary_validation_protocol/manifest.json
python -m random_event.experiment evaluate \
    --manifest results/random_event/tapes/preliminary_test_protocol/manifest.json

# 6. 验证结果
python -c "
import json
with open('results/random_event/test_eval/evaluation_summary.json') as f:
    results = json.load(f)
print(f'Algorithms: {results[\"algorithms\"]}')
print(f'Tape count: {results[\"tape_count\"]}')
"
```

## 10. 注意事项

### 10.1 训练限制

- Preliminary 仅使用 3 个训练种子
- 结果不应作为最终结论
- Formal 需要 5 个训练种子

### 10.2 结果解释

- 成功率主要受可行性和 mask 支配
- GPPO 推理明显更慢
- overlap 是当前最明显的成功率压力场景

### 10.3 禁止事项

- ❌ 不声称 GPPO 显著优于 PPO
- ❌ 不声称 Adaptive Gate 带来提升
- ❌ 不修改 Test、删除失败
- ❌ 不自动进入 formal
