# 随机事件 GPPO 复现环境

本文件记录本次初步实验实际使用的环境，而不是仅给出宽松的最低版本。旧 MLP-PPO checkpoint 由较新 NumPy 保存，但本机部分二进制依赖不兼容 NumPy 2，因此本次固定使用 NumPy 1.26.4，并由 `ppo_allocation/utils/sb3_compat.py` 在加载旧模型前注册只读模块别名。

## 已验证环境

- Windows，CPU 训练与推理
- Python 3.11.5（Anaconda build，64 bit）
- NumPy 1.26.4
- PyTorch 2.7.1+cpu
- Gymnasium 1.2.3
- Stable-Baselines3 / sb3-contrib 2.8.0
- Matplotlib 3.10.5，SciPy 1.15.3，Pandas 2.2.3

完整 Python 包版本见 `ppo_allocation/requirements-random-event-lock.txt`。原项目的 `requirements.txt` 只给下限，不能作为旧 checkpoint 的严格复现锁。

## 推荐安装

```powershell
cd E:\Z博士\random_event_gppo\54_20-master\ppo_allocation
py -3.11 -m venv ..\.venv
..\.venv\Scripts\python.exe -m pip install -r requirements-random-event-lock.txt
```

若 PyTorch CPU wheel 在当前索引中不可用，应按 PyTorch 官方 CPU 索引安装相同版本，再执行其余依赖。不要使用本机默认 Python 3.14 训练或加载当前 SB3 模型。

## 验证命令

```powershell
cd E:\Z博士\random_event_gppo\54_20-master\ppo_allocation
..\.venv\Scripts\python.exe -m unittest discover -s tests_random_event -v
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
..\.venv\Scripts\python.exe -m pytest tests\test_cpp_interface.py -q --rootdir=tests --confcutdir=tests --import-mode=prepend --disable-warnings
```

固定事件带与模型文件均在 `ppo_allocation/results/random_event/` 下保存 SHA-256；报告中的结论必须同时注明训练种子、训练步数、事件带 manifest 和 checkpoint 哈希。
