"""评估脚本。

加载训练好的模型，在随机事件环境下运行多个 episode，
使用 MetricsTracker 统计各项评估指标，并保存汇总结果到 JSON 文件。

运行方式（修改 default_model 变量后）：
    python evaluate.py

输出：
    results/eval/evaluation_summary.json   — 平均指标汇总
"""

from pathlib import Path
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from env.uav_env import UAVTaskAllocationEnv
from utils.metrics import MetricsTracker
from utils.logger import save_json


def evaluate(model_path: str, episodes: int = 10):
    """加载模型并评估指定 episode 数。

    每个 episode 从头运行到截断（MAX_DECISION_STEPS 步），
    累积各步指标后计算 episode 平均值和跨 episode 均值。

    Args:
        model_path: 训练好的 MaskablePPO 模型 zip 文件路径
        episodes:   评估 episode 数量（默认 10）
    """
    # 创建评估环境
    env = UAVTaskAllocationEnv(random_event_mode=True)

    # 加载训练好的模型
    model = MaskablePPO.load(model_path, env=env, device="cpu")

    summaries = []
    for ep in range(episodes):
        obs, info = env.reset()
        tracker = MetricsTracker()
        terminated = False
        truncated = False

        # 运行一个完整 episode
        while not (terminated or truncated):
            # 获取当前合法动作掩码
            masks = get_action_masks(env)
            # 确定性推理（取最优动作而非采样）
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, reward, terminated, truncated, info = env.step(action)
            tracker.update(env, reward, info)

        summaries.append(tracker.summary())

    # 计算跨 episode 的平均值
    avg = {
        key: float(np.mean([s[key] for s in summaries]))
        for key in summaries[0]
    }

    # 保存评估结果到 JSON
    output_path = Path("results/eval/evaluation_summary.json")
    save_json(output_path, avg)
    print(f"Saved evaluation summary to: {output_path}")
    print(avg)


if __name__ == "__main__":
    # 默认模型路径，请根据实际训练结果修改
    default_model = "results/models/latest/maskable_ppo_uav_task_allocation.zip"
    evaluate(default_model)
