"""训练入口脚本。

使用 MaskablePPO 在 UAVTaskAllocationEnv 环境上进行训练。

训练流程：
  1. 创建带时间戳的运行目录（供日志和模型保存）
  2. 创建环境（训练模式，启用随机事件生成）
  3. 使用 Monitor 包装环境以记录 episode 级统计
  4. 构建 MaskablePPO 模型
  5. 分阶段训练（总 500,000 步，每 50,000 步保存一次模型、CSV数据、奖励曲线图）
  6. 保存最终模型到 results/models/<run_name>/

运行方式：
    python train.py

模型保存路径：
    results/models/<run_name>/maskable_ppo_uav_task_allocation.zip
"""

import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，避免在没有 GUI 的环境中报错
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor

from env.uav_env import UAVTaskAllocationEnv
from policy.ppo_agent import build_model
from utils.logger import make_run_dir


def save_csv_data(episode_rewards, episode_lengths, save_path, total_steps):
    """将训练数据保存为 CSV 文件。

    CSV 包含以下列：
      episode:      episode 编号（从 1 开始）
      reward:       本 episode 的累计奖励
      length:       本 episode 的步数
      cum_avg_reward: 到当前 episode 为止的累计平均奖励
      total_steps:  到当前 episode 为止的总训练步数

    Args:
        episode_rewards:  每个 episode 的总奖励列表
        episode_lengths:  每个 episode 的步数列表
        save_path:        CSV 文件保存路径
        total_steps:      总训练步数（用于文件命名信息）
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cum_sum = 0
    cum_steps = 0
    rows = []
    for i, (reward, length) in enumerate(zip(episode_rewards, episode_lengths), start=1):
        cum_sum += reward
        cum_steps += length
        rows.append({
            "episode": i,
            "reward": round(reward, 4),
            "length": length,
            "cum_avg_reward": round(cum_sum / i, 4),
            "total_steps": cum_steps,
        })

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "length", "cum_avg_reward", "total_steps"])
        writer.writeheader()
        writer.writerows(rows)


def plot_rewards(episode_rewards, save_path, title="Training Reward Curve"):
    """绘制奖励曲线并保存为图片。

    包含两张子图：
      1. 每 episode 的即时奖励散点图 + 滑动平均曲线
      2. 累计平均奖励趋势

    Args:
        episode_rewards: 每个 episode 的总奖励列表
        save_path:       图片保存路径
        title:           图表标题
    """
    if len(episode_rewards) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    episodes = np.arange(1, len(episode_rewards) + 1)
    rewards = np.array(episode_rewards)

    # 左图：每 episode 奖励 + 滑动平均（窗口=10）
    ax1.plot(episodes, rewards, alpha=0.4, color="steelblue", linewidth=0.8, label="Episode reward")
    window = min(10, len(rewards))
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax1.plot(episodes[window - 1:], smoothed, color="darkorange", linewidth=2, label=f"MA{window}")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title(f"{title} - Per Episode")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图：累计平均值趋势
    cum_avg = np.cumsum(rewards) / episodes
    ax2.plot(episodes, cum_avg, color="seagreen", linewidth=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Cumulative Average Reward")
    ax2.set_title(f"{title} - Cumulative Average")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    # 创建带时间戳的运行目录（如 results/logs/run_20260115_143052）
    run_dir = make_run_dir("results/logs")

    # 模型保存目录与日志目录同名
    model_dir = Path("results/models") / run_dir.name
    model_dir.mkdir(parents=True, exist_ok=True)

    # 创建训练环境（启用随机事件模式）
    env = UAVTaskAllocationEnv(random_event_mode=True)
    # Monitor 包装器记录 episode 累计奖励和 episode 长度
    env = Monitor(env)

    # 构建 MaskablePPO 模型（使用 CPU 训练），TensorBoard 日志写入运行目录
    model = build_model(env, tensorboard_log=str(run_dir), verbose=1, device="cpu")

    # 分阶段训练：总 500,000 步，每 50,000 步保存一次检查点、CSV和奖励曲线图
    total_timesteps = 5000    # 总训练步数
    save_interval = 50_000        # 保存间隔（步数）

    steps_done = 0
    checkpoint = 1
    while steps_done < total_timesteps:
        # 本阶段训练的步数（最后一段可能不足 save_interval）
        chunk = min(save_interval, total_timesteps - steps_done)

        # reset_num_timesteps=False 使学习率调度和内部计数器跨阶段连续
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)

        steps_done += chunk

        # 从 Monitor 中读取 episode 统计数据
        episode_rewards = list(env.get_episode_rewards())
        episode_lengths = list(env.get_episode_lengths())

        # 保存检查点模型
        checkpoint_path = model_dir / f"maskable_ppo_uav_task_allocation_{steps_done}_steps.zip"
        model.save(checkpoint_path)
        print(f"[{steps_done}/{total_timesteps}] Saved checkpoint to: {checkpoint_path}")

        # 保存 CSV 数据
        csv_path = model_dir / f"training_data_{steps_done}_steps.csv"
        save_csv_data(episode_rewards, episode_lengths, str(csv_path), steps_done)
        print(f"[{steps_done}/{total_timesteps}] Saved CSV to: {csv_path}")

        # 绘制并保存奖励曲线图
        reward_plot_path = model_dir / f"reward_curve_{steps_done}_steps.png"
        plot_rewards(episode_rewards, str(reward_plot_path),
                     title=f"Reward ({steps_done}/{total_timesteps} steps, {len(episode_rewards)} episodes)")
        print(f"[{steps_done}/{total_timesteps}] Saved reward plot to: {reward_plot_path}")

        checkpoint += 1

    # 保存最终模型
    model_path = model_dir / "maskable_ppo_uav_task_allocation.zip"
    model.save(model_path)

    # 保存最终 CSV 数据
    episode_rewards = list(env.get_episode_rewards())
    episode_lengths = list(env.get_episode_lengths())
    final_csv_path = model_dir / "training_data_final.csv"
    save_csv_data(episode_rewards, episode_lengths, str(final_csv_path), total_timesteps)

    # 保存最终奖励曲线图
    final_plot_path = model_dir / "reward_curve_final.png"
    plot_rewards(episode_rewards, str(final_plot_path),
                 title=f"Final Reward ({total_timesteps} steps, {len(episode_rewards)} episodes)")

    print(f"Training finished. Final model saved to: {model_path}")
    print(f"Final CSV saved to: {final_csv_path}")
    print(f"Final reward plot saved to: {final_plot_path}")


if __name__ == "__main__":
    main()
