"""PPO 智能体构建与预测模块。

封装 MaskablePPO 模型的创建、训练和推理接口。
使用 sb3-contrib 的 MaskablePPO 而非 stable-baselines3 的标准 PPO，
因为标准 PPO 不支持动作掩码（action mask）功能。
"""

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks


def build_model(env, tensorboard_log=None, verbose=1, device="cpu"):
    """构建 MaskablePPO 模型。

    使用 MLP 策略网络处理扁平化的一维观测向量。
    超参数选择基于经验调优，适用于本项目的离散动作空间规模。

    Args:
        env:             Gymnasium 环境（需支持 action_masks() 方法）
        tensorboard_log: TensorBoard 日志目录（可选）
        verbose:         日志详细级别（0=安静, 1=进度条, 2=每步日志）
        device:          训练设备 ("cpu" 或 "cuda")

    Returns:
        MaskablePPO: 可训练的 PPO 模型实例

    关键超参数说明：
        learning_rate=3e-4:  学习率，使用标准的 Adam 默认值
        n_steps=512:         每次策略更新前收集的步数（rollout buffer 大小）
        batch_size=128:      小批量大小
        n_epochs=10:         每次更新中重复使用数据的次数
        gamma=0.99:          折扣因子，接近 1 表示重视远期奖励
        gae_lambda=0.95:     GAE 的 λ 参数，平衡偏差与方差
        clip_range=0.2:      PPO 的裁剪范围
        ent_coef=0.01:       熵正则化系数，鼓励探索
        vf_coef=0.5:         价值函数损失在总损失中的权重
        max_grad_norm=0.5:   梯度裁剪阈值，防止梯度爆炸
    """
    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        device=device,
    )
    return model


def predict_with_mask(model, obs, env, deterministic=True):
    """使用动作掩码进行预测（推理）。

    从环境中获取当前合法动作掩码，传递到 model.predict() 中，
    确保输出动作合法。

    Args:
        model:         已加载的 MaskablePPO 模型
        obs:           当前观测向量
        env:           环境对象（用于获取 action_mask）
        deterministic: 是否使用确定性策略（True 取 argmax，False 采样）

    Returns:
        np.ndarray: shape=(4,) 的动作向量
    """
    action_masks = get_action_masks(env)
    action, _ = model.predict(obs, deterministic=deterministic, action_masks=action_masks)
    return action
