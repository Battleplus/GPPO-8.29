"""UAV PPO Task Allocation 包。

基于 PPO 强化学习的无人机搜索任务局部重分配系统。

核心模块：
    config:   全局配置常量
    env:      环境模型（无人机、区域、目标、事件、Gymnasium 环境）
    policy:   策略模块（动作掩码、动作修复、PPO Agent）
    utils:    工具函数（几何计算、奖励函数、指标追踪、日志、可视化）

入口脚本：
    train.py      — 训练模型
    evaluate.py   — 评估模型
    apply.py      — 应用模型并导出结果
"""
