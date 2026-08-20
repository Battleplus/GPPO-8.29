"""日志与文件输出工具模块。

提供结果目录创建和 JSON 文件保存等基础 I/O 功能。
"""

import json
from pathlib import Path
from datetime import datetime


def make_run_dir(base_dir: str = "results/logs") -> Path:
    """创建带时间戳的运行目录。

    每次训练/评估运行创建独立的时间戳目录，避免文件覆盖。

    Args:
        base_dir: 基础目录路径

    Returns:
        Path: 创建的运行目录（例如 results/logs/run_20260115_143052）
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path, data):
    """将字典保存为格式化的 JSON 文件。

    自动创建父目录，使用 UTF-8 编码和 2 空格缩进。

    Args:
        path: 输出文件路径（字符串或 Path 对象）
        data: 要保存的字典/列表数据
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
