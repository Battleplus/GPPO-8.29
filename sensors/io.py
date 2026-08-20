from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from PIL import Image

    HAS_PIL = True
except Exception:
    Image = None
    HAS_PIL = False


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    arr = np.clip(rgb[:, :, :3], 0, 255).astype(np.uint8)
    if HAS_PIL:
        Image.fromarray(arr).save(path)
    else:
        np.save(path.with_suffix(".npy"), arr)

