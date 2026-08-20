from __future__ import annotations

import numpy as np
from .weather_effects import apply_sar_weather

try:
    from PIL import Image, ImageFilter

    HAS_PIL = True
except Exception:
    Image = None
    ImageFilter = None
    HAS_PIL = False


def process(
    rgb: np.ndarray,
    speckle_shape: float = 3.0,
    edge_gain: float = 0.55,
    image_gain: float = 0.82,
    weather: dict | None = None,
    **_kwargs,
) -> np.ndarray:
    arr = rgb[:, :, :3].astype(np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    speckle_shape = max(0.1, float(speckle_shape))
    speckle = np.random.gamma(shape=speckle_shape, scale=1.0 / speckle_shape, size=gray.shape).astype(np.float32)
    gray = np.clip(gray * speckle, 0, 255).astype(np.uint8)
    if HAS_PIL:
        edges = np.array(Image.fromarray(gray, mode="L").filter(ImageFilter.FIND_EDGES)).astype(np.float32)
        gray = np.clip(gray.astype(np.float32) * float(image_gain) + edges * float(edge_gain), 0, 255).astype(np.uint8)
    out = np.dstack([gray, gray, gray])
    return apply_sar_weather(out, weather=weather)
