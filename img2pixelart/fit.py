"""固定调色板加载与颜色匹配。"""

import cv2
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


# ---------------------------------------------------------------------------
# 调色盘加载
# ---------------------------------------------------------------------------


def load_palette(path: str) -> NDArray[np.uint8]:
    """从文本文件加载调色盘。

    支持格式：
    - AARRGGBB / RRGGBB 十六进制，每行一个
    - ``;`` 或 ``#`` 开头的行视为注释

    返回 ``(N, 3)`` BGR uint8 数组。
    """
    colors: list[tuple[int, int, int]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith((";", "#")):
                continue
            hex_str = line.lstrip("#")
            if len(hex_str) >= 8:
                hex_str = hex_str[-6:]
            if len(hex_str) != 6:
                continue
            r = int(hex_str[0:2], 16)
            g = int(hex_str[2:4], 16)
            b = int(hex_str[4:6], 16)
            colors.append((b, g, r))
    if not colors:
        raise ValueError(f"调色盘文件无有效颜色: {path}")
    return np.array(colors, dtype=np.uint8)


def fit_ramps_to_palette(
    ramps_bgr: FloatArray, palette_bgr: NDArray[np.uint8]
) -> FloatArray:
    """把原图自适应 ramp 的每一档匹配到 Lab 距离最近的调色板色。"""
    ramp_lab = cv2.cvtColor(
        ramps_bgr.astype(np.float32).reshape(-1, 1, 3) / 255.0,
        cv2.COLOR_BGR2LAB,
    ).reshape(-1, 3)
    palette_lab = cv2.cvtColor(
        palette_bgr.astype(np.float32)[None] / 255.0, cv2.COLOR_BGR2LAB
    )[0]
    distances = np.sum((ramp_lab[:, None] - palette_lab[None]) ** 2, axis=2)
    return (
        palette_bgr[np.argmin(distances, axis=1)]
        .reshape(ramps_bgr.shape)
        .astype(np.float32)
    )


def quantize_to_ramps(
    image_bgr: NDArray[np.uint8],
    family_labels: NDArray[np.int16],
    ramps_bgr: FloatArray,
    steps_per_family: NDArray[np.int32],
) -> NDArray[np.uint8]:
    """把每个前景像素收回所属 family 的精确 ramp 色。"""
    image_lab = cv2.cvtColor(image_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    result = image_bgr.copy()
    for family, steps in enumerate(steps_per_family):
        mask = family_labels == family
        if not mask.any():
            continue
        colors = ramps_bgr[family, :steps].astype(np.uint8)
        colors_lab = cv2.cvtColor(
            colors.astype(np.float32)[None] / 255.0, cv2.COLOR_BGR2LAB
        )[0]
        distances = np.sum(
            (image_lab[mask, None] - colors_lab[None]) ** 2,
            axis=2,
        )
        result[mask] = colors[np.argmin(distances, axis=1)]
    return result
