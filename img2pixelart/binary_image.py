"""二值图像原语：掩码降采样与骨架细化。

从 structure 阶段抽出的通用操作，像素画与 ASCII 字符画共用：
- :func:`down_mask`：按覆盖率把掩码缩到目标网格；
- :func:`thin`：1px 骨架化（cv2.ximgproc 优先，Zhang-Suen 后备）。
"""

import cv2
import numpy as np
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]


# ---------------------------------------------------------------------------
# 掩码降采样
# ---------------------------------------------------------------------------


def down_mask(mask: np.ndarray, width: int, height: int, threshold: float) -> BoolArray:
    """掩码降采样：目标网格内原掩码的覆盖率 >= threshold 才置 1。"""
    average = cv2.resize(
        mask.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA
    )
    if mask.max(initial=0) > 1:
        average = average / 255.0
    return average >= threshold


# ---------------------------------------------------------------------------
# 骨架细化
# ---------------------------------------------------------------------------


def _zhang_suen_thinning(mask: BoolArray) -> BoolArray:
    """Zhang-Suen 细化的纯 NumPy 实现（cv2 缺少 ximgproc 时的后备方案）。"""
    image = mask.astype(np.uint8).copy()
    changed = True
    while changed:
        changed = False
        for phase in (0, 1):
            padded = np.pad(image, 1)
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1))
                + ((p4 == 0) & (p5 == 1))
                + ((p5 == 0) & (p6 == 1))
                + ((p6 == 0) & (p7 == 1))
                + ((p7 == 0) & (p8 == 1))
                + ((p8 == 0) & (p9 == 1))
                + ((p9 == 0) & (p2 == 1))
            )
            if phase == 0:
                condition3 = (p2 * p4 * p6) == 0
                condition4 = (p4 * p6 * p8) == 0
            else:
                condition3 = (p2 * p4 * p8) == 0
                condition4 = (p2 * p6 * p8) == 0
            delete = (
                (image == 1)
                & (neighbors >= 2)
                & (neighbors <= 6)
                & (transitions == 1)
                & condition3
                & condition4
            )
            if delete.any():
                image[delete] = 0
                changed = True
    return image.astype(bool)


def thin(mask: BoolArray) -> BoolArray:
    """把掩码细化成 1 像素宽的线：优先用 cv2.ximgproc.thinning，否则用本地实现。"""
    if not mask.any():
        return mask
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is not None and hasattr(ximgproc, "thinning"):
        result = ximgproc.thinning(
            mask.astype(np.uint8) * 255, thinningType=ximgproc.THINNING_ZHANGSUEN
        )
        return result > 0
    return _zhang_suen_thinning(mask)
