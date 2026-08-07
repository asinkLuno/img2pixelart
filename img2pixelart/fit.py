"""调色盘加载与结构化。

把目标像素画调色板组织为色相族明度阶梯（ramp），
供后续使用（例如自动配置梯度参数）。
"""

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


# ---------------------------------------------------------------------------
# 调色盘结构化：平铺颜色 → 色相族明度阶梯
# ---------------------------------------------------------------------------


def _angular_distance(a: float, b: float) -> float:
    """两个弧度角的最小绝对距离（[0, π]）。"""
    d = abs(a - b)
    return min(d, 2 * np.pi - d)


def structure_palette_ramps(
    palette_bgr: NDArray[np.uint8],
    chroma_floor: float,
    hue_gap_degrees: float,
) -> list[tuple[FloatArray, NDArray[np.uint8]]]:
    """把平铺调色板组织为色相族明度阶梯列表。

    每个阶梯是 ``(lab_ramp, bgr_ramp)``，其中：
    - lab_ramp: ``(S_i, 3)`` float32 Lab，按明度升序
    - bgr_ramp: ``(S_i, 3)`` uint8 BGR，与 lab_ramp 一一对应的精确调色盘颜色

    中性色（色度 < chroma_floor）归入一个独立阶梯。
    彩色按色相角贪心聚类，角差距 > hue_gap_degrees 则新建一族。
    """
    bgr_f32 = palette_bgr.astype(np.float32) / 255.0
    lab: FloatArray = cv2.cvtColor(bgr_f32[None, ...], cv2.COLOR_BGR2LAB)[0]

    chroma = np.linalg.norm(lab[:, 1:3], axis=1)
    hue = np.arctan2(lab[:, 2], lab[:, 1])

    ramps: list[tuple[FloatArray, NDArray[np.uint8]]] = []

    def _make_ramp(indices: NDArray[np.int_]) -> tuple[FloatArray, NDArray[np.uint8]]:
        order = np.argsort(lab[indices, 0])
        idx_sorted = indices[order]
        return lab[idx_sorted], palette_bgr[idx_sorted]

    # ── 中性色 ──
    n_idx = np.where(chroma < chroma_floor)[0]
    if len(n_idx) > 0:
        ramps.append(_make_ramp(n_idx))

    # ── 彩色：贪心色相聚类 ──
    c_idx = np.where(chroma >= chroma_floor)[0]
    if len(c_idx) == 0:
        return ramps

    gap_rad = np.deg2rad(hue_gap_degrees)
    c_order = np.argsort(-chroma[c_idx])
    c_idx = c_idx[c_order]

    clusters: list[tuple[float, list[int]]] = []
    for idx in c_idx:
        h = hue[idx]
        assigned = False
        for ci, (mean_h, members) in enumerate(clusters):
            if _angular_distance(h, mean_h) < gap_rad:
                members.append(idx)
                hues_in_cluster = np.array([hue[m] for m in members])
                mean_sin = float(np.sin(hues_in_cluster).mean())
                mean_cos = float(np.cos(hues_in_cluster).mean())
                clusters[ci] = (np.arctan2(mean_sin, mean_cos), members)
                assigned = True
                break
        if not assigned:
            clusters.append((h, [idx]))

    for _, members in clusters:
        ramps.append(_make_ramp(np.array(members, dtype=np.int_)))

    return ramps


# ---------------------------------------------------------------------------
# 自动配置
# ---------------------------------------------------------------------------


def palette_auto_config(
    palette_ramps: list[tuple[FloatArray, NDArray[np.uint8]]],
    auto_chroma_floor: float,
) -> tuple[int, int]:
    """从调色盘 ramp 结构中自动推导 ``(requested_groups, ramp_steps)``。

    - requested_groups = max(2, 彩色 ramp 数)
    - ramp_steps = median(所有 ramp 的原生长度)，至少为 3
    """
    chromatic_count = 0
    all_steps: list[int] = []
    for lab, _ in palette_ramps:
        mean_chroma = float(np.linalg.norm(lab[:, 1:3], axis=1).mean())
        if mean_chroma >= auto_chroma_floor:
            chromatic_count += 1
        all_steps.append(len(lab))

    requested_groups = max(2, chromatic_count)
    ramp_steps = max(3, int(np.median(all_steps)))
    return requested_groups, ramp_steps
