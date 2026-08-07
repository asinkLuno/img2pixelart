"""调色盘拟合：把自适应色板映射到目标像素画调色盘。

将目标调色盘组织为色相族明度阶梯（ramp），再把源图的每个色相族
匹配到最合适的调色盘 ramp，最后按像素的明度位置重新分配档位。
"""

import cv2
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
LabelArray = NDArray[np.int16]


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
    chroma_floor: float = 5.0,
    hue_gap_degrees: float = 25.0,
) -> list[tuple[FloatArray, NDArray[np.uint8]]]:
    """把平铺调色板组织为色相族明度阶梯列表。

    每个阶梯是 ``(lab_ramp, bgr_ramp)``，其中：
    - lab_ramp: ``(S_i, 3)`` float32 Lab，按明度升序
    - bgr_ramp: ``(S_i, 3)`` uint8 BGR，与 lab_ramp 一一对应的精确调色盘颜色

    中性色（色度 < chroma_floor）归入一个独立阶梯。
    彩色按色相角贪心聚类，角差距 > hue_gap_degrees 则新建一族。
    """
    bgr_f32 = palette_bgr.astype(np.float32) / 255.0
    lab: FloatArray = cv2.cvtColor(
        bgr_f32[None, ...], cv2.COLOR_BGR2LAB
    )[0]

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
) -> tuple[int, int]:
    """从调色盘 ramp 结构中自动推导 ``(requested_groups, ramp_steps)``。

    - requested_groups = max(2, 彩色 ramp 数)
    - ramp_steps = median(所有 ramp 的原生长度)，至少为 3
    """
    chroma_floor = 3.0
    chromatic_count = 0
    all_steps: list[int] = []
    for lab, _ in palette_ramps:
        mean_chroma = float(np.linalg.norm(lab[:, 1:3], axis=1).mean())
        if mean_chroma >= chroma_floor:
            chromatic_count += 1
        all_steps.append(len(lab))

    requested_groups = max(2, chromatic_count)
    ramp_steps = max(3, int(np.median(all_steps)))
    return requested_groups, ramp_steps


# ---------------------------------------------------------------------------
# 阶梯采样
# ---------------------------------------------------------------------------


def _sample_ramp(
    ramp_lab: FloatArray,
    ramp_bgr: NDArray[np.uint8],
    steps: int,
) -> tuple[FloatArray, FloatArray]:
    """从调色盘阶梯中等距采样 steps 档，保留精确 BGR 色值。

    返回 ``(sampled_lab, sampled_bgr)``，均为 float32。
    """
    src_l = ramp_lab[:, 0]
    lo, hi = float(src_l.min()), float(src_l.max())
    tgt_l = np.linspace(lo, hi, steps, dtype=np.float32)

    result_lab = np.zeros((steps, 3), dtype=np.float32)
    result_bgr = np.zeros((steps, 3), dtype=np.float32)
    for i, tl in enumerate(tgt_l):
        idx = int(np.argmin(np.abs(src_l - tl)))
        result_lab[i] = ramp_lab[idx]
        result_bgr[i] = ramp_bgr[idx].astype(np.float32)

    result_lab[:, 0] = tgt_l  # 用均匀明度替代真实明度，供连续位置计算
    return result_lab, result_bgr


def build_target_ramps(
    palette_ramps: list[tuple[FloatArray, NDArray[np.uint8]]],
    steps: int,
) -> tuple[FloatArray, FloatArray]:
    """把调色盘阶梯统一采样为 ``(F, steps, 3)`` BGR 和 ``(F, steps)`` L。

    颜色取自原始调色盘精确 BGR 值（非 Lab 插值），确保输出只用调色盘颜色。
    返回 ``(ramps_bgr, ramp_l)``。
    """
    F = len(palette_ramps)
    ramps_bgr = np.zeros((F, steps, 3), dtype=np.float32)
    ramp_l = np.zeros((F, steps), dtype=np.float32)

    for t in range(F):
        lab, bgr = palette_ramps[t]
        sampled_lab, sampled_bgr = _sample_ramp(lab, bgr, steps)
        ramp_l[t] = sampled_lab[:, 0]
        ramps_bgr[t] = sampled_bgr

    return ramps_bgr, ramp_l


# ---------------------------------------------------------------------------
# 色相族匹配
# ---------------------------------------------------------------------------


def match_families(
    source_hue_directions: FloatArray,
    source_ramp_l: FloatArray,
    source_ramps_bgr: FloatArray,
    target_ramps: list[tuple[FloatArray, NDArray[np.uint8]]],
) -> NDArray[np.int32]:
    """为每个源色相族选择最匹配的目标调色盘阶梯。

    返回 ``(F_src,)`` int32 数组，值域 ``0 .. F_tgt-1``。
    允许多个源族映射到同一个目标阶梯（多对一）。
    """
    F_src = len(source_hue_directions)
    F_tgt = len(target_ramps)

    # ── 源族描述符 ──
    src_neutral = np.array([
        float(np.linalg.norm(d)) < 1e-6 for d in source_hue_directions
    ])
    src_median_l = np.array([
        float(np.median(r)) for r in source_ramp_l
    ])

    src_lab_flat = cv2.cvtColor(
        (source_ramps_bgr.astype(np.float32) / 255.0).reshape(-1, 1, 3),
        cv2.COLOR_BGR2LAB,
    ).reshape(F_src, -1, 3)
    src_chroma = np.array([
        float(np.linalg.norm(s[:, 1:3], axis=1).mean()) for s in src_lab_flat
    ])

    # ── 目标族描述符 ──
    tgt_median_l = np.array([float(np.median(r[0][:, 0])) for r in target_ramps])
    tgt_chroma = np.array([
        float(np.linalg.norm(r[0][:, 1:3], axis=1).mean()) for r in target_ramps
    ])
    tgt_hue: FloatArray = np.zeros((F_tgt, 2), dtype=np.float32)
    for i, (ramp_lab, _) in enumerate(target_ramps):
        ab = ramp_lab[:, 1:3]
        weights = np.linalg.norm(ab, axis=1)
        if weights.max() > 1e-6:
            weighted = (ab * weights[:, None]).sum(axis=0)
            norm = float(np.linalg.norm(weighted))
            tgt_hue[i] = weighted / max(norm, 1e-6)
    tgt_neutral = np.array([float(np.linalg.norm(h)) < 1e-6 for h in tgt_hue])

    # ── 贪心匹配 ──
    mapping = np.full(F_src, -1, dtype=np.int32)
    for s in range(F_src):
        best_cost = float("inf")
        best_t = 0
        for t in range(F_tgt):
            # 中性族必须匹配中性族，彩色族必须匹配彩色族
            if src_neutral[s] != tgt_neutral[t]:
                continue

            cost = 0.0
            if not src_neutral[s] and not tgt_neutral[t]:
                hue_sim = float(np.dot(source_hue_directions[s], tgt_hue[t]))
                cost += 1.0 - hue_sim

            cost += abs(src_median_l[s] - tgt_median_l[t]) / 100.0
            cost += abs(src_chroma[s] - tgt_chroma[t]) / 100.0

            if cost < best_cost:
                best_cost = cost
                best_t = t

        mapping[s] = best_t

    return mapping


# ---------------------------------------------------------------------------
# 像素重着色
# ---------------------------------------------------------------------------


def remap_families_and_tiers(
    family_down: LabelArray,
    tier_down: LabelArray,
    valid: BoolArray,
    l_down: FloatArray,
    source_ramp_l: FloatArray,
    target_ramp_l: FloatArray,
    family_mapping: NDArray[np.int32],
) -> tuple[LabelArray, LabelArray]:
    """把 family_down 和 tier_down 从源阶梯空间映射到目标阶梯空间。

    family_down 索引从 0..F_src-1 重映射为 0..F_tgt-1。
    tier_down 按像素明度就近落到目标阶梯的最近档位。

    返回 ``(new_family_down, new_tier_down)``。
    """
    F_src = len(family_mapping)

    new_family = family_down.copy()
    for s in range(F_src):
        new_family[family_down == s] = family_mapping[s]
    new_family[~valid] = -1

    new_tier = np.full(family_down.shape, -1, dtype=np.int16)
    for s in range(F_src):
        t = int(family_mapping[s])
        mask = valid & (family_down == s)
        if not mask.any():
            continue
        l_vals = l_down[mask]
        tgt_l = target_ramp_l[t]
        diffs = np.abs(l_vals[:, None] - tgt_l[None, :])
        new_tier[mask] = np.argmin(diffs, axis=1).astype(np.int16)

    return new_family, new_tier
