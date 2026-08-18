"""结构阶段：全分辨率信息降采样到目标网格，推导轮廓与内部细节。"""

from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from .binary_image import down_mask, thin  # 兼容历史导入；实现位于 binary_image
from .debug import DebugImageWriter

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
LabelArray = NDArray[np.int16]

_SMALL_DETAIL_MIN_LENGTH: Final = 4


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _cells(length: int, size: int):
    """把长度为 length 的区间均匀切成 size 段，逐段产出 (start, end)。"""
    step = length / size
    for idx in range(size):
        start = int(idx * step)
        end = max(start + 1, int((idx + 1) * step))
        yield start, min(end, length)


def bool_open(mask: BoolArray, iterations: int = 1) -> BoolArray:
    """布尔掩码开运算（先腐蚀后膨胀）：抹掉细小毛刺与孤立点。"""
    kernel = np.ones((3, 3), np.uint8)
    return (
        cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=iterations
        )
        > 0
    )


def bool_close(mask: BoolArray, iterations: int = 1) -> BoolArray:
    """布尔掩码闭运算（先膨胀后腐蚀）：填充小孔、连接断口。"""
    kernel = np.ones((3, 3), np.uint8)
    return (
        cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=iterations
        )
        > 0
    )


def bool_dilate(mask: BoolArray, iterations: int = 1) -> BoolArray:
    """布尔掩码膨胀：向外扩展一圈。"""
    return (
        cv2.dilate(
            mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=iterations
        )
        > 0
    )


def alpha_inner_boundary(mask: BoolArray) -> BoolArray:
    """掩码内侧边界：腐蚀后消失的那一圈像素，用作精灵轮廓。"""
    binary = mask.astype(np.uint8)
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)
    return mask & ~(eroded.astype(bool))


def _largest_reasonable_components(mask: BoolArray, min_fraction: float) -> BoolArray:
    """只保留面积 >= mask.size * min_fraction 的连通分量，滤除零星噪点。"""
    binary = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return mask
    min_area = max(4, int(mask.size * min_fraction))
    keep = np.zeros_like(mask, dtype=bool)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            keep |= labels == label
    return keep


# ---------------------------------------------------------------------------
# 降采样到目标网格
# ---------------------------------------------------------------------------


def down_area(image: np.ndarray, width: int, height: int) -> FloatArray:
    """用面积平均（INTER_AREA）把图像缩到 width x height。"""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )


def down_labels_majority(
    labels: LabelArray,
    valid: BoolArray,
    width: int,
    height: int,
    default: int = -1,
) -> LabelArray:
    """标签图降采样：每个目标网格取原区域内有效像素中出现次数最多的标签。"""
    h, w = labels.shape
    out = np.full((height, width), default, dtype=labels.dtype)
    rows = list(_cells(h, height))
    cols = list(_cells(w, width))
    for iy, (y0, y1) in enumerate(rows):
        for ix, (x0, x1) in enumerate(cols):
            local_valid = valid[y0:y1, x0:x1]
            if not local_valid.any():
                continue
            values = labels[y0:y1, x0:x1][local_valid]
            values, counts = np.unique(values[values >= 0], return_counts=True)
            if len(values):
                out[iy, ix] = values[int(np.argmax(counts))]
    return out


def down_edges_coverage(
    edges: np.ndarray, width: int, height: int, threshold: float
) -> BoolArray:
    """边缘图降采样：网格内边缘覆盖率 >= threshold 才保留为边缘像素。"""
    avg = (
        cv2.resize(
            edges.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
        / 255.0
    )
    return avg >= threshold


def _down_family_weighted_L(
    l_channel: FloatArray,
    family_labels: LabelArray,
    family_down: LabelArray,
    foreground: BoolArray,
    width: int,
    height: int,
) -> FloatArray:
    """对每个目标网格，只统计获胜 family 的前景像素明度均值。

    没有有效像素的格子 fallback 到整格面积均值。
    """
    h, w = l_channel.shape
    out = np.zeros((height, width), dtype=np.float32)
    rows = list(_cells(h, height))
    cols = list(_cells(w, width))

    for iy, (y0, y1) in enumerate(rows):
        for ix, (x0, x1) in enumerate(cols):
            local_fg = foreground[y0:y1, x0:x1]
            if not local_fg.any():
                out[iy, ix] = float(l_channel[y0:y1, x0:x1].mean())
                continue
            local_family = family_labels[y0:y1, x0:x1]
            target = family_down[iy, ix]
            mask = local_fg & (local_family == target)
            if mask.any():
                out[iy, ix] = float(l_channel[y0:y1, x0:x1][mask].mean())
            else:
                out[iy, ix] = float(l_channel[y0:y1, x0:x1][local_fg].mean())

    return out


def _quantize_tiers(
    l_down: FloatArray,
    family_down: LabelArray,
    valid: BoolArray,
    ramp_l: FloatArray,
) -> LabelArray:
    """从降采样明度图和 family ramp 重新计算 tier_down。

    每个有效像素取 ramp_l[family] 中与 l_down 最接近的档位。
    """
    out = np.full(family_down.shape, -1, dtype=np.int16)
    for f in range(len(ramp_l)):
        mask = valid & (family_down == f)
        if not mask.any():
            continue
        l_vals = l_down[mask]
        rl = ramp_l[f]
        diffs = np.abs(l_vals[:, None] - rl[None, :])
        out[mask] = np.argmin(diffs, axis=1).astype(np.int16)
    return out


# ---------------------------------------------------------------------------
# 线条后处理
# ---------------------------------------------------------------------------


def remove_short_components(mask: BoolArray, min_length: int) -> BoolArray:
    """删除面积小于 min_length 的连通分量。"""
    if min_length <= 1 or not mask.any():
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    out = np.zeros_like(mask, dtype=bool)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= min_length:
            out |= labels == label
    return out


# ---------------------------------------------------------------------------
# 结构分析
# ---------------------------------------------------------------------------


def label_boundary(labels: LabelArray, valid: BoolArray) -> BoolArray:
    """标签图边界：相邻有效像素的标签不同处即为边界（两侧各标 1 像素）。"""
    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = valid[:, :-1] & valid[:, 1:] & (labels[:, :-1] != labels[:, 1:])
    vertical = valid[:-1, :] & valid[1:, :] & (labels[:-1, :] != labels[1:, :])
    boundary[:, :-1] |= horizontal
    boundary[:, 1:] |= horizontal
    boundary[:-1, :] |= vertical
    boundary[1:, :] |= vertical
    return boundary


def local_majority_smooth(
    labels: LabelArray,
    valid: BoolArray,
    protect: BoolArray,
    passes: int,
    min_count: int,
) -> LabelArray:
    """局部多数投票平滑：邻域内某标签得票 >= min_count 才改写中心像素。

    protect 中的像素保持不动。
    """
    out = labels.copy()
    h, w = out.shape
    for _ in range(max(0, passes)):
        new = out.copy()
        for y in range(h):
            y0, y1 = max(0, y - 1), min(h, y + 2)
            for x in range(w):
                if not valid[y, x] or protect[y, x]:
                    continue
                x0, x1 = max(0, x - 1), min(w, x + 2)
                sub_valid = valid[y0:y1, x0:x1]
                vals = out[y0:y1, x0:x1][sub_valid]
                vals = vals[vals >= 0]
                if len(vals) == 0:
                    continue
                uniq, counts = np.unique(vals, return_counts=True)
                idx = int(np.argmax(counts))
                if counts[idx] >= min_count:
                    new[y, x] = uniq[idx]
        out = new
    return out


def repair_missing_family(family_down: LabelArray, alpha_down: BoolArray) -> LabelArray:
    """把 alpha 内缺失的色相族标签（-1）用最近的有效标签补齐。"""
    missing = alpha_down & (family_down < 0)
    if not missing.any():
        family_down[~alpha_down] = -1
        return family_down
    coords = np.argwhere((family_down >= 0) & alpha_down)
    if len(coords) == 0:
        family_down[alpha_down] = 0
        family_down[~alpha_down] = -1
        return family_down
    for y, x in np.argwhere(missing):
        idx = int(np.argmin(np.sum((coords - np.array([y, x])) ** 2, axis=1)))
        sy, sx = coords[idx]
        family_down[y, x] = family_down[sy, sx]
    family_down[~alpha_down] = -1
    return family_down


# ---------------------------------------------------------------------------
# 小尺寸精灵清理
# ---------------------------------------------------------------------------


def _simplify_small_sprite(
    *,
    alpha_down: BoolArray,
    family_down: LabelArray,
    tier_down: LabelArray,
    canny_down: BoolArray,
    internal_detail: BoolArray,
    width: int,
    height: int,
    small_cleanup_threshold: int,
    small_cleanup_passes: int,
    small_tier_smooth_majority: int,
    small_skip_canny_under: int,
    edge_canny_support_radius: int,
    edge_min_length: int,
) -> dict:
    """小尺寸精灵的语义清理通道。

    小网格上面积平均会留下大量杂点与不稳定的明度过渡，通过
    闭孔、去碎块、局部多数平滑等操作让画面更干净。
    """
    output_size = max(width, height)
    if output_size > small_cleanup_threshold:
        return {
            "alpha_down": alpha_down,
            "family_down": family_down,
            "tier_down": tier_down,
            "silhouette": alpha_inner_boundary(alpha_down),
            "family_boundary": label_boundary(family_down, alpha_down),
            "shade_boundary": label_boundary(tier_down, alpha_down),
            "canny_down": canny_down,
            "internal_detail": internal_detail,
            "outline": alpha_inner_boundary(alpha_down) | internal_detail,
            "small_cleanup_applied": False,
        }

    # 闭合小孔并去掉过小的孤立连通块
    alpha = bool_close(alpha_down, 1)
    alpha = _largest_reasonable_components(alpha, max(0.001, 2.0 / alpha.size))

    family = repair_missing_family(family_down.copy(), alpha)
    tier = tier_down.copy()

    # 轮廓在平滑时始终受保护；正在平滑的维度，其边界不再保护自身
    silhouette = alpha_inner_boundary(alpha)

    # 平滑 family：只保护轮廓（不保护 family_boundary，否则 1px 噪声改不掉）
    family = local_majority_smooth(family, alpha, silhouette, small_cleanup_passes, 5)
    family = repair_missing_family(family, alpha)
    family_boundary = label_boundary(family, alpha)

    # 平滑 tier：保护轮廓 + 色相族边界（不保护 shade_boundary 自身）
    shade_boundary = label_boundary(tier, alpha)
    tier = np.clip(
        local_majority_smooth(
            tier,
            alpha,
            silhouette | family_boundary,
            small_cleanup_passes,
            small_tier_smooth_majority,
        ),
        0,
        np.max(tier),
    )
    shade_boundary = label_boundary(tier, alpha)

    # 网格太小时 Canny 细节基本无意义，直接关闭
    canny = canny_down & alpha
    if output_size <= small_skip_canny_under:
        canny[:] = False

    support = bool_dilate(silhouette | family_boundary, edge_canny_support_radius)
    internal_detail = (
        (family_boundary | (canny & support)) & alpha & ~shade_boundary & ~silhouette
    )
    internal_detail = remove_short_components(
        internal_detail, max(edge_min_length, _SMALL_DETAIL_MIN_LENGTH)
    )
    internal_detail = thin(internal_detail)

    return {
        "alpha_down": alpha,
        "family_down": family,
        "tier_down": tier,
        "silhouette": silhouette,
        "family_boundary": family_boundary,
        "shade_boundary": shade_boundary,
        "canny_down": canny,
        "internal_detail": internal_detail,
        "outline": silhouette | internal_detail,
        "small_cleanup_applied": True,
    }


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def structure(
    perceived: dict,
    *,
    width: int,
    height: int,
    alpha_coverage: float,
    edge_coverage: float,
    edge_min_length: int,
    edge_canny_support_radius: int,
    small_cleanup_threshold: int,
    small_cleanup_passes: int,
    small_tier_smooth_majority: int,
    small_skip_canny_under: int,
    debug: bool,
    debug_dir: Path,
) -> dict:
    """阶段 B：把全分辨率信息降采样到 width x height 网格，推导轮廓与内部细节。

    perceived 为 :func:`perceive` 的输出字典。
    debug 为 False 时跳过调试 PNG 写入；debug_dir 语义不变（定位输出目录）。
    """

    debug_images = DebugImageWriter(debug, debug_dir)

    # ── 降采样 ──
    alpha_down = down_mask(perceived["alpha_full"], width, height, alpha_coverage)
    family_down = down_labels_majority(
        perceived["family_labels"], perceived["foreground"], width, height
    )
    family_down = repair_missing_family(family_down, alpha_down)
    l_down = _down_family_weighted_L(
        perceived["L"],
        perceived["family_labels"],
        family_down,
        perceived["foreground"],
        width,
        height,
    )
    tier_down = _quantize_tiers(l_down, family_down, alpha_down, perceived["ramp_l"])

    debug_images.save("09_alpha_down", alpha_down)

    # ── 轮廓、Canny 细节、色相族边界、色阶边界 ──
    silhouette = alpha_inner_boundary(alpha_down)
    canny_down = (
        down_edges_coverage(perceived["canny"], width, height, edge_coverage)
        & alpha_down
    )
    family_boundary = label_boundary(family_down, alpha_down)
    shade_boundary = label_boundary(tier_down, alpha_down)

    # Canny 线只有在靠近真实结构（轮廓/色相族边界）时才被采纳
    support = bool_dilate(silhouette | family_boundary, edge_canny_support_radius)
    internal_detail = (
        (family_boundary | (canny_down & support))
        & alpha_down
        & ~shade_boundary
        & ~silhouette
    )
    internal_detail = remove_short_components(internal_detail, edge_min_length)
    internal_detail = thin(internal_detail)
    outline = silhouette | internal_detail

    debug_images.save("12_silhouette", silhouette)
    debug_images.save("13_canny_down", canny_down)
    debug_images.save("14_family_boundary", family_boundary)
    debug_images.save("15_shade_boundary", shade_boundary)
    debug_images.save("16_internal_detail", internal_detail)
    debug_images.save("17_outline", outline)

    # ── 小尺寸精灵清理 ──
    cleaned = _simplify_small_sprite(
        alpha_down=alpha_down,
        family_down=family_down,
        tier_down=tier_down,
        canny_down=canny_down,
        internal_detail=internal_detail,
        width=width,
        height=height,
        small_cleanup_threshold=small_cleanup_threshold,
        small_cleanup_passes=small_cleanup_passes,
        small_tier_smooth_majority=small_tier_smooth_majority,
        small_skip_canny_under=small_skip_canny_under,
        edge_canny_support_radius=edge_canny_support_radius,
        edge_min_length=edge_min_length,
    )

    return {
        "alpha_down": cleaned["alpha_down"],
        "family_down": cleaned["family_down"],
        "tier_down": cleaned["tier_down"],
        "L_down": l_down,
        "silhouette": cleaned["silhouette"],
        "canny_down": cleaned["canny_down"],
        "family_boundary": cleaned["family_boundary"],
        "shade_boundary": cleaned["shade_boundary"],
        "internal_detail": cleaned["internal_detail"],
        "outline": cleaned["outline"],
        "small_cleanup_applied": cleaned["small_cleanup_applied"],
    }
