"""结构阶段：全分辨率信息降采样到目标网格，推导轮廓与内部细节。

所有可配置参数由 Hydra 配置显式传入，本模块不设代码默认值。
"""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
LabelArray = NDArray[np.int16]


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


def down_area(image: np.ndarray, size: int) -> FloatArray:
    """用面积平均（INTER_AREA）把图像缩到 size x size。"""
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )


def down_mask(mask: np.ndarray, size: int, threshold: float) -> BoolArray:
    """掩码降采样：目标网格内原掩码的覆盖率 >= threshold 才置 1。"""
    average = cv2.resize(
        mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA
    )
    if mask.max(initial=0) > 1:
        average = average / 255.0
    return average >= threshold


def down_labels_majority(
    labels: LabelArray,
    valid: BoolArray,
    size: int,
    default: int = -1,
) -> LabelArray:
    """标签图降采样：每个目标网格取原区域内有效像素中出现次数最多的标签。"""
    h, w = labels.shape
    out = np.full((size, size), default, dtype=labels.dtype)
    rows = list(_cells(h, size))
    cols = list(_cells(w, size))
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


def down_edges_coverage(edges: np.ndarray, size: int, threshold: float) -> BoolArray:
    """边缘图降采样：网格内边缘覆盖率 >= threshold 才保留为边缘像素。"""
    avg = (
        cv2.resize(edges.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
        / 255.0
    )
    return avg >= threshold


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
    size: int,
    small_cleanup_threshold: int,
    small_hole_close: bool,
    small_cleanup_passes: int,
    small_tier_smooth_majority: int,
    small_skip_canny_under: int,
    edge_canny_support_radius: int,
    edge_min_length: int,
    small_detail_min_length: int,
    edge_open_before_thin: bool,
) -> dict:
    """小尺寸精灵的语义清理通道。

    小网格上面积平均会留下大量杂点与不稳定的明度过渡，通过
    闭孔、去碎块、局部多数平滑等操作让画面更干净。
    """
    if size > small_cleanup_threshold:
        return {
            "alpha_down": alpha_down,
            "family_down": family_down,
            "tier_down": tier_down,
            "silhouette": alpha_inner_boundary(alpha_down),
            "family_boundary": label_boundary(family_down, alpha_down),
            "shade_boundary": label_boundary(tier_down, alpha_down),
            "canny_down": canny_down,
            "internal_detail": np.zeros_like(alpha_down, dtype=bool),
            "outline": alpha_inner_boundary(alpha_down),
            "small_cleanup_applied": False,
        }

    # 闭合小孔并去掉过小的孤立连通块
    alpha = alpha_down.copy()
    if small_hole_close:
        alpha = bool_close(alpha, 1)
    alpha = _largest_reasonable_components(alpha, max(0.001, 2.0 / alpha.size))

    family = repair_missing_family(family_down.copy(), alpha)
    tier = tier_down.copy()

    # 轮廓 / 色相族边界 / 色阶边界在平滑时受保护
    silhouette = alpha_inner_boundary(alpha)
    family_boundary = label_boundary(family, alpha)
    shade_boundary = label_boundary(tier, alpha)

    protect = silhouette | family_boundary | shade_boundary
    family = local_majority_smooth(family, alpha, protect, small_cleanup_passes, 5)
    family = repair_missing_family(family, alpha)
    family_boundary = label_boundary(family, alpha)
    protect = silhouette | family_boundary | shade_boundary
    tier = np.clip(
        local_majority_smooth(
            tier,
            alpha,
            protect,
            small_cleanup_passes,
            small_tier_smooth_majority,
        ),
        0,
        np.max(tier),
    )
    shade_boundary = label_boundary(tier, alpha)

    # 网格太小时 Canny 细节基本无意义，直接关闭
    canny = canny_down & alpha
    if size <= small_skip_canny_under:
        canny[:] = False

    support = bool_dilate(silhouette | family_boundary, edge_canny_support_radius)
    internal_detail = (
        (family_boundary | (canny & support)) & alpha & ~shade_boundary & ~silhouette
    )
    min_len = max(edge_min_length, small_detail_min_length)
    internal_detail = remove_short_components(internal_detail, min_len)
    if edge_open_before_thin:
        internal_detail = bool_open(internal_detail, 1)
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
    size: int,
    alpha_coverage: float,
    edge_coverage: float,
    edge_min_length: int,
    edge_canny_support_radius: int,
    edge_open_before_thin: bool,
    small_cleanup_threshold: int,
    small_detail_min_length: int,
    small_cleanup_passes: int,
    small_hole_close: bool,
    small_tier_smooth_majority: int,
    small_skip_canny_under: int,
    debug_dir: Path,
) -> dict:
    """阶段 B：把全分辨率信息降采样到 size x size 网格，推导轮廓与内部细节。

    perceived 为 :func:`perceive` 的输出字典。
    debug_dir 非空时，每步结果即时保存为 PNG。
    """

    def _save(name: str, img: np.ndarray) -> None:
        if img.dtype == bool:
            img = img.astype(np.uint8) * 255
        cv2.imwrite(str(debug_dir / f"{name}.png"), img)

    # ── 降采样 ──
    alpha_down = down_mask(perceived["alpha_full"], size, alpha_coverage)
    family_down = down_labels_majority(
        perceived["family_labels"], perceived["foreground"], size
    )
    tier_down = down_labels_majority(
        perceived["tier"], perceived["foreground"], size, default=0
    )
    l_down = down_area(perceived["L"], size)
    family_down = repair_missing_family(family_down, alpha_down)

    _save("09_alpha_down", alpha_down.astype(np.uint8) * 255)

    # ── 轮廓、Canny 细节、色相族边界、色阶边界 ──
    silhouette = alpha_inner_boundary(alpha_down)
    canny_down = (
        down_edges_coverage(perceived["canny"], size, edge_coverage) & alpha_down
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
    if edge_open_before_thin:
        internal_detail = bool_open(internal_detail, 1)
    internal_detail = thin(internal_detail)
    outline = silhouette | internal_detail

    _save("12_silhouette", silhouette)
    _save("13_canny_down", canny_down)
    _save("14_family_boundary", family_boundary)
    _save("15_shade_boundary", shade_boundary)
    _save("16_internal_detail", internal_detail)
    _save("17_outline", outline)

    # ── 小尺寸精灵清理 ──
    cleaned = _simplify_small_sprite(
        alpha_down=alpha_down,
        family_down=family_down,
        tier_down=tier_down,
        canny_down=canny_down,
        size=size,
        small_cleanup_threshold=small_cleanup_threshold,
        small_hole_close=small_hole_close,
        small_cleanup_passes=small_cleanup_passes,
        small_tier_smooth_majority=small_tier_smooth_majority,
        small_skip_canny_under=small_skip_canny_under,
        edge_canny_support_radius=edge_canny_support_radius,
        edge_min_length=edge_min_length,
        small_detail_min_length=small_detail_min_length,
        edge_open_before_thin=edge_open_before_thin,
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
