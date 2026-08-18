"""渲染阶段：按 (色相族, 明度档) 取色渲染，仅在合适处抖动，叠加同色相轮廓。"""

from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
LabelArray = NDArray[np.int16]

# 抖动区域与梯度阈值：算法细节，固化。
_DITHER_FRACTION_MIN: Final = 0.18
_DITHER_FRACTION_MAX: Final = 0.82
_DITHER_GRADIENT_MIN: Final = 0.8
# 轮廓暗化强度映射曲线系数（见 render 的 darkness 参数说明）。
_SILHOUETTE_SCALE_COEFF: Final = 0.25
_INTERNAL_STEPS_MAX: Final = 2
_INTERNAL_SCALE_COEFF: Final = 0.4

# 4x4 Bayer 有序抖动阈值矩阵（已归一化到 [0,1)）。
BAYER_4X4: FloatArray = (
    np.array(
        [
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5],
        ],
        dtype=np.float32,
    )
    / 16.0
)


# ── 微图案字典：4×4 二值图案，coverage 0/16 .. 16/16 ──


def _make_patterns(pixel_order: np.ndarray) -> np.ndarray:
    """按给定的像素访问顺序生成 17 个 4×4 二值图案。

    pixel_order: (16, 2) 数组，按优先级排列的 (row, col) 位置。
    返回 (17, 4, 4) bool 数组，patterns[k] 恰好有 k 个 True。
    """
    patterns = np.zeros((17, 4, 4), dtype=bool)
    for k in range(1, 17):
        for i in range(k):
            r, c = pixel_order[i]
            patterns[k, r, c] = True
    return patterns


def _bayer_order() -> np.ndarray:
    """Bayer 矩阵从小到大排序的像素访问顺序。"""
    flat_order = np.argsort(BAYER_4X4.ravel())
    return np.stack([flat_order // 4, flat_order % 4], axis=1)


# 对角线扫描顺序
_DIAG_ORDER: np.ndarray = np.array(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [2, 0],
        [1, 1],
        [0, 2],
        [3, 0],
        [2, 1],
        [1, 2],
        [0, 3],
        [3, 1],
        [2, 2],
        [1, 3],
        [3, 2],
        [2, 3],
        [3, 3],
    ],
    dtype=np.int16,
)

# 聚类顺序（中心向外扩散）
_CLUSTER_ORDER: np.ndarray = np.array(
    [
        [1, 1],
        [1, 2],
        [2, 1],
        [2, 2],
        [0, 1],
        [0, 2],
        [3, 1],
        [3, 2],
        [1, 0],
        [1, 3],
        [2, 0],
        [2, 3],
        [0, 0],
        [0, 3],
        [3, 0],
        [3, 3],
    ],
    dtype=np.int16,
)

PATTERNS: dict[str, np.ndarray] = {
    "ordered": _make_patterns(_bayer_order()),
    "diagonal": _make_patterns(_DIAG_ORDER),
    "clustered": _make_patterns(_CLUSTER_ORDER),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def neighbor_max_difference(grid: np.ndarray) -> FloatArray:
    """计算每个像素与上下左右邻居的最大差值，衡量局部梯度强度。"""
    grid = np.asarray(grid, dtype=np.float32)
    result = np.zeros(grid.shape[:2], dtype=np.float32)
    horizontal = np.abs(grid[:, 1:] - grid[:, :-1])
    vertical = np.abs(grid[1:, :] - grid[:-1, :])
    if grid.ndim == 3:
        horizontal = horizontal.max(axis=-1)
        vertical = vertical.max(axis=-1)
    result[:, 1:] = np.maximum(result[:, 1:], horizontal)
    result[:, :-1] = np.maximum(result[:, :-1], horizontal)
    result[1:, :] = np.maximum(result[1:, :], vertical)
    result[:-1, :] = np.maximum(result[:-1, :], vertical)
    return result


def continuous_ramp_position(
    l_down: FloatArray,
    family_down: LabelArray,
    ramp_l: FloatArray,
    valid: BoolArray,
    steps_per_family: NDArray[np.int32],
) -> FloatArray:
    """把每个像素的明度线性映射到其色相族阶梯内的连续位置 [0, S_f-1]。

    该连续值供抖动阶段在相邻两档之间做插值取舍。
    steps_per_family 为各族的原生长度。
    """
    position = np.zeros(l_down.shape, dtype=np.float32)
    for group in range(len(ramp_l)):
        mask = valid & (family_down == group)
        if not mask.any():
            continue
        s = int(steps_per_family[group])
        values = l_down[mask]
        position[mask] = np.interp(
            values, ramp_l[group, :s], np.arange(s, dtype=np.float32)
        )
    return position


def make_palette_strip(
    ramps: np.ndarray,
    steps_per_family: NDArray[np.int32],
    cell: int = 18,
) -> np.ndarray:
    """把各族阶梯色板拼成一张可视化条带（用于调试输出）。"""
    groups = len(steps_per_family)
    max_steps = int(steps_per_family.max())
    strip = np.zeros((groups * cell, max_steps * cell, 3), dtype=np.uint8)
    for group in range(groups):
        s = int(steps_per_family[group])
        for step in range(s):
            strip[
                group * cell : (group + 1) * cell,
                step * cell : (step + 1) * cell,
            ] = ramps[group, step].astype(np.uint8)
    return strip


# ---------------------------------------------------------------------------
# 抖动
# ---------------------------------------------------------------------------


def _render_floyd(
    positions: FloatArray,
    families: LabelArray,
    ramps: FloatArray,
    mask: BoolArray,
    steps_per_family: NDArray[np.int32],
) -> FloatArray:
    """Floyd–Steinberg 误差扩散抖动。

    误差只扩散给同一色相族的像素，避免跨族混色。
    """
    work = positions.astype(np.float32).copy()
    chosen = np.rint(work).astype(np.int16)
    h, w = work.shape
    kernel = (
        (0, 1, 7 / 16),
        (1, -1, 3 / 16),
        (1, 0, 5 / 16),
        (1, 1, 1 / 16),
    )
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            old = float(work[y, x])
            family = families[y, x]
            max_t = max(0, int(steps_per_family[family]) - 1)
            new = int(np.clip(round(old), 0, max_t))
            chosen[y, x] = new
            error = old - new
            for dy, dx, weight in kernel:
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < h
                    and 0 <= nx < w
                    and mask[ny, nx]
                    and families[ny, nx] == family
                ):
                    work[ny, nx] += error * weight
    safe_family = np.maximum(families, 0)
    bgr = ramps[safe_family, chosen]
    return bgr.astype(np.float32)


def _render_pattern(
    positions: FloatArray,
    families: LabelArray,
    ramps: FloatArray,
    mask: BoolArray,
    pattern_set: np.ndarray,
    steps_per_family: NDArray[np.int32],
) -> FloatArray:
    """微图案抖动：用 4×4 图案字典在相邻两档之间选择。

    每个像素根据连续位置的小数部分量化到 0/16..16/16 覆盖率，
    再查对应图案决定使用低档还是高档颜色。图案按世界坐标平铺。
    """
    h, w = positions.shape
    safe_family = np.maximum(families, 0)
    max_tier = np.maximum(steps_per_family[safe_family] - 1, 0)
    low = np.floor(positions).astype(np.int16)
    low = np.minimum(low, max_tier)
    high = np.minimum(low + 1, max_tier)
    fraction = positions - low
    levels = np.clip(np.round(fraction * 16).astype(np.int16), 0, 16)

    yy = np.arange(h, dtype=np.int16)[:, None] % 4
    xx = np.arange(w, dtype=np.int16)[None, :] % 4

    use_upper = np.zeros((h, w), dtype=bool)
    for level in range(1, 17):
        pattern = pattern_set[level]
        level_mask = (levels == level) & mask
        use_upper |= level_mask & pattern[yy, xx]

    chosen = np.where(use_upper, high, low)
    chosen[~mask] = np.minimum(
        np.rint(positions[~mask]).astype(np.int16),
        max_tier[~mask],
    )

    bgr = ramps[safe_family, chosen]
    return bgr.astype(np.float32)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def render(
    perceived: dict,
    struct: dict,
    *,
    dither_style: str,
    silhouette_darkness: float,
    internal_darkness: float,
    steps_per_family: NDArray[np.int32],
    debug: bool,
    debug_dir: Path,
) -> np.ndarray:
    """阶段 C：按 (色相族, 明度档) 取色渲染，叠加同色相轮廓。

    dither_style: none | ordered | diagonal | clustered | floyd_steinberg。
    silhouette_darkness / internal_darkness 取值 [0, 1]，0 关闭；
    默认 1.0 复现原轮廓暗化的档位与缩放：
      - silhouette：档位固定最暗档，scale = 1 - 0.25 × strength；
      - internal：steps = round(2 × strength)，scale = 1 - 0.4 × strength。
    perceived 为 :func:`perceive` 的输出字典。
    struct 为 :func:`structure` 的输出字典。
    steps_per_family 为各族原生长度，支持变长 ramp。
    debug 为 False 时跳过调试 PNG 写入；debug_dir 语义不变（定位输出目录）。

    返回 final_bgr（uint8 BGR）。
    """

    def _save(name: str, img: np.ndarray) -> None:
        if not debug:
            return
        if img.dtype == bool:
            img = img.astype(np.uint8) * 255
        cv2.imwrite(str(debug_dir / f"{name}.png"), img)

    ramps = perceived["ramps_bgr"]  # (family_count, max_steps, 3) BGR
    ramp_l = perceived["ramp_l"]
    valid = struct["alpha_down"]
    families = struct["family_down"]
    safe_family = np.maximum(families, 0)
    max_tier = np.maximum(steps_per_family[safe_family] - 1, 0)

    hard_tiers = np.minimum(np.maximum(struct["tier_down"], 0), max_tier).astype(
        np.int16
    )
    hard_bgr = ramps[safe_family, hard_tiers].astype(np.float32)

    position = continuous_ramp_position(
        struct["L_down"], families, ramp_l, valid, steps_per_family
    )
    fraction = position - np.floor(position)
    gradient = neighbor_max_difference(struct["L_down"])

    # 抖动只作用于：非轮廓、非色阶边界、明度处于中间区段且局部确有梯度的像素
    dither_mask = (
        valid
        & ~struct["outline"]
        & ~struct["shade_boundary"]
        & (fraction >= _DITHER_FRACTION_MIN)
        & (fraction <= _DITHER_FRACTION_MAX)
        & (gradient >= _DITHER_GRADIENT_MIN)
    )

    if dither_style in PATTERNS:
        pattern_set = PATTERNS[dither_style]
        dithered_bgr = _render_pattern(
            position, families, ramps, dither_mask, pattern_set, steps_per_family
        )
        final_bgr = hard_bgr.copy()
        final_bgr[dither_mask] = dithered_bgr[dither_mask]
    elif dither_style == "floyd_steinberg":
        dithered_bgr = _render_floyd(
            position, families, ramps, dither_mask, steps_per_family
        )
        final_bgr = hard_bgr.copy()
        final_bgr[dither_mask] = dithered_bgr[dither_mask]
    elif dither_style == "none":
        dithered_bgr = hard_bgr.copy()
        final_bgr = hard_bgr.copy()
        dither_mask[:] = False
    else:
        raise ValueError(f"unknown dither style: {dither_style!r}")

    # 轮廓像素统一压到该色相族的最暗档（同族变暗）；strength=0 时关闭
    if silhouette_darkness > 0:
        silhouette_scale = 1.0 - _SILHOUETTE_SCALE_COEFF * silhouette_darkness
        for y, x in np.argwhere(struct["silhouette"]):
            family = families[y, x]
            if family >= 0:
                final_bgr[y, x] = ramps[family, 0] * silhouette_scale

    # 内部细节线比所在像素的档位再暗 internal_delta 档；strength=0 时关闭
    if internal_darkness > 0:
        internal_delta = round(_INTERNAL_STEPS_MAX * internal_darkness)
        internal_scale = 1.0 - _INTERNAL_SCALE_COEFF * internal_darkness
        for y, x in np.argwhere(struct["internal_detail"]):
            family = families[y, x]
            if family >= 0:
                step = max(0, int(hard_tiers[y, x]) - internal_delta)
                final_bgr[y, x] = ramps[family, step] * internal_scale

    final_bgr[~valid] = 0
    final_u8 = np.clip(final_bgr, 0, 255).astype(np.uint8)
    dithered_u8 = np.clip(dithered_bgr, 0, 255).astype(np.uint8)

    _save("18_hard_ramp", hard_bgr.astype(np.uint8))
    _save("19_dither_mask", dither_mask)
    _save("20_dithered", dithered_u8)
    _save("21_final", final_u8)
    _save("22_palette_strip", make_palette_strip(ramps, steps_per_family))

    return final_u8
