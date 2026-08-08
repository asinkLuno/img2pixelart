"""感知阶段：去噪 → 色块化 → 色相族聚类 → 明度阶梯 → Canny 边缘。

所有可配置参数由 Hydra 配置显式传入，本模块不设代码默认值。
"""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .fit import fit_ramps_to_palette

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
LabelArray = NDArray[np.int16]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _bgr_u8_to_lab_f32(bgr: np.ndarray) -> FloatArray:
    """uint8 BGR [0,255] → float32 CIE Lab (L: 0–100, a,b 以零为中心)。"""
    if bgr.dtype != np.uint8:
        raise TypeError(f"期望 uint8 BGR，实际为 {bgr.dtype}")
    bgr_f32 = bgr.astype(np.float32) / 255.0
    return cv2.cvtColor(bgr_f32, cv2.COLOR_BGR2LAB)


def _pad_to_square(img, side_len):
    h, w = img.shape[:2]
    top = (side_len - h) // 2
    bottom = side_len - h - top
    left = (side_len - w) // 2
    right = side_len - w - left
    return cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )


def _normalize_vectors(vectors: FloatArray) -> FloatArray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, np.float32(1e-6))


def _merge_hue_centers(centers: FloatArray, angular_threshold_degrees: float):
    """合并色相角距离过近的单位向量中心。"""
    if len(centers) <= 1:
        return centers.astype(np.float32, copy=False)
    centers = _normalize_vectors(np.asarray(centers, dtype=np.float32))
    cosine_threshold = np.cos(np.deg2rad(angular_threshold_degrees))
    similarities = centers @ centers.T
    groups: list[list[int]] = []
    for index in range(len(centers)):
        group = next(
            (
                members
                for members in groups
                if np.all(similarities[index, members] >= cosine_threshold)
            ),
            None,
        )
        if group is None:
            groups.append([index])
        else:
            group.append(index)

    merged: list[FloatArray] = []
    for group in groups:
        mean_direction = centers[group].mean(axis=0, keepdims=True)
        merged.append(_normalize_vectors(mean_direction)[0])

    return np.asarray(merged, dtype=np.float32)


def _assign_nearest_hue(
    ab: FloatArray,
    centers: FloatArray,
) -> LabelArray:
    """把每个 a/b 向量分配给色相角最近的中心。"""
    directions = _normalize_vectors(ab)
    similarities = directions @ centers.T
    return np.argmax(similarities, axis=-1).astype(np.int16)


def spherical_kmeans_2d(
    directions: FloatArray,
    cluster_count: int,
    iterations: int,
    random_seed: int,
) -> tuple[LabelArray, FloatArray]:
    """在单位圆上对二维方向向量做 spherical K-Means 聚类。

    返回 (labels, centers)。
    """
    if directions.ndim != 2 or directions.shape[1] != 2:
        raise ValueError("directions 必须具有形状 (N, 2)")
    if cluster_count < 1 or cluster_count > len(directions):
        raise ValueError("cluster_count 必须位于 [1, N]")

    directions = _normalize_vectors(np.asarray(directions, dtype=np.float32))
    rng = np.random.default_rng(random_seed)
    initial_indices = rng.choice(len(directions), size=cluster_count, replace=False)
    centers = directions[initial_indices].copy()
    labels = np.full(len(directions), -1, dtype=np.int16)

    for _ in range(iterations):
        similarities = directions @ centers.T
        new_labels = np.argmax(similarities, axis=1).astype(np.int16)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        new_centers = np.empty_like(centers)
        for cluster_index in range(cluster_count):
            members = directions[labels == cluster_index]
            if len(members) == 0:
                # 用与现有中心最不相似的点重建空簇。
                nearest_similarity = np.max(directions @ centers.T, axis=1)
                replacement_index = int(np.argmin(nearest_similarity))
                new_centers[cluster_index] = directions[replacement_index]
                continue

            resultant = members.sum(axis=0)
            norm = float(np.linalg.norm(resultant))
            if norm <= 1e-6:
                replacement_index = int(rng.integers(len(directions)))
                new_centers[cluster_index] = directions[replacement_index]
            else:
                new_centers[cluster_index] = resultant / norm
        centers = new_centers

    return labels, centers.astype(np.float32)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def cluster_hue_families(
    lab: FloatArray,
    foreground: BoolArray,
    requested_groups: int,
    chroma_floor: float,
    merge_angle_degrees: float,
    minimum_lightness: float,
    minimum_fit_pixels: int,
    maximum_fit_pixels: int,
    random_seed: int,
    spherical_kmeans_iterations: int,
):
    """按色相角将前景像素聚成色相族，低色度像素归入中性族。

    返回 (labels, hue_directions_ab)。

    labels:
        -1 = 背景
        0..C-1 = 彩色族
        C = 中性族（色度不足的前景像素）

    hue_directions_ab:
        单位色相方向 (C+1, 2)，最后一行是中性族 ``[0, 0]``。
        若无前景则返回 ``(0, 2)`` 空数组。
    """
    labels = np.full(foreground.shape, -1, dtype=np.int16)
    visible_lab = lab[foreground]

    if len(visible_lab) == 0:
        return labels, np.empty((0, 2), dtype=np.float32)

    visible_ab = visible_lab[:, 1:3]
    visible_chroma = np.linalg.norm(visible_ab, axis=1)

    # ── 彩色像素聚类 ──
    chromatic_fit = (visible_chroma >= chroma_floor) & (
        visible_lab[:, 0] >= minimum_lightness
    )
    fit_ab = visible_ab[chromatic_fit]

    chromatic_centers: FloatArray | None = None

    if len(fit_ab) >= minimum_fit_pixels:
        if len(fit_ab) > maximum_fit_pixels:
            rng = np.random.default_rng(random_seed)
            indices = rng.choice(len(fit_ab), size=maximum_fit_pixels, replace=False)
            fit_ab = fit_ab[indices]

        fit_directions = _normalize_vectors(fit_ab)
        cluster_count = min(requested_groups, len(fit_directions))

        if cluster_count == 1:
            chromatic_centers = _normalize_vectors(
                fit_directions.mean(axis=0, keepdims=True)
            )
        else:
            _, centers = spherical_kmeans_2d(
                fit_directions,
                cluster_count=cluster_count,
                iterations=spherical_kmeans_iterations,
                random_seed=random_seed,
            )
            chromatic_centers = centers
            chromatic_centers = _merge_hue_centers(
                chromatic_centers,
                angular_threshold_degrees=merge_angle_degrees,
            )

    # ── 建立完整色相方向表：彩色族 + 中性族 ──
    if chromatic_centers is not None and len(chromatic_centers) > 0:
        chromatic_count = len(chromatic_centers)
        hue_directions = np.zeros((chromatic_count + 1, 2), dtype=np.float32)
        hue_directions[:chromatic_count] = chromatic_centers
        neutral_index = chromatic_count
    else:
        # 纯灰度图：只有中性族。
        hue_directions = np.zeros((1, 2), dtype=np.float32)
        chromatic_count = 0
        neutral_index = 0

    # ── 分配彩色族标签 ──
    if chromatic_count > 0:
        chromatic_mask = foreground & (
            np.linalg.norm(lab[..., 1:3], axis=-1) >= chroma_floor
        )
        labels[chromatic_mask] = _assign_nearest_hue(
            lab[chromatic_mask, 1:3], hue_directions[:chromatic_count]
        )

    # ── 分配中性族标签 ──
    neutral_mask = foreground & (labels == -1)
    labels[neutral_mask] = neutral_index

    return labels, hue_directions


def _enforce_span(
    low: float,
    high: float,
    center: float,
    minimum_span: float,
) -> tuple[float, float]:
    """保证明度区间至少具有指定跨度，并限制在 Lab 的合法明度范围内。"""
    low = float(np.clip(low, 0.0, 100.0))
    high = float(np.clip(high, 0.0, 100.0))
    center = float(np.clip(center, 0.0, 100.0))

    if high < low:
        low, high = high, low

    current_span = high - low
    if current_span >= minimum_span:
        return low, high

    half_span = minimum_span / 2.0
    low = center - half_span
    high = center + half_span

    if low < 0.0:
        high -= low
        low = 0.0

    if high > 100.0:
        low -= high - 100.0
        high = 100.0

    return max(0.0, low), min(100.0, high)


def build_adaptive_ramps(
    lab: FloatArray,
    foreground: BoolArray,
    family_labels: LabelArray,
    hue_directions_ab: FloatArray,
    steps: int,
    minimum_span: float,
    low_quantile: float,
    high_quantile: float,
    minimum_family_pixels: int,
    chroma_quantile: float,
    endpoint_chroma_scale: float,
    maximum_chroma: float,
) -> tuple[FloatArray, FloatArray]:
    """为每个色相族（含中性族）构建明度阶梯色板。

    ``hue_directions_ab`` 是单位色相方向 (family_count, 2)；中性族为 ``[0, 0]``。

    返回：
        ramps_bgr: ``(family_count, steps, 3)`` BGR 色板，范围 ``[0, 255]``。
        ramp_l: ``(family_count, steps)`` Lab 明度值。
    """
    lab = np.asarray(lab, dtype=np.float32)
    foreground = np.asarray(foreground, dtype=bool)
    family_labels = np.asarray(family_labels)
    hue_directions_ab = np.asarray(hue_directions_ab, dtype=np.float32)

    if lab.ndim != 3 or lab.shape[-1] != 3:
        raise ValueError(f"lab 必须具有形状 (H, W, 3)，实际为 {lab.shape}")

    if foreground.shape != lab.shape[:2]:
        raise ValueError("foreground 的形状必须与 lab 的前两个维度一致")

    if family_labels.shape != lab.shape[:2]:
        raise ValueError("family_labels 的形状必须与 lab 的前两个维度一致")

    if hue_directions_ab.ndim != 2 or hue_directions_ab.shape[1] != 2:
        raise ValueError("hue_directions_ab 必须具有形状 (family_count, 2)")

    if steps < 3:
        raise ValueError("steps 必须至少为 3")

    if minimum_span < 0.0 or minimum_span > 100.0:
        raise ValueError("minimum_span 必须位于 [0, 100]")

    if not 0.0 <= low_quantile < high_quantile <= 1.0:
        raise ValueError("必须满足 0 <= low_quantile < high_quantile <= 1")

    if minimum_family_pixels < 1:
        raise ValueError("minimum_family_pixels 必须至少为 1")

    if not 0.0 <= chroma_quantile <= 1.0:
        raise ValueError("chroma_quantile 必须位于 [0, 1]")

    if not 0.0 <= endpoint_chroma_scale <= 1.0:
        raise ValueError("endpoint_chroma_scale 必须位于 [0, 1]")

    if maximum_chroma <= 0.0:
        raise ValueError("maximum_chroma 必须大于 0")

    family_count = len(hue_directions_ab)
    if family_count == 0:
        return (
            np.zeros((0, steps, 3), dtype=np.float32),
            np.zeros((0, steps), dtype=np.float32),
        )

    foreground_lab = lab[foreground]
    foreground_labels = family_labels[foreground]

    if len(foreground_lab) == 0:
        return (
            np.zeros((family_count, steps, 3), dtype=np.float32),
            np.zeros((family_count, steps), dtype=np.float32),
        )

    all_chroma = np.linalg.norm(foreground_lab[:, 1:3], axis=1)

    global_chroma = float(np.quantile(all_chroma, chroma_quantile))
    global_chroma = min(global_chroma, maximum_chroma)

    # 中间档保持原色度，两端平滑衰减到 endpoint_chroma_scale。
    # ponytail: 偶数 steps 时没有恰好为 1.0 的中间档，只有奇数 steps 才精确。
    positions = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    chroma_scale = (1.0 - (1.0 - endpoint_chroma_scale) * np.abs(positions)).astype(
        np.float32
    )

    ramp_labs = np.empty((family_count, steps, 3), dtype=np.float32)
    ramp_l = np.empty((family_count, steps), dtype=np.float32)

    for family_index in range(family_count):
        family_mask = foreground_labels == family_index
        family_lab = foreground_lab[family_mask]

        if len(family_lab) >= minimum_family_pixels:
            source_lab = family_lab
        else:
            source_lab = foreground_lab

        lightness = source_lab[:, 0]

        low = float(np.quantile(lightness, low_quantile))
        high = float(np.quantile(lightness, high_quantile))
        center_l = float(np.median(lightness))

        low, high = _enforce_span(
            low=low,
            high=high,
            center=center_l,
            minimum_span=minimum_span,
        )

        levels = np.linspace(low, high, steps, dtype=np.float32)
        ramp_l[family_index] = levels

        direction = hue_directions_ab[family_index]
        direction_norm = float(np.linalg.norm(direction))

        if direction_norm <= 1e-6:
            # 中性族：无色度，纯明度阶梯。
            base_chroma = 0.0
            direction = np.zeros(2, dtype=np.float32)
        else:
            direction = direction / direction_norm

            if len(family_lab) >= minimum_family_pixels:
                family_chroma = np.linalg.norm(family_lab[:, 1:3], axis=1)
                base_chroma = float(np.quantile(family_chroma, chroma_quantile))
            else:
                base_chroma = global_chroma

            base_chroma = min(base_chroma, maximum_chroma)

        ramp_chroma = base_chroma * chroma_scale

        ramp_labs[family_index, :, 0] = levels
        ramp_labs[family_index, :, 1:3] = ramp_chroma[:, None] * direction[None, :]

    # 一次性转换所有色板，避免每个 family 单独调用 cvtColor。
    flat_lab = np.ascontiguousarray(
        ramp_labs.reshape(-1, 1, 3),
        dtype=np.float32,
    )
    flat_bgr = cv2.cvtColor(flat_lab, cv2.COLOR_LAB2BGR)
    ramps_bgr = np.clip(flat_bgr.reshape(family_count, steps, 3), 0.0, 1.0) * 255.0

    return ramps_bgr.astype(np.float32), ramp_l


def assign_lightness_tiers(
    l_channel: np.ndarray,
    family_labels: np.ndarray,
    ramp_l: np.ndarray,
    foreground: np.ndarray,
) -> np.ndarray:
    """把每个前景像素的明度就近量化到其色相族阶梯的某一档（tier）。

    返回的 tiers：前景为 ``0 .. steps-1``，背景为 ``-1``。
    """
    tiers = np.full(l_channel.shape, -1, dtype=np.int16)
    for group in range(len(ramp_l)):
        mask = foreground & (family_labels == group)
        if not mask.any():
            continue
        values = l_channel[mask, None]
        tiers[mask] = np.argmin(np.abs(values - ramp_l[group][None, :]), axis=1)
    return tiers


def perceive(
    bgra: np.ndarray,
    denoise_d: int,
    denoise_sigma: float,
    mean_shift_sp: float,
    mean_shift_sr: float,
    requested_groups: int,
    chroma_floor: float,
    merge_angle_degrees: float,
    minimum_lightness: float,
    minimum_fit_pixels: int,
    maximum_fit_pixels: int,
    random_seed: int,
    spherical_kmeans_iterations: int,
    ramp_steps: int,
    ramp_minimum_span: float,
    ramp_low_quantile: float,
    ramp_high_quantile: float,
    ramp_minimum_family_pixels: int,
    ramp_chroma_quantile: float,
    ramp_endpoint_chroma_scale: float,
    ramp_maximum_chroma: float,
    canny_low: int,
    canny_high: int,
    alpha_threshold: int,
    palette_bgr: NDArray[np.uint8] | None,
    debug_dir: Path,
):
    """感知阶段：去噪 → 色块化 → 色相族聚类 → 明度阶梯 → Canny 细节线。

    所有参数由 Hydra 配置显式传入。
    debug_dir 非空时，每步结果即时保存为 PNG。
    """

    def _save(name: str, img: np.ndarray) -> None:
        if img.dtype == bool:
            img = img.astype(np.uint8) * 255
        cv2.imwrite(str(debug_dir / f"{name}.png"), img)

    # ── 输入校验 ──
    if bgra.ndim != 3 or bgra.shape[2] not in (3, 4):
        raise ValueError(f"bgra 必须为 (H, W, 3) 或 (H, W, 4)，实际 shape={bgra.shape}")
    if bgra.size == 0:
        raise ValueError("输入图像为空")
    if bgra.dtype != np.uint8:
        raise TypeError(f"bgra 必须为 uint8，实际为 {bgra.dtype}")

    # ── alpha / 前景 ──
    if bgra.shape[2] == 4:
        source_alpha = bgra[..., 3]
        foreground = source_alpha >= alpha_threshold
        alpha_source = source_alpha.copy()
        bgr = bgra[..., :3].copy()
        # 填充透明区域，防止其中残留的 RGB 在滤波时污染前景边缘。
        # ponytail: cv2.inpaint 对大面积透明效果有限，严重时改用 alpha 加权统计。
        transparent = ~foreground
        if transparent.any() and foreground.any():
            bgr = cv2.inpaint(bgr, transparent.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    else:
        bgr = bgra.copy()
        foreground = np.ones(bgr.shape[:2], dtype=bool)
        alpha_source = np.full(bgr.shape[:2], 255, dtype=np.uint8)

    # ── 去噪 + 色块化 ──
    denoised = cv2.bilateralFilter(bgr, denoise_d, denoise_sigma, denoise_sigma)
    _save("01_original", bgr)
    _save("02_denoised", denoised)

    blocks = cv2.pyrMeanShiftFiltering(denoised, mean_shift_sp, mean_shift_sr)
    _save("03_blocks", blocks)

    # ── 补正方形（记录偏移和原始尺寸） ──
    original_h, original_w = bgr.shape[:2]
    side_len = max(original_h, original_w)
    top = (side_len - original_h) // 2
    left = (side_len - original_w) // 2

    bgr = _pad_to_square(bgr, side_len)
    denoised = _pad_to_square(denoised, side_len)
    blocks = _pad_to_square(blocks, side_len)
    foreground = _pad_to_square(foreground.astype(np.uint8), side_len).astype(bool)
    alpha_full = _pad_to_square(alpha_source, side_len)

    # ── uint8 BGR → float32 CIE Lab ──
    blocks_lab = _bgr_u8_to_lab_f32(blocks)

    l_smooth = blocks_lab[..., 0]

    family_labels, hue_directions_ab = cluster_hue_families(
        blocks_lab,
        foreground,
        requested_groups=requested_groups,
        chroma_floor=chroma_floor,
        merge_angle_degrees=merge_angle_degrees,
        minimum_lightness=minimum_lightness,
        minimum_fit_pixels=minimum_fit_pixels,
        maximum_fit_pixels=maximum_fit_pixels,
        random_seed=random_seed,
        spherical_kmeans_iterations=spherical_kmeans_iterations,
    )
    ramps_bgr, ramp_l = build_adaptive_ramps(
        lab=blocks_lab,
        foreground=foreground,
        family_labels=family_labels,
        hue_directions_ab=hue_directions_ab,
        steps=ramp_steps,
        minimum_span=ramp_minimum_span,
        low_quantile=ramp_low_quantile,
        high_quantile=ramp_high_quantile,
        minimum_family_pixels=ramp_minimum_family_pixels,
        chroma_quantile=ramp_chroma_quantile,
        endpoint_chroma_scale=ramp_endpoint_chroma_scale,
        maximum_chroma=ramp_maximum_chroma,
    )
    if palette_bgr is not None:
        ramps_bgr = fit_ramps_to_palette(ramps_bgr, palette_bgr)
    tier = assign_lightness_tiers(l_smooth, family_labels, ramp_l, foreground)
    steps_per_family = np.full(len(hue_directions_ab), ramp_steps, dtype=np.int32)

    # families 可视化
    fam_viz = np.zeros((*family_labels.shape, 3), dtype=np.uint8)
    fam_colors = [
        (255, 0, 0),
        (0, 150, 255),
        (128, 128, 128),
        (255, 200, 0),
        (200, 0, 255),
        (0, 200, 100),
    ]
    for fi in range(len(hue_directions_ab)):
        fam_viz[family_labels == fi] = fam_colors[fi % len(fam_colors)]
    _save("04_families", fam_viz)

    # palette
    ramps_u8 = ramps_bgr.astype(np.uint8)
    family_count, max_steps = ramps_u8.shape[:2]
    bh, bw = 24, 48
    pal = np.zeros((family_count * bh, max_steps * bw, 3), dtype=np.uint8)
    for fi, steps in enumerate(steps_per_family):
        for si in range(steps):
            pal[fi * bh : (fi + 1) * bh, si * bw : (si + 1) * bw] = ramps_u8[fi, si]
    _save("05_palette", pal)

    # ── Canny 细节线（排除 alpha 轮廓边界带） ──
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, canny_low, canny_high)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(foreground.astype(np.uint8), kernel).astype(bool)
    canny[~eroded] = 0
    _save("06_canny", canny)

    # reconstructed
    h, w = tier.shape
    recon = np.zeros((h, w, 3), dtype=np.uint8)
    for fi, steps in enumerate(steps_per_family):
        mask = (family_labels == fi) & (tier >= 0)
        if not mask.any():
            continue
        for si in range(steps):
            recon[mask & (tier == si)] = ramps_u8[fi, si]
    recon[canny > 0] = [0, 0, 0]
    _save("07_reconstructed", recon)

    return {
        "original": bgr,
        "denoised": denoised,
        "blocks": blocks,
        "blocks_lab": blocks_lab,
        "L": l_smooth,
        "tier": tier,
        "family_labels": family_labels,
        "hue_directions_ab": hue_directions_ab,
        "ramps_bgr": ramps_bgr,
        "ramp_l": ramp_l,
        "steps_per_family": steps_per_family,
        "foreground": foreground,
        "alpha_full": alpha_full,
        "canny": canny,
        "original_shape": (original_h, original_w),
        "pad_offset": (top, left),
    }
