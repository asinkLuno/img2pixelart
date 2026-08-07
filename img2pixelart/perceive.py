from numpy.char import center
import cv2
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
LabelArray = NDArray[np.int16]


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
    """
    合并色相角距离过近的单位向量中心
    """
    if len(centers) <= 1:
        return centers.astype(np.float32, copy=False)
    centers = _normalize_vectors(np.asarray(centers, dtype=np.float32))
    cosine_threshold = np.cos(np.deg2rad(angular_threshold_degrees))
    similarities = centers @ centers.T
    adjacency = similarities >= cosine_threshold

    visited = np.zeros(len(centers), dtype=bool)
    merged: list[FloatArray] = []

    for start in range(len(centers)):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        component: list[int] = []

        while stack:
            current = stack.pop()
            component.append(current)

            neighbors = np.flatnonzero(adjacency[current] & ~visited)
            visited[neighbors] = True
            stack.extend(neighbors.tolist())

        mean_direction = centers[component].mean(axis=0, keepdims=True)
        merged.append(_normalize_vectors(mean_direction)[0])

    return np.asarray(merged, dtype=np.float32)


def _assign_nearest_hue(
    ab: FloatArray,
    centers: FloatArray,
) -> LabelArray:
    """把每个 a/b 向量分配给色相角最近的中心。"""
    directions = _normalize_vectors(ab)

    # 单位向量的点积越大，夹角越小。
    similarities = directions @ centers.T
    return np.argmax(similarities, axis=-1).astype(np.int16)


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
):
    """
    按色相角将前景像素聚成若干色相族
    """
    labels = np.full(foreground.shape, -1, dtype=np.int16)
    visible_lab = lab[foreground]

    if len(visible_lab) == 0:
        return labels, np.empty((0, 2), dtype=np.float32)

    visible_ab = visible_lab[:, 1:3]
    visible_chroma = np.linalg.norm(visible_ab, axis=1)

    fit_mask = (visible_chroma >= chroma_floor) & (
        visible_lab[:, 0] >= minimum_lightness
    )
    fit_ab = visible_ab[fit_mask]

    # 没有足够的有效彩色像素时，不应拿灰色像素强行拟合色相。
    if len(fit_ab) < minimum_fit_pixels:
        return labels, np.empty((0, 2), dtype=np.float32)

    if len(fit_ab) > maximum_fit_pixels:
        rng = np.random.default_rng(random_seed)
        indices = rng.choice(
            len(fit_ab),
            size=maximum_fit_pixels,
            replace=False,
        )
        fit_ab = fit_ab[indices]

    fit_directions = _normalize_vectors(fit_ab)
    cluster_count = min(requested_groups, len(fit_directions))

    if cluster_count == 1:
        centers = _normalize_vectors(fit_directions.mean(axis=0, keepdims=True))
    else:
        # OpenCV 的全局 RNG 控制 k-means 初始化。
        cv2.setRNGSeed(random_seed)

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            50,
            1e-4,
        )

        _, _, centers = cv2.kmeans(
            data=np.ascontiguousarray(fit_directions, dtype=np.float32),
            K=cluster_count,
            bestLabels=None,
            criteria=criteria,
            attempts=6,
            flags=cv2.KMEANS_PP_CENTERS,
        )

        centers = _normalize_vectors(centers)
        centers = _merge_hue_centers(
            centers,
            angular_threshold_degrees=merge_angle_degrees,
        )

    # 只给色度足够高的前景像素分配色相标签。
    chromatic_mask = foreground & (
        np.linalg.norm(lab[..., 1:3], axis=-1) >= chroma_floor
    )

    labels[chromatic_mask] = _assign_nearest_hue(
        lab[chromatic_mask, 1:3],
        centers,
    )

    return labels, centers


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


def _lab_to_bgr(lab: FloatArray) -> FloatArray:
    """将标准 float32 CIE Lab 转换为 OpenCV BGR，输出范围为 [0, 255]。"""
    lab = np.ascontiguousarray(lab, dtype=np.float32)
    bgr = cv2.cvtColor(lab[None, ...], cv2.COLOR_LAB2BGR)[0]
    return np.clip(bgr * 255.0, 0.0, 255.0).astype(np.float32)


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
    """为每个色相族构建具有稳定色相的自适应明度阶梯。

    `hue_directions_ab` 应为 a/b 平面上的单位色相方向，而不是实际的
    Lab a/b 中心。每个族的实际色度从该族原始像素中估计。

    返回：
        ramps_bgr:
            形状为 ``(family_count, steps, 3)`` 的 OpenCV BGR 色板，
            数值范围为 ``[0, 255]``。
        ramp_l:
            形状为 ``(family_count, steps)`` 的 Lab 明度值。
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
    ramps_bgr = np.empty((family_count, steps, 3), dtype=np.float32)
    ramp_l = np.empty((family_count, steps), dtype=np.float32)

    if family_count == 0:
        return ramps_bgr, ramp_l

    foreground_lab = lab[foreground]
    foreground_labels = family_labels[foreground]

    if len(foreground_lab) == 0:
        return (
            np.empty((family_count, steps, 3), dtype=np.float32),
            np.empty((family_count, steps), dtype=np.float32),
        )

    all_lightness = foreground_lab[:, 0]
    all_chroma = np.linalg.norm(foreground_lab[:, 1:3], axis=1)

    global_center_l = float(np.median(all_lightness))
    global_chroma = float(np.quantile(all_chroma, chroma_quantile))
    global_chroma = min(global_chroma, maximum_chroma)

    positions = np.linspace(-1.0, 1.0, steps, dtype=np.float32)

    # 中间档保持原色度，两端平滑衰减到 endpoint_chroma_scale。
    chroma_scale = (1.0 - (1.0 - endpoint_chroma_scale) * np.abs(positions)).astype(
        np.float32
    )

    ramp_labs = np.empty((family_count, steps, 3), dtype=np.float32)

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

        if len(family_lab) >= minimum_family_pixels:
            family_chroma = np.linalg.norm(family_lab[:, 1:3], axis=1)
            base_chroma = float(np.quantile(family_chroma, chroma_quantile))
        else:
            base_chroma = global_chroma

        base_chroma = min(base_chroma, maximum_chroma)

        direction = hue_directions_ab[family_index]
        direction_norm = float(np.linalg.norm(direction))

        if direction_norm <= 1e-6:
            direction = np.zeros(2, dtype=np.float32)
            base_chroma = 0.0
        else:
            direction = direction / direction_norm

        ramp_chroma = base_chroma * chroma_scale

        ramp_labs[family_index, :, 0] = levels
        ramp_labs[family_index, :, 1:3] = ramp_chroma[:, None] * direction[None, :]

    # 一次性转换所有色板，避免每个 family 单独调用 cvtColor。
    flat_lab = np.ascontiguousarray(
        ramp_labs.reshape(-1, 1, 3),
        dtype=np.float32,
    )
    flat_bgr = cv2.cvtColor(flat_lab, cv2.COLOR_LAB2BGR)
    ramps_bgr[:] = np.clip(flat_bgr.reshape(family_count, steps, 3), 0.0, 1.0) * 255.0

    return ramps_bgr.astype(np.float32), ramp_l


def assign_lightness_tiers(
    l_channel: np.ndarray,
    family_labels: np.ndarray,
    ramp_l: np.ndarray,
    foreground: np.ndarray,
) -> np.ndarray:
    """把每个前景像素的实际明度就近量化到其色相族阶梯的某一档（tier）。"""
    tiers = np.zeros(l_channel.shape, dtype=np.int16)
    for group in range(len(ramp_l)):
        mask = foreground & (family_labels == group)
        if not mask.any():
            continue
        values = l_channel[mask, None]
        tiers[mask] = np.argmin(np.abs(values - ramp_l[group][None, :]), axis=1)
    tiers[~foreground] = 0
    return tiers


def perceive(
    bgra: np.ndarray,
    denoise_d,
    denoise_sigma,
    mean_shift_sp,
    mean_shift_sr,
    requested_groups,
    chroma_floor,
    merge_angle_degrees,
    minimum_lightness,
    minimum_fit_pixels,
    maximum_fit_pixels,
    random_seed,
    ramp_steps,
    ramp_minimum_span,
    ramp_low_quantile,
    ramp_high_quantile,
    ramp_minimum_family_pixels,
    ramp_chroma_quantile,
    ramp_endpoint_chroma_scale,
    ramp_maximum_chroma,
    canny_low,
    canny_high,
):
    has_alpha = bgra.ndim == 3 and bgra.shape[-1] == 4
    bgr = bgra[..., :3].copy() if has_alpha else bgra.copy()
    source_alpha = bgra[..., 3] if has_alpha else None

    if source_alpha is not None:
        foreground = source_alpha > 0
        alpha_source = source_alpha.copy()
    else:
        foreground = np.ones(bgr.shape[:2], dtype=bool)
        alpha_source = np.full(bgr.shape[:2], 255, dtype=np.uint8)

    denoised = cv2.bilateralFilter(bgr, denoise_d, denoise_sigma, denoise_sigma)
    blocks = cv2.pyrMeanShiftFiltering(denoised, mean_shift_sp, mean_shift_sr)

    side_len = max(bgr.shape[0], bgr.shape[1])
    bgr = _pad_to_square(bgr, side_len)
    denoised = _pad_to_square(denoised, side_len)
    blocks = _pad_to_square(blocks, side_len)

    foreground = _pad_to_square(foreground.astype(np.uint8), side_len).astype(bool)
    alpha_full = _pad_to_square(alpha_source, side_len)

    blocks_lab = cv2.cvtColor(blocks, cv2.COLOR_BGR2LAB).astype(np.float32)

    l_smooth = blocks_lab[..., 0]
    # 在色块化的 Lab 图像上聚类色相族、建明度阶梯、分配明度层级
    family_labels, centers_ab = cluster_hue_families(
        blocks_lab,
        foreground,
        requested_groups=requested_groups,
        chroma_floor=chroma_floor,
        merge_angle_degrees=merge_angle_degrees,
        minimum_lightness=minimum_lightness,
        minimum_fit_pixels=minimum_fit_pixels,
        maximum_fit_pixels=maximum_fit_pixels,
        random_seed=random_seed,
    )

    ramps_bgr, ramp_l = build_adaptive_ramps(
        lab=blocks_lab,
        foreground=foreground,
        family_labels=family_labels,
        hue_directions_ab=centers_ab,
        steps=ramp_steps,
        minimum_span=ramp_minimum_span,
        low_quantile=ramp_low_quantile,
        high_quantile=ramp_high_quantile,
        minimum_family_pixels=ramp_minimum_family_pixels,
        chroma_quantile=ramp_chroma_quantile,
        endpoint_chroma_scale=ramp_endpoint_chroma_scale,
        maximum_chroma=ramp_maximum_chroma,
    )

    tier = assign_lightness_tiers(l_smooth, family_labels, ramp_l, foreground)

    # 对去噪后的灰度图提取 Canny 边缘，作为内部细节线的候选来源
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, canny_low, canny_high)
    canny[~foreground] = 0

    return {
        "original": bgr,
        "denoised": denoised,
        "blocks": blocks,
        "blocks_lab": blocks_lab,
        "L": l_smooth,
        "tier": tier,
        "family_labels": family_labels,
        "family_centers_ab": centers_ab,
        "ramps_bgr": ramps_bgr,
        "ramp_l": ramp_l,
        "foreground": foreground,
        "alpha_full": alpha_full,
        "canny": canny,
    }
