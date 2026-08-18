"""图片转 ASCII 字符画：边缘提取 → 字符格分类 → 线段合并。

移植自 gEInk 的 art/ascii.py，并复用本仓 structure 阶段的算法：
骨架细化用 :func:`img2pixelart.structure.thin`（替代 skimage 骨架化），
主体遮罩用源图 alpha 通道降采样（替代 gEInk 的 SAM 分割）。
所有可配置参数由 Hydra 配置显式传入，本模块不设代码层默认值。
"""

from pathlib import Path

import cv2
import numpy as np

from .structure import down_mask, thin

# 半角等宽字符栅格比例 2:1（终端字符几何常量，非调参项）
CELL_H = 16
CELL_W = 8

# direction index → chars at intensity 0..4
# 0: horizontal, 1: vertical, 2: right diagonal (/), 3: left diagonal (\)
_EDGE_CHARS_TABLE = [
    [" ", "╌", "─", "━", "▀"],
    [" ", "╎", "│", "┃", "▌"],
    [" ", "⋰", "╱", "/", "◢"],
    [" ", "⋱", "╲", "\\", "◣"],
]

_H_CHARS = frozenset("╌─━▀")
_V_CHARS = frozenset("╎│┃▌")
_D2_CHARS = frozenset("⋰╱/◢")
_D1_CHARS = frozenset("⋱╲\\◣")
_ALL_EDGE = _H_CHARS | _V_CHARS | _D1_CHARS | _D2_CHARS


# ---------------------------------------------------------------------------
# 边缘提取
# ---------------------------------------------------------------------------


def _is_line_art(
    img_bgr: np.ndarray, white_ratio: float, max_saturation: float
) -> bool:
    """判断是否线条画：亮背景占比高且整体低饱和。"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    bright_ratio = float((hsv[:, :, 2] > 200).mean())
    sat_mean = float(hsv[:, :, 1].mean())
    return bright_ratio > white_ratio and sat_mean < max_saturation


def _line_art_edges(gray: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """线条画边缘：Otsu 二值化暗笔画后细化成 1px 骨架。"""
    # Otsu 把亮背景上的暗笔画二值化；反转使笔画 = 255
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if binary.shape != (target_h, target_w):
        binary = cv2.resize(
            binary, (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )

    return thin(binary > 0).astype(np.uint8) * 255


def _photo_edges(
    bgr: np.ndarray,
    foreground: np.ndarray,
    target_h: int,
    target_w: int,
    bilateral_d: int,
    bilateral_sigma_color: float,
    bilateral_sigma_space: float,
    canny_low_ratio: float,
    canny_low_floor: float,
    color_edge_quantile: float,
) -> np.ndarray:
    """照片边缘：亮度 Canny 与 Lab 色彩梯度合并后细化。

    亮度 Canny 保留外轮廓和强明暗边界；Lab 三通道梯度专门补充相近亮度、
    不同颜色或色阶形成的内部轮廓。色彩梯度只在主体前景内自适应阈值化，
    不会把透明背景或整张图的低频变化当作细节。
    """
    smooth = cv2.bilateralFilter(
        bgr, bilateral_d, bilateral_sigma_color, bilateral_sigma_space
    )
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    luma_edges = cv2.Canny(
        gray,
        max(float(otsu) * canny_low_ratio, canny_low_floor),
        float(otsu),
    )

    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    grad_x = cv2.Sobel(lab, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(lab, cv2.CV_32F, 0, 1, ksize=3)
    color_magnitude = np.sqrt(np.sum(grad_x**2 + grad_y**2, axis=2))
    scale = float(np.quantile(color_magnitude[foreground], color_edge_quantile))
    if scale > 0:
        normalized_color = np.clip(color_magnitude * 255.0 / scale, 0, 255).astype(
            np.uint8
        )
        color_threshold, _ = cv2.threshold(
            normalized_color[foreground].reshape(-1, 1),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        color_edges = (normalized_color > color_threshold) & foreground
    else:
        color_edges = np.zeros_like(foreground)

    edges = (luma_edges > 0) | color_edges
    if edges.shape != (target_h, target_w):
        edges = (
            cv2.resize(
                edges.astype(np.uint8),
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )

    return thin(edges).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# 字符格分类
# ---------------------------------------------------------------------------


def _density_to_intensity(density: float, breaks: list[float]) -> int:
    """边缘像素密度 → 强度档位（0..len(breaks)）。"""
    for i, threshold in enumerate(breaks):
        if density < threshold:
            return i
    return len(breaks)


def _circular_mean_angle(angles_deg: np.ndarray) -> tuple[float, float]:
    """返回 [0°, 180°) 角度的 (均值角度, 一致性)。

    用倍角技巧处理边缘方向的 180° 周期性；
    consistency ∈ [0, 1]：1 = 完全同向，0 = 均匀散布。
    """
    a2 = np.radians(angles_deg * 2.0)
    mc, ms = float(np.cos(a2).mean()), float(np.sin(a2).mean())
    mean = float(np.degrees(np.arctan2(ms, mc))) / 2.0 % 180.0
    consistency = float(np.sqrt(mc**2 + ms**2))
    return mean, consistency


def _classify_edge_cell(
    cell_gx: np.ndarray,
    cell_gy: np.ndarray,
    edge_density: float,
    breaks: list[float],
) -> str:
    """把单个字符格的梯度数据与边缘密度映射为 _EDGE_CHARS_TABLE 中的一个字符。"""
    intensity = _density_to_intensity(edge_density, breaks)
    if intensity == 0:
        return " "

    mag = np.hypot(cell_gx, cell_gy)
    peak = float(mag.max())
    if peak == 0:
        return " "

    mask = mag > peak * 0.3
    if not mask.any():
        return " "

    edge_angles = (np.degrees(np.arctan2(cell_gy, cell_gx)) + 90.0) % 180.0
    mean_angle, _ = _circular_mean_angle(edge_angles[mask])

    if mean_angle < 22.5 or mean_angle > 157.5:
        direction = 0  # horizontal
    elif mean_angle < 67.5:
        direction = 3  # left diagonal \
    elif mean_angle < 112.5:
        direction = 1  # vertical
    else:
        direction = 2  # right diagonal /

    return _EDGE_CHARS_TABLE[direction][intensity]


# ---------------------------------------------------------------------------
# 线段合并
# ---------------------------------------------------------------------------


def _merge_edge_segments(rows: list[str], max_gap: int) -> list[str]:
    """填补同方向边缘字符间的小空隙，并删除孤立噪点。

    对每个方向（横 / 竖 / 斜）扫描序列，把两个同类字符段之间长度
    <= max_gap 的纯空格段填补；最后删除没有任何非空 8 连通邻居的字符。
    """
    if not rows:
        return rows

    nrows = len(rows)
    ncols = max(len(r) for r in rows)
    grid = [list(r.ljust(ncols)) for r in rows]

    def _fill_seq(seq: list[str], char_set: frozenset[str]) -> None:
        n = len(seq)
        runs: list[tuple[int, int]] = []
        i = 0
        while i < n:
            if seq[i] in char_set:
                s = i
                while i < n and seq[i] in char_set:
                    i += 1
                runs.append((s, i))
            else:
                i += 1
        for k in range(len(runs) - 1):
            gap_s, gap_e = runs[k][1], runs[k + 1][0]
            if 0 < gap_e - gap_s <= max_gap and all(
                seq[j] == " " for j in range(gap_s, gap_e)
            ):
                fill = seq[gap_s - 1]
                for j in range(gap_s, gap_e):
                    seq[j] = fill

    # 横向
    for r in range(nrows):
        _fill_seq(grid[r], _H_CHARS)

    # 纵向
    for c in range(ncols):
        col = [grid[r][c] for r in range(nrows)]
        _fill_seq(col, _V_CHARS)
        for r in range(nrows):
            grid[r][c] = col[r]

    # 斜线 \ (c - r = k)
    for k in range(-(nrows - 1), ncols):
        r0 = max(0, -k)
        c0 = r0 + k
        length = min(nrows - r0, ncols - c0)
        coords = [(r0 + i, c0 + i) for i in range(length)]
        if len(coords) >= 2:
            seq = [grid[r][c] for r, c in coords]
            _fill_seq(seq, _D1_CHARS)
            for i, (r, c) in enumerate(coords):
                grid[r][c] = seq[i]

    # 斜线 / (r + c = k)
    for k in range(nrows + ncols - 1):
        r0 = max(0, k - ncols + 1)
        c0 = k - r0
        length = min(nrows - r0, c0 + 1)
        coords = [(r0 + i, c0 - i) for i in range(length)]
        if len(coords) >= 2:
            seq = [grid[r][c] for r, c in coords]
            _fill_seq(seq, _D2_CHARS)
            for i, (r, c) in enumerate(coords):
                grid[r][c] = seq[i]

    # 删除孤立噪点：没有任何非空 8 连通邻居的边缘字符
    to_clear = [
        (r, c)
        for r in range(nrows)
        for c in range(ncols)
        if grid[r][c] in _ALL_EDGE
        and not any(
            0 <= r + dr < nrows and 0 <= c + dc < ncols and grid[r + dr][c + dc] != " "
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        )
    ]
    for r, c in to_clear:
        grid[r][c] = " "

    return ["".join(row).rstrip() for row in grid]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def generate_ascii_art(
    bgra: np.ndarray,
    *,
    rows: int,
    alpha_threshold: int,
    alpha_coverage: float,
    line_art_white_ratio: float,
    line_art_max_saturation: float,
    bilateral_d: int,
    bilateral_sigma_color: float,
    bilateral_sigma_space: float,
    canny_low_ratio: float,
    canny_low_floor: float,
    color_edge_quantile: float,
    edge_blur_sigma: float,
    intensity_breaks: list[float],
    merge_max_gap: int,
    debug_dir: Path,
) -> list[str]:
    """把图片转换为 ASCII 字符画，返回每行字符串的列表。

    rows 控制输出行数：图片等比缩放到高度 rows * CELL_H，列数按比例自动推导。
    带 alpha 通道的源图用 alpha 做主体遮罩（透明区域合成到白底参与边缘检测，
    主体外的字符格置空）；不带 alpha 的照片全图处理。

    流程：
    1. 等比缩放到 rows * CELL_H 高
    2. 边缘提取：线条画走 Otsu 二值化 + 细化，照片合并亮度 Canny 与 Lab 色彩梯度后细化
    3. 逐 8×16 字符格，用边缘图上的 Sobel 梯度方向选横/竖/斜字符，密度定线宽
    4. 合并线段：填补小空隙、删除孤立噪点
    5. 主体遮罩外的字符格置空
    """
    debug_dir.mkdir(parents=True, exist_ok=True)

    def _save(name: str, img: np.ndarray) -> None:
        if img.dtype == bool:
            img = img.astype(np.uint8) * 255
        cv2.imwrite(str(debug_dir / f"{name}.png"), img)

    # ── 等比缩放 ──
    h, w = bgra.shape[:2]
    target_h = rows * CELL_H
    scale = target_h / h
    target_w = int(w * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    img = cv2.resize(bgra, (target_w, target_h), interpolation=interp)

    # ── alpha 主体遮罩；透明区域合成到白底，避免垃圾 RGB 干扰边缘检测 ──
    if img.shape[2] == 4:
        foreground = img[..., 3] >= alpha_threshold
        bgr = img[..., :3].copy()
        bgr[~foreground] = 255
    else:
        foreground = np.ones(img.shape[:2], dtype=bool)
        bgr = img[..., :3]

    grid_rows = max(1, img.shape[0] // CELL_H)
    grid_cols = max(1, target_w // CELL_W)
    grid_h, grid_w = grid_rows * CELL_H, grid_cols * CELL_W

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _save("ascii_gray", gray)
    _save("ascii_subject", foreground)

    if _is_line_art(bgr, line_art_white_ratio, line_art_max_saturation):
        edges = _line_art_edges(gray, grid_h, grid_w)
    else:
        edges = _photo_edges(
            bgr,
            foreground,
            grid_h,
            grid_w,
            bilateral_d,
            bilateral_sigma_color,
            bilateral_sigma_space,
            canny_low_ratio,
            canny_low_floor,
            color_edge_quantile,
        )
    _save("ascii_edges", edges)

    # 边缘图先高斯模糊再做 Sobel，让梯度方向来自边缘几何而非图像纹理，
    # 避免纹理区域的角度误判
    edge_blur = cv2.GaussianBlur(
        edges.astype(np.float32), (0, 0), sigmaX=edge_blur_sigma
    )
    gx = cv2.Sobel(edge_blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(edge_blur, cv2.CV_64F, 0, 1, ksize=3)

    # ── 逐字符格分类 ──
    lines: list[str] = []
    for row in range(grid_rows):
        y0, y1 = row * CELL_H, (row + 1) * CELL_H
        row_chars: list[str] = []
        for col in range(grid_cols):
            x0, x1 = col * CELL_W, (col + 1) * CELL_W
            density = float(edges[y0:y1, x0:x1].mean())
            char = _classify_edge_cell(
                gx[y0:y1, x0:x1], gy[y0:y1, x0:x1], density, intensity_breaks
            )
            row_chars.append(char)
        lines.append("".join(row_chars))

    lines = _merge_edge_segments(lines, merge_max_gap)

    # ── 主体遮罩：遮罩外字符格置空 ──
    if not foreground.all():
        subject = down_mask(foreground, grid_cols, grid_rows, alpha_coverage)
        for r in range(grid_rows):
            if subject[r].all():
                continue
            chars = list(lines[r].ljust(grid_cols))
            for c in range(grid_cols):
                if not subject[r, c]:
                    chars[c] = " "
            lines[r] = "".join(chars).rstrip()

    return lines
