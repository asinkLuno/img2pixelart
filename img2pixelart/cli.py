import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
from hydra.experimental.callback import Callback
from loguru import logger
from omegaconf import DictConfig

from .ascii import generate_ascii_art
from .config import validate_ascii, validate_settings
from .fit import load_palette, quantize_to_ramps
from .perceive import perceive
from .render import render
from .structure import structure


def run_pipeline(
    bgra: np.ndarray, cfg: DictConfig, debug_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    """perceive → structure → render，返回 (final_bgr, alpha_down)。"""
    debug_dir.mkdir(parents=True, exist_ok=True)

    p = cfg.perceive
    palette_bgr = None

    if cfg.palette:
        palette_bgr = load_palette(hydra.utils.to_absolute_path(cfg.palette))
        logger.info("palette: {} colors", len(palette_bgr))

    perceived = perceive(
        bgra,
        denoise_d=p.denoise_d,
        denoise_sigma=p.denoise_sigma,
        mean_shift_sp=p.mean_shift_sp,
        mean_shift_sr=p.mean_shift_sr,
        requested_groups=p.requested_groups,
        chroma_floor=p.chroma_floor,
        merge_angle_degrees=p.merge_angle_degrees,
        minimum_lightness=p.minimum_lightness,
        minimum_fit_pixels=p.minimum_fit_pixels,
        maximum_fit_pixels=p.maximum_fit_pixels,
        random_seed=p.random_seed,
        spherical_kmeans_iterations=p.spherical_kmeans_iterations,
        ramp_steps=p.ramp_steps,
        ramp_minimum_span=p.ramp_minimum_span,
        ramp_low_quantile=p.ramp_low_quantile,
        ramp_high_quantile=p.ramp_high_quantile,
        ramp_minimum_family_pixels=p.ramp_minimum_family_pixels,
        ramp_chroma_quantile=p.ramp_chroma_quantile,
        ramp_endpoint_chroma_scale=p.ramp_endpoint_chroma_scale,
        ramp_maximum_chroma=p.ramp_maximum_chroma,
        canny_low=p.canny_low,
        canny_high=p.canny_high,
        alpha_threshold=p.alpha_threshold,
        palette_bgr=palette_bgr,
        debug_dir=debug_dir,
    )
    F_src = len(perceived["hue_directions_ab"])
    logger.info("perceive: {} families", F_src)

    # 无有效色相族 → 直接返回透明图
    if F_src == 0:
        empty_bgr = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)
        empty_alpha = np.zeros((cfg.height, cfg.width), dtype=np.float32)
        return empty_bgr, empty_alpha

    s = cfg.structure
    struct = structure(
        perceived,
        width=cfg.width,
        height=cfg.height,
        alpha_coverage=s.alpha_coverage,
        edge_coverage=s.edge_coverage,
        edge_min_length=s.edge_min_length,
        edge_canny_support_radius=s.edge_canny_support_radius,
        edge_open_before_thin=s.edge_open_before_thin,
        small_cleanup_threshold=s.small_cleanup_threshold,
        small_detail_min_length=s.small_detail_min_length,
        small_cleanup_passes=s.small_cleanup_passes,
        small_hole_close=s.small_hole_close,
        small_tier_smooth_majority=s.small_tier_smooth_majority,
        small_skip_canny_under=s.small_skip_canny_under,
        debug_dir=debug_dir,
    )
    logger.info(
        "structure: {}×{} grid, {} fg pixels, small_cleanup={}",
        cfg.width,
        cfg.height,
        struct["alpha_down"].sum(),
        struct["small_cleanup_applied"],
    )

    steps_per_family = perceived["steps_per_family"]

    r = cfg.render
    final_bgr, _meta = render(
        perceived,
        struct,
        dither_method=r.dither_method,
        pattern_style=r.pattern_style,
        dither_fraction_min=r.dither_fraction_min,
        dither_fraction_max=r.dither_fraction_max,
        dither_gradient_min=r.dither_gradient_min,
        silhouette_dark_step=r.silhouette_dark_step,
        silhouette_dark_scale=r.silhouette_dark_scale,
        internal_outline_dark_steps=r.internal_outline_dark_steps,
        internal_outline_dark_scale=r.internal_outline_dark_scale,
        steps_per_family=steps_per_family,
        debug_dir=debug_dir,
    )
    if palette_bgr is not None:
        final_bgr = quantize_to_ramps(
            final_bgr,
            struct["family_down"],
            perceived["ramps_bgr"],
            steps_per_family,
        )

    return final_bgr, struct["alpha_down"]


class _StitchCallback(Callback):
    """multirun 结束后自动拼接所有 result.png。"""

    def on_multirun_end(self, config: DictConfig, **_: object) -> None:  # type: ignore[override]
        sweep_dir = Path(hydra.utils.to_absolute_path(config.hydra.sweep.dir))
        if not sweep_dir.is_dir():
            return
        # 只有存在 result.png 时才拼接
        if not list(sweep_dir.rglob("result.png")):
            return
        stitch_results(sweep_dir, output_path=sweep_dir / "combined.png")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def _hydra_main(cfg: DictConfig) -> None:
    """将图片转换为像素画风格（Hydra pipeline）。"""
    validate_settings(cfg)

    img_path = Path(hydra.utils.to_absolute_path(cfg.img))
    bgra = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if bgra is None:
        raise ValueError(f"cannot read image: {img_path}")
    if bgra.ndim != 3 or bgra.shape[2] not in (3, 4):
        raise ValueError(f"image must be BGR or BGRA, got shape {bgra.shape}")

    # Hydra 已 chdir 到输出目录，中间产物和结果直接写当前目录
    final_bgr, alpha = run_pipeline(bgra, cfg, Path.cwd())

    alpha_u8 = alpha.astype(np.uint8) * 255
    final_bgra = np.dstack([final_bgr, alpha_u8])

    out = Path("result.png")
    if not cv2.imwrite(str(out), final_bgra):
        logger.error(f"failed to write output: {out}")
        raise SystemExit(3)
    logger.info("output: {}/{}", Path.cwd(), out)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def _ascii_main(cfg: DictConfig) -> None:
    """将图片转换为 ASCII 字符画（Hydra pipeline）。"""
    validate_ascii(cfg.ascii)

    img_path = Path(hydra.utils.to_absolute_path(cfg.img))
    bgra = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if bgra is None:
        raise ValueError(f"cannot read image: {img_path}")
    if bgra.ndim != 3 or bgra.shape[2] not in (3, 4):
        raise ValueError(f"image must be BGR or BGRA, got shape {bgra.shape}")

    # Hydra 已 chdir 到输出目录，中间产物和结果直接写当前目录
    a = cfg.ascii
    lines = generate_ascii_art(
        bgra,
        rows=a.rows,
        alpha_threshold=a.alpha_threshold,
        alpha_coverage=a.alpha_coverage,
        line_art_white_ratio=a.line_art_white_ratio,
        line_art_max_saturation=a.line_art_max_saturation,
        bilateral_d=a.bilateral_d,
        bilateral_sigma_color=a.bilateral_sigma_color,
        bilateral_sigma_space=a.bilateral_sigma_space,
        canny_low_ratio=a.canny_low_ratio,
        canny_low_floor=a.canny_low_floor,
        color_edge_quantile=a.color_edge_quantile,
        edge_blur_sigma=a.edge_blur_sigma,
        intensity_breaks=list(a.intensity_breaks),
        merge_max_gap=a.merge_max_gap,
        debug_dir=Path.cwd(),
    )

    out = Path("result_ascii.txt")
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("output: {}/{} ({} rows)", Path.cwd(), out, len(lines))


def crop_padding(img_path: Path) -> None:
    """裁掉图片边缘空白区域，保存为 {stem}_no_padding{ext}。

    有 alpha 通道直接用 alpha 找前景；否则用 Canny 边缘检测定位主体外轮廓。
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        logger.error(f"cannot read image: {img_path}")
        raise SystemExit(1)

    h, w = img.shape[:2]

    # ── 前景遮罩 ──
    if img.ndim == 3 and img.shape[2] == 4:
        fg = img[..., 3] >= 128
    else:
        gray = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # ponytail: MORPH_CLOSE 填充边缘间隙形成实心区域，大图可能需要更多 iterations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3).astype(bool)

    rows = np.any(fg, axis=1)
    cols = np.any(fg, axis=0)
    if not rows.any():
        logger.warning(f"no foreground found in {img_path}, skipping")
        return

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    margin = min(h, w) // 20
    y_min = max(0, int(y_min) - margin)
    y_max = min(h, int(y_max) + margin)
    x_min = max(0, int(x_min) - margin)
    x_max = min(w, int(x_max) + margin)

    cropped = img[y_min : y_max + 1, x_min : x_max + 1]
    out_path = img_path.with_stem(img_path.stem + "_no_padding")
    if not cv2.imwrite(str(out_path), cropped):
        logger.error(f"failed to write: {out_path}")
        raise SystemExit(3)
    logger.info(
        "cropped {} → {} ({}×{} → {}×{})",
        img_path.name,
        out_path.name,
        w,
        h,
        x_max - x_min + 1,
        y_max - y_min + 1,
    )


def stitch_results(
    multirun_dir: Path,
    output_path: Path | None = None,
    columns: int | None = None,
    no_label: bool = False,
) -> None:
    """把 multirun 输出目录里的所有 result.png 拼成一张网格图。"""
    results = sorted(multirun_dir.rglob("result.png"))
    if not results:
        logger.error(f"no result.png found in {multirun_dir}")
        raise SystemExit(1)

    images: list[np.ndarray] = []
    labels: list[str] = []
    for p in results:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        images.append(img)
        overrides_path = p.parent / ".hydra" / "overrides.yaml"
        if overrides_path.exists():
            lines = overrides_path.read_text().strip().splitlines()
            parts = [line.lstrip("- ") for line in lines if line.startswith("- ")]
            labels.append(", ".join(parts) if parts else p.parent.name)
        else:
            labels.append(p.parent.name)

    if not images:
        logger.error("no readable images found")
        raise SystemExit(1)

    n = len(images)
    cols = columns or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    ch = max(img.shape[2] for img in images)

    pad = 4
    label_h = 0
    short_labels: list[list[str]] = []
    font_scale = 0.35
    if not no_label:
        font_scale = max(0.25, min(0.45, max_w / 400))
        for raw in labels:
            parts = raw.split(", ")
            lines: list[str] = []
            cur = ""
            for part in parts:
                candidate = f"{cur}, {part}" if cur else part
                if len(candidate) <= 60:
                    cur = candidate
                else:
                    if cur:
                        lines.append(cur)
                    cur = part
            if cur:
                lines.append(cur)
            short_labels.append(lines)
        max_lines = max((len(ln) for ln in short_labels), default=0)
        label_h = int(max_lines * (max_h * font_scale * 1.6 / 10) + pad * 2)

    cell_h, cell_w = max_h + label_h, max_w
    canvas_h = rows * cell_h
    canvas_w = cols * cell_w
    canvas = np.zeros((canvas_h, canvas_w, ch), dtype=np.uint8)

    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        x0, y0 = c * cell_w, r * cell_h

        if short_labels and short_labels[idx]:
            for li, line in enumerate(short_labels[idx]):
                y_text = y0 + label_h - pad - (len(short_labels[idx]) - 1 - li) * 4
                cv2.putText(
                    canvas,
                    line,
                    (x0 + 2, max(y_text, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

        ih, iw = img.shape[:2]
        ox = x0 + (max_w - iw) // 2
        oy = y0 + label_h + (max_h - ih) // 2

        if img.shape[-1] != ch:
            if ch == 4:
                img = np.dstack(
                    [img[..., :3], np.full((ih, iw, 1), 255, dtype=np.uint8)]
                )
            else:
                img = img[..., :ch]

        canvas[oy : oy + ih, ox : ox + iw] = img

    out = output_path or (multirun_dir / "combined.png")
    suffix = out.suffix.lower()
    if suffix in (".jpg", ".jpeg") and canvas.shape[-1] == 4:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
    if not cv2.imwrite(str(out), canvas):
        logger.error(f"failed to write: {out}")
        raise SystemExit(3)
    logger.info("saved {}×{} grid ({} images) → {}", rows, cols, n, out)


def main() -> None:
    """img2pixelart CLI 调度入口。

    用法:
      img2pixelart img=test.png                        # 像素画转换
      img2pixelart ascii img=test.png ascii.rows=60    # ASCII 字符画
      img2pixelart crop-padding <image_path>           # 裁边
    """
    if len(sys.argv) > 1 and sys.argv[1] == "crop-padding":
        if len(sys.argv) < 3:
            print("usage: img2pixelart crop-padding <image_path>", file=sys.stderr)
            raise SystemExit(2)
        crop_padding(Path(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "ascii":
        # 摘掉子命令名，剩余参数交给 hydra 解析为配置覆盖
        del sys.argv[1]
        _ascii_main()
    else:
        _hydra_main()


if __name__ == "__main__":
    main()
