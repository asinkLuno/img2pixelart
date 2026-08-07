import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from .config import validate_settings
from .fit import (
    build_target_ramps,
    load_palette,
    match_families,
    palette_auto_config,
    remap_families_and_tiers,
    structure_palette_ramps,
)
from .perceive import perceive
from .render import render
from .structure import structure


def _run_pipeline(
    bgra: np.ndarray, cfg: DictConfig, debug_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    """perceive → structure → render，返回 (final_bgr, alpha_down)。"""
    debug_dir.mkdir(parents=True, exist_ok=True)

    # ── 调色盘预处理：必须在 perceive 之前，因为要据此设置色相/阶梯参数 ──
    palette_ramps = None
    if cfg.palette:
        palette_bgr = load_palette(hydra.utils.to_absolute_path(cfg.palette))
        palette_ramps = structure_palette_ramps(palette_bgr)
        logger.info(
            "palette: {} ramps from {} colors",
            len(palette_ramps),
            len(palette_bgr),
        )

        auto_groups, auto_steps = palette_auto_config(palette_ramps)
        cfg.perceive.requested_groups = auto_groups
        cfg.perceive.ramp_steps = auto_steps
        logger.info(
            "palette: auto-config requested_groups={} ramp_steps={}",
            auto_groups,
            auto_steps,
        )

    p = cfg.perceive
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
        debug_dir=debug_dir,
    )
    logger.info(
        "perceive: {} families × {} steps",
        len(perceived["hue_directions_ab"]),
        p.ramp_steps,
    )

    s = cfg.structure
    struct = structure(
        perceived,
        size=cfg.size,
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
        cfg.size,
        cfg.size,
        struct["alpha_down"].sum(),
        struct["small_cleanup_applied"],
    )

    # ── 套用目标调色盘 ──
    if palette_ramps is not None:
        family_mapping = match_families(
            perceived["hue_directions_ab"],
            perceived["ramp_l"],
            perceived["ramps_bgr"],
            palette_ramps,
        )
        logger.info(
            "palette: family mapping → {}",
            {i: int(t) for i, t in enumerate(family_mapping)},
        )

        target_ramps_bgr, target_ramp_l = build_target_ramps(
            palette_ramps,
            steps=p.ramp_steps,
        )

        new_family, new_tier = remap_families_and_tiers(
            struct["family_down"],
            struct["tier_down"],
            struct["alpha_down"],
            struct["L_down"],
            perceived["ramp_l"],
            target_ramp_l,
            family_mapping,
        )

        perceived["ramps_bgr"] = target_ramps_bgr
        perceived["ramp_l"] = target_ramp_l
        struct["family_down"] = new_family
        struct["tier_down"] = new_tier

    r = cfg.render
    final_bgr, _meta = render(
        perceived,
        struct,
        dither_method=r.dither_method,
        dither_fraction_min=r.dither_fraction_min,
        dither_fraction_max=r.dither_fraction_max,
        dither_gradient_min=r.dither_gradient_min,
        silhouette_dark_step=r.silhouette_dark_step,
        internal_outline_dark_steps=r.internal_outline_dark_steps,
        debug_dir=debug_dir,
    )

    return final_bgr, struct["alpha_down"]


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
    final_bgr, alpha = _run_pipeline(bgra, cfg, Path.cwd())

    alpha_u8 = alpha.astype(np.uint8) * 255
    final_bgra = np.dstack([final_bgr, alpha_u8])

    out = Path("result.png")
    if not cv2.imwrite(str(out), final_bgra):
        logger.error(f"failed to write output: {out}")
        raise SystemExit(3)
    logger.info("output: {}/{}", Path.cwd(), out)


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

    margin = max(2, min(h, w) // 20)
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


def main() -> None:
    """img2pixelart CLI 调度入口。

    用法:
      img2pixelart img=test.png                        # 像素画转换
      img2pixelart crop-padding <image_path>           # 裁边
    """
    if len(sys.argv) > 1 and sys.argv[1] == "crop-padding":
        if len(sys.argv) < 3:
            print("usage: img2pixelart crop-padding <image_path>", file=sys.stderr)
            raise SystemExit(2)
        crop_padding(Path(sys.argv[2]))
    else:
        _hydra_main()


if __name__ == "__main__":
    main()
