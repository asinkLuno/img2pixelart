import shutil
from pathlib import Path
from typing import Annotated

import cv2
import cyclopts
import numpy as np
from loguru import logger
from omegaconf import OmegaConf

from .config import load_settings
from .perceive import perceive
from .render import render
from .structure import structure

app = cyclopts.App(name="img2pixelart")


@app.default
def main(
    img: Path,
    auto_background: bool = False,
    setting: Annotated[
        list[str] | None, cyclopts.Parameter(allow_repeating=True)
    ] = None,
) -> None:
    """将图片转换为像素画风格。

    算法 settings 由 conf/config.yaml（hydra-core）提供，可用 --setting key=value 覆盖。
    """
    cfg = load_settings(setting)
    logger.info("settings loaded:\n{}", OmegaConf.to_yaml(cfg))

    if auto_background:
        cfg.auto_background = True
    logger.info("auto_background = {}", cfg.auto_background)
    if cfg.auto_background:
        logger.warning("auto_background 开关尚未接入像素画管线，当前为 no-op")

    bgra = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
    if bgra is None:
        raise ValueError(f"cannot read image: {img}")
    if bgra.ndim != 3 or bgra.shape[2] not in (3, 4):
        raise ValueError(f"image must be BGR or BGRA, got shape {bgra.shape}")

    debug_dir = img.parent / img.stem
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True)

    # ── perceive ──
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

    # ── structure ──
    s = cfg.structure
    struct = structure(
        perceived,
        size=s.size,
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
        s.size,
        s.size,
        struct["alpha_down"].sum(),
        struct["small_cleanup_applied"],
    )

    # ── render ──
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

    # ── 叠 alpha 写出 ──
    alpha = struct["alpha_down"].astype(np.uint8) * 255
    final_bgra = np.dstack([final_bgr, alpha])

    out = img.parent / f"{img.stem}_pixelart.png"
    if not cv2.imwrite(str(out), final_bgra):
        logger.error(f"failed to write output: {out}")
        raise SystemExit(3)
    logger.info("output: {}", out)
    logger.info("debug intermediates: {}", debug_dir)


if __name__ == "__main__":
    app()
