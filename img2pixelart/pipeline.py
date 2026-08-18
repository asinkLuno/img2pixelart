"""像素画流水线编排：perceive → structure → render。

与 CLI / GUI 解耦的纯转换入口，供 ``img2pixelart.cli`` 与
``img2pixelart.ui`` 共用；参数默认值仍由 ``img2pixelart/conf/`` 提供。
"""

from pathlib import Path

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from .fit import load_palette, quantize_to_ramps
from .perceive import perceive
from .render import render
from .structure import structure


def run_pipeline(
    bgra: np.ndarray, cfg: DictConfig, debug_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    """perceive → structure → render，返回 (final_bgr, alpha_down)。"""
    debug = bool(cfg.debug)
    if debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    p = cfg.perceive
    palette_bgr = None

    if cfg.palette:
        palette_bgr = load_palette(hydra.utils.to_absolute_path(cfg.palette))
        logger.info("palette: {} colors", len(palette_bgr))

    perceived = perceive(
        bgra,
        mean_shift_sp=p.mean_shift_sp,
        mean_shift_sr=p.mean_shift_sr,
        requested_groups=p.requested_groups,
        ramp_steps=p.ramp_steps,
        alpha_threshold=cfg.alpha_threshold,
        palette_bgr=palette_bgr,
        debug=debug,
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
        small_cleanup_threshold=s.small_cleanup_threshold,
        small_cleanup_passes=s.small_cleanup_passes,
        small_tier_smooth_majority=s.small_tier_smooth_majority,
        small_skip_canny_under=s.small_skip_canny_under,
        debug=debug,
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
    final_bgr = render(
        perceived,
        struct,
        dither_style=r.dither_style,
        silhouette_darkness=r.silhouette_darkness,
        internal_darkness=r.internal_darkness,
        steps_per_family=steps_per_family,
        debug=debug,
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
