"""#5 调试输出开关与 render 返回值清理。"""

import cv2
import numpy as np
from hydra import compose, initialize_config_module

from img2pixelart.cli import run_pipeline
from img2pixelart.perceive import perceive
from img2pixelart.render import render
from img2pixelart.structure import structure


def _cfg(**overrides: object):
    with initialize_config_module(version_base=None, config_module="img2pixelart.conf"):
        return compose(
            config_name="config",
            overrides=[
                "img=tests/cup.png",
                *[f"{k}={v}" for k, v in overrides.items()],
            ],
        )


def _source() -> np.ndarray:
    source = cv2.imread("tests/cup.png", cv2.IMREAD_UNCHANGED)
    assert source is not None
    return source


def test_debug_off_writes_no_debug_pngs(tmp_path) -> None:
    cfg = _cfg()
    assert cfg.debug is False
    run_pipeline(_source(), cfg, tmp_path)
    assert not list(tmp_path.glob("*.png"))
    assert not list(tmp_path.glob("*_*"))


def test_debug_on_writes_debug_pngs(tmp_path) -> None:
    cfg = _cfg(debug=True)
    run_pipeline(_source(), cfg, tmp_path)
    for name in (
        "01_original",
        "06_canny",
        "09_alpha_down",
        "17_outline",
        "18_hard_ramp",
        "22_palette_strip",
    ):
        assert (tmp_path / f"{name}.png").is_file(), name


def test_debug_true_output_hash_is_stable(tmp_path) -> None:
    """debug 开关只影响中间产物写入，不改变最终结果。"""
    cfg_off = _cfg()
    cfg_on = _cfg(debug=True)
    a, _ = run_pipeline(_source(), cfg_off, tmp_path / "off")
    b, _ = run_pipeline(_source(), cfg_on, tmp_path / "on")
    assert np.array_equal(a, b)


def test_render_returns_ndarray_only(tmp_path) -> None:
    """render() 不再返回 meta dict，只返回 final_bgr。"""
    cfg = _cfg(debug=False)
    p = cfg.perceive
    perceived = perceive(
        _source(),
        denoise_d=p.denoise_d,
        denoise_sigma=p.denoise_sigma,
        mean_shift_sp=p.mean_shift_sp,
        mean_shift_sr=p.mean_shift_sr,
        requested_groups=p.requested_groups,
        chroma_floor=p.chroma_floor,
        ramp_steps=p.ramp_steps,
        ramp_minimum_span=p.ramp_minimum_span,
        alpha_threshold=cfg.alpha_threshold,
        palette_bgr=None,
        debug=False,
        debug_dir=tmp_path,
    )
    s = cfg.structure
    struct = structure(
        perceived,
        width=80,
        height=45,
        alpha_coverage=s.alpha_coverage,
        edge_coverage=s.edge_coverage,
        edge_min_length=s.edge_min_length,
        edge_canny_support_radius=s.edge_canny_support_radius,
        small_cleanup_threshold=s.small_cleanup_threshold,
        small_cleanup_passes=s.small_cleanup_passes,
        small_tier_smooth_majority=s.small_tier_smooth_majority,
        small_skip_canny_under=s.small_skip_canny_under,
        debug=False,
        debug_dir=tmp_path,
    )
    r = cfg.render
    result = render(
        perceived,
        struct,
        dither_style=r.dither_style,
        silhouette_darkness=r.silhouette_darkness,
        internal_darkness=r.internal_darkness,
        steps_per_family=perceived["steps_per_family"],
        debug=False,
        debug_dir=tmp_path,
    )
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == (45, 80, 3)
