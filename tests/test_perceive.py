"""#4 感知阶段简化：死键清理、调试图编号、Canny Otsu 自适应。"""

import cv2
import numpy as np
from hydra import compose, initialize_config_module
from omegaconf import DictConfig

from img2pixelart.perceive import _BILATERAL_D, _BILATERAL_SIGMA, perceive

# perceive 返回 dict 的全部存活键（#4 后无死键）。
LIVE_KEYS = {
    "L",
    "family_labels",
    "hue_directions_ab",
    "ramps_bgr",
    "ramp_l",
    "steps_per_family",
    "foreground",
    "alpha_full",
    "canny",
}


def _compose(**overrides: object):
    with initialize_config_module(version_base=None, config_module="img2pixelart.conf"):
        return compose(
            config_name="config",
            overrides=[
                "img=tests/cup.png",
                *[f"perceive.{k}={v}" for k, v in overrides.items()],
            ],
        )


def _perceive(tmp_path) -> tuple[dict, DictConfig]:
    cfg = _compose()
    p = cfg.perceive
    source = cv2.imread("tests/cup.png", cv2.IMREAD_UNCHANGED)
    assert source is not None
    result = perceive(
        source,
        mean_shift_sp=p.mean_shift_sp,
        mean_shift_sr=p.mean_shift_sr,
        requested_groups=p.requested_groups,
        ramp_steps=p.ramp_steps,
        alpha_threshold=cfg.alpha_threshold,
        palette_bgr=None,
        debug=True,
        debug_dir=tmp_path,
    )
    return result, cfg


def test_perceive_returns_only_live_keys(tmp_path) -> None:
    result, _ = _perceive(tmp_path)
    assert set(result) == LIVE_KEYS


def test_perceive_debug_numbering_is_continuous(tmp_path) -> None:
    _perceive(tmp_path)
    for name in (
        "01_original",
        "02_denoised",
        "03_blocks",
        "04_families",
        "05_palette",
        "06_canny",
    ):
        assert (tmp_path / f"{name}.png").is_file(), name
    assert not (tmp_path / "07_reconstructed.png").exists()


def test_canny_uses_otsu_adaptive_thresholds(tmp_path) -> None:
    """Canny 阈值 = max(Otsu × 0.33, 10) / Otsu（AB-3 结论），输入为去噪灰度。"""
    result, cfg = _perceive(tmp_path)
    source = cv2.imread("tests/cup.png", cv2.IMREAD_UNCHANGED)
    assert source is not None

    if source.shape[2] == 4:
        foreground = source[..., 3] >= cfg.alpha_threshold
        bgr = source[..., :3].copy()
        transparent = ~foreground
        if transparent.any() and foreground.any():
            bgr = cv2.inpaint(bgr, transparent.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    else:
        bgr = source.copy()
        foreground = np.ones(bgr.shape[:2], dtype=bool)

    denoised = cv2.bilateralFilter(
        bgr, _BILATERAL_D, _BILATERAL_SIGMA, _BILATERAL_SIGMA
    )
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    expected = cv2.Canny(gray, max(float(otsu) * 0.33, 10.0), float(otsu))
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(foreground.astype(np.uint8), kernel).astype(bool)
    expected[~eroded] = 0

    assert np.array_equal(result["canny"], expected)
