"""#3 参数迁移：验证旧键消失、新键与默认值就位。"""

from hydra import compose, initialize_config_module
from omegaconf import DictConfig

from img2pixelart.config import validate_ascii, validate_settings

LEGACY_RENDER_KEYS = (
    "dither_method",
    "pattern_style",
    "dither_fraction_min",
    "dither_fraction_max",
    "dither_gradient_min",
    "silhouette_dark_step",
    "silhouette_dark_scale",
    "internal_outline_dark_steps",
    "internal_outline_dark_scale",
)
LEGACY_ASCII_KEYS = (
    "alpha_threshold",
    "alpha_coverage",
    "bilateral_d",
    "bilateral_sigma_color",
    "bilateral_sigma_space",
)


def _compose() -> DictConfig:
    with initialize_config_module(version_base=None, config_module="img2pixelart.conf"):
        return compose(config_name="config", overrides=["img=tests/cup.png"])


def test_legacy_keys_are_gone() -> None:
    cfg = _compose()
    for key in LEGACY_RENDER_KEYS:
        assert key not in cfg.render, key
    for key in LEGACY_ASCII_KEYS:
        assert key not in cfg.ascii, key
    assert "alpha_threshold" not in cfg.perceive


def test_merged_keys_have_defaults() -> None:
    cfg = _compose()
    assert cfg.alpha_threshold == 128
    assert cfg.render.dither_style == "ordered"
    assert cfg.render.silhouette_darkness == 1.0
    assert cfg.render.internal_darkness == 1.0
    assert cfg.ascii.subject_coverage == 0.5
    assert cfg.ascii.denoise_strength == 1.0


def test_validation_passes_on_defaults() -> None:
    cfg = _compose()
    validate_settings(cfg)
    validate_ascii(cfg)
