"""配置加载与校验测试。运行：uv run python -m pytest tests/test_config.py -q"""

import pytest
from omegaconf import DictConfig, OmegaConf

from img2pixelart.config import Settings, load_settings, validate_settings


def test_load_settings_ok():
    cfg = load_settings()
    assert isinstance(cfg, DictConfig)
    assert cfg.perceive.denoise_d == 9
    assert cfg.perceive.denoise_sigma == 75.0
    assert cfg.perceive.mean_shift_sp == 15.0
    assert cfg.perceive.mean_shift_sr == 40.0
    assert cfg.auto_background is False


def test_load_settings_override_ok():
    cfg = load_settings(["perceive.denoise_d=5", "auto_background=true"])
    assert cfg.perceive.denoise_d == 5
    assert cfg.auto_background is True


def test_unknown_key_rejected():
    from hydra.errors import ConfigCompositionException

    with pytest.raises(ConfigCompositionException) as ei:
        load_settings(["perceive.typo=1"])
    assert "typo" in str(ei.value)


def test_unknown_key_in_yaml_rejected():
    from omegaconf.errors import ConfigKeyError

    raw = OmegaConf.create(
        {"auto_background": False, "perceive": {"denoise_d": 9, "denoise_sigma": 75, "mean_shift_sp": 15, "mean_shift_sr": 40, "typo": 1}}
    )
    with pytest.raises(ConfigKeyError) as ei:
        OmegaConf.merge(OmegaConf.structured(Settings), raw)
    assert "typo" in str(ei.value)


def test_type_error_rejected():
    from omegaconf.errors import ValidationError

    with pytest.raises(ValidationError) as ei:
        load_settings(["perceive.denoise_d=abc"])
    assert "denoise_d" in str(ei.value)


def test_missing_field_rejected():
    # 缺 mean_shift_sr：merge 后为 ???，validate_settings 汇总为友好错误
    raw = OmegaConf.create(
        {"auto_background": False, "perceive": {"denoise_d": 9, "denoise_sigma": 75, "mean_shift_sp": 15}}
    )
    merged = OmegaConf.merge(OmegaConf.structured(Settings), raw)
    assert OmegaConf.is_missing(merged.perceive, "mean_shift_sr")
    with pytest.raises(ValueError, match="mean_shift_sr.*缺失"):
        validate_settings(merged)


def test_business_rules():
    # denoise_d 偶数 -> load_settings 内部即触发校验
    with pytest.raises(ValueError, match="正奇数"):
        load_settings(["perceive.denoise_d=8"])
    # denoise_d 非正
    with pytest.raises(ValueError, match="正奇数"):
        load_settings(["perceive.denoise_d=-3"])
    # sigma 非正
    with pytest.raises(ValueError, match="denoise_sigma"):
        load_settings(["perceive.denoise_sigma=0"])
    # sp / sr 非正 -> 一次性列出全部问题
    with pytest.raises(ValueError) as ei:
        load_settings(["perceive.mean_shift_sp=-1", "perceive.mean_shift_sr=0"])
    msg = str(ei.value)
    assert "mean_shift_sp" in msg and "mean_shift_sr" in msg


def test_validate_settings_unit():
    # 单独验证 validate_settings：构造合法 cfg 后改坏其中一个字段
    cfg = load_settings()
    cfg.perceive.denoise_d = 10
    with pytest.raises(ValueError, match="正奇数"):
        validate_settings(cfg)


def test_valid_business_rules_pass():
    cfg = load_settings(["perceive.denoise_d=5", "perceive.denoise_sigma=10", "perceive.mean_shift_sp=3", "perceive.mean_shift_sr=5"])
    validate_settings(cfg)  # 不应抛异常
