"""配置 schema 与校验。

所有参数默认值由 conf/（hydra-core）提供，本模块不设代码层默认值：
- dataclass 字段无默认值，值全部来自 conf/*.yaml；
- 结构校验（未知 key / 类型 / 缺失字段）由 OmegaConf.merge(structured schema, cfg) 完成；
- 业务规则校验（取值范围等）由 validate_settings 完成。
"""

from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf


@dataclass
class PerceiveConfig:
    denoise_d: int
    denoise_sigma: float
    mean_shift_sp: float
    mean_shift_sr: float


@dataclass
class Settings:
    auto_background: bool
    perceive: PerceiveConfig


def validate_settings(cfg: DictConfig) -> None:
    """业务规则校验，违反规则时抛 ValueError（列出全部问题）。"""
    p = cfg.perceive

    errors: list[str] = []

    # 缺失字段：merge 后为 ???，直接访问会抛 MissingMandatoryValue，先统一收集
    for field in ("denoise_d", "denoise_sigma", "mean_shift_sp", "mean_shift_sr"):
        if OmegaConf.is_missing(p, field):
            errors.append(
                f"perceive.{field} 缺失（conf/perceive/default.yaml 或覆盖项中未提供）"
            )

    if errors:
        raise ValueError("配置校验失败：\n  - " + "\n  - ".join(errors))

    # cv2.bilateralFilter 要求 d 为正奇数；d <= 0 时按 OpenCV 行为退化
    if p.denoise_d <= 0 or p.denoise_d % 2 == 0:
        errors.append(
            f"perceive.denoise_d={p.denoise_d} 必须是正奇数（bilateralFilter 要求）"
        )

    if p.denoise_sigma <= 0:
        errors.append(f"perceive.denoise_sigma={p.denoise_sigma} 必须 > 0")

    # cv2.pyrMeanShiftFiltering 要求 sp、sr 均 > 0
    if p.mean_shift_sp <= 0:
        errors.append(f"perceive.mean_shift_sp={p.mean_shift_sp} 必须 > 0")

    if p.mean_shift_sr <= 0:
        errors.append(f"perceive.mean_shift_sr={p.mean_shift_sr} 必须 > 0")

    if errors:
        raise ValueError("配置校验失败：\n  - " + "\n  - ".join(errors))


def load_settings(overrides: list[str] | None = None) -> DictConfig:
    """用 hydra-core 从 conf/ 加载 settings 并做结构 + 业务校验。

    overrides 为 hydra 覆盖项（dotlist），如 ["perceive.denoise_d=5", "auto_background=true"]。
    返回 struct 模式 DictConfig：未知 key / 类型错误在加载时直接报错，
    缺失字段与取值范围问题由 validate_settings 汇总为 ValueError。
    """
    from hydra import compose, initialize

    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="config", overrides=overrides)

    validated = OmegaConf.merge(OmegaConf.structured(Settings), cfg)
    validate_settings(validated)
    return validated
