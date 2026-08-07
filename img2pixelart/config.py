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
    requested_groups: int
    chroma_floor: float
    merge_angle_degrees: float
    minimum_lightness: float
    minimum_fit_pixels: int
    maximum_fit_pixels: int
    random_seed: int
    ramp_steps: int
    ramp_minimum_span: float
    ramp_low_quantile: float
    ramp_high_quantile: float
    ramp_minimum_family_pixels: int
    ramp_chroma_quantile: float
    ramp_endpoint_chroma_scale: float
    ramp_maximum_chroma: float
    canny_low: int
    canny_high: int


@dataclass
class Settings:
    auto_background: bool
    perceive: PerceiveConfig


def validate_settings(cfg: DictConfig) -> None:
    """业务规则校验，违反规则时抛 ValueError（列出全部问题）。"""
    p = cfg.perceive

    errors: list[str] = []

    # 缺失字段：merge 后为 ???，直接访问会抛 MissingMandatoryValue，先统一收集
    for field in (
        "denoise_d",
        "denoise_sigma",
        "mean_shift_sp",
        "mean_shift_sr",
        "requested_groups",
        "chroma_floor",
        "merge_angle_degrees",
        "minimum_lightness",
        "minimum_fit_pixels",
        "maximum_fit_pixels",
        "random_seed",
        "ramp_steps",
        "ramp_minimum_span",
        "ramp_low_quantile",
        "ramp_high_quantile",
        "ramp_minimum_family_pixels",
        "ramp_chroma_quantile",
        "ramp_endpoint_chroma_scale",
        "ramp_maximum_chroma",
        "canny_low",
        "canny_high",
    ):
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

    if p.requested_groups < 1:
        errors.append(f"perceive.requested_groups={p.requested_groups} 必须 >= 1")

    if p.chroma_floor < 0:
        errors.append(f"perceive.chroma_floor={p.chroma_floor} 必须 >= 0")

    if p.minimum_fit_pixels < 1:
        errors.append(f"perceive.minimum_fit_pixels={p.minimum_fit_pixels} 必须 >= 1")

    if p.maximum_fit_pixels <= p.minimum_fit_pixels:
        errors.append(
            f"perceive.maximum_fit_pixels={p.maximum_fit_pixels} 必须 > "
            f"minimum_fit_pixels ({p.minimum_fit_pixels})"
        )

    if p.ramp_steps < 3:
        errors.append(f"perceive.ramp_steps={p.ramp_steps} 必须 >= 3")

    if p.ramp_minimum_span < 0 or p.ramp_minimum_span > 100:
        errors.append(
            f"perceive.ramp_minimum_span={p.ramp_minimum_span} 必须位于 [0, 100]"
        )

    if not 0 <= p.ramp_low_quantile < p.ramp_high_quantile <= 1:
        errors.append(
            f"perceive.ramp_low_quantile={p.ramp_low_quantile} / "
            f"ramp_high_quantile={p.ramp_high_quantile} 必须满足 0 <= low < high <= 1"
        )

    if p.ramp_minimum_family_pixels < 1:
        errors.append(
            f"perceive.ramp_minimum_family_pixels={p.ramp_minimum_family_pixels} 必须 >= 1"
        )

    if not 0 <= p.ramp_chroma_quantile <= 1:
        errors.append(
            f"perceive.ramp_chroma_quantile={p.ramp_chroma_quantile} 必须位于 [0, 1]"
        )

    if not 0 <= p.ramp_endpoint_chroma_scale <= 1:
        errors.append(
            f"perceive.ramp_endpoint_chroma_scale={p.ramp_endpoint_chroma_scale} 必须位于 [0, 1]"
        )

    if p.ramp_maximum_chroma <= 0:
        errors.append(f"perceive.ramp_maximum_chroma={p.ramp_maximum_chroma} 必须 > 0")

    if p.canny_low < 0 or p.canny_high < 0:
        errors.append(
            f"perceive.canny_low={p.canny_low} / canny_high={p.canny_high} 必须 >= 0"
        )

    if p.canny_low > p.canny_high:
        errors.append(
            f"perceive.canny_low={p.canny_low} 必须 <= canny_high={p.canny_high}"
        )

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
