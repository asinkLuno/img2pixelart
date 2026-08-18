"""配置 schema 与业务规则校验。

所有参数默认值由 conf/（hydra-core）提供，本模块不设代码层默认值。
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
    ramp_steps: int
    ramp_minimum_span: float
    canny_low: int
    canny_high: int
    alpha_threshold: int


@dataclass
class StructureConfig:
    alpha_coverage: float
    edge_coverage: float
    edge_min_length: int
    edge_canny_support_radius: int
    small_cleanup_threshold: int
    small_cleanup_passes: int
    small_tier_smooth_majority: int
    small_skip_canny_under: int


@dataclass
class RenderConfig:
    dither_method: str
    pattern_style: str
    dither_fraction_min: float
    dither_fraction_max: float
    dither_gradient_min: float
    silhouette_dark_step: int
    silhouette_dark_scale: float
    internal_outline_dark_steps: int
    internal_outline_dark_scale: float


@dataclass
class Settings:
    perceive: PerceiveConfig
    structure: StructureConfig
    render: RenderConfig


def _collect_missing(cfg: DictConfig, prefix: str, fields: list[str]) -> list[str]:
    """收集 prefix 下所有 OmegaConf.is_missing 的字段。"""
    errors: list[str] = []
    for field in fields:
        if OmegaConf.is_missing(cfg, field):
            errors.append(f"{prefix}.{field} 缺失（conf/ 中未提供）")
    return errors


def validate_ascii(cfg: DictConfig) -> None:
    """ascii 阶段业务规则校验。"""
    errors: list[str] = []

    errors += _collect_missing(
        cfg,
        "ascii",
        [
            "rows",
            "alpha_threshold",
            "alpha_coverage",
            "line_art_white_ratio",
            "bilateral_d",
            "bilateral_sigma_color",
            "bilateral_sigma_space",
            "canny_low_ratio",
            "merge_max_gap",
        ],
    )

    if errors:
        raise ValueError("配置校验失败：\n  - " + "\n  - ".join(errors))

    if cfg.rows < 1:
        errors.append(f"ascii.rows={cfg.rows} 必须 >= 1")
    if not 0 <= cfg.alpha_threshold <= 255:
        errors.append(f"ascii.alpha_threshold={cfg.alpha_threshold} 必须位于 [0, 255]")
    if not 0.0 < cfg.alpha_coverage <= 1.0:
        errors.append(f"ascii.alpha_coverage={cfg.alpha_coverage} 必须位于 (0, 1]")
    if not 0.0 <= cfg.line_art_white_ratio <= 1.0:
        errors.append(
            f"ascii.line_art_white_ratio={cfg.line_art_white_ratio} 必须位于 [0, 1]"
        )
    if cfg.bilateral_d <= 0:
        errors.append(f"ascii.bilateral_d={cfg.bilateral_d} 必须 > 0")
    if cfg.bilateral_sigma_color <= 0:
        errors.append(
            f"ascii.bilateral_sigma_color={cfg.bilateral_sigma_color} 必须 > 0"
        )
    if cfg.bilateral_sigma_space <= 0:
        errors.append(
            f"ascii.bilateral_sigma_space={cfg.bilateral_sigma_space} 必须 > 0"
        )
    if not 0.0 < cfg.canny_low_ratio <= 1.0:
        errors.append(f"ascii.canny_low_ratio={cfg.canny_low_ratio} 必须位于 (0, 1]")
    if cfg.merge_max_gap < 0:
        errors.append(f"ascii.merge_max_gap={cfg.merge_max_gap} 必须 >= 0")

    if errors:
        raise ValueError("配置校验失败：\n  - " + "\n  - ".join(errors))


def validate_settings(cfg: DictConfig) -> None:
    """业务规则校验，违反规则时抛 ValueError（列出全部问题）。"""
    p = cfg.perceive
    s = cfg.structure
    r = cfg.render

    errors: list[str] = []

    if cfg.width < 1:
        errors.append(f"width={cfg.width} 必须 >= 1")
    if cfg.height < 1:
        errors.append(f"height={cfg.height} 必须 >= 1")

    # ── perceive 缺失字段 ──
    errors += _collect_missing(
        p,
        "perceive",
        [
            "denoise_d",
            "denoise_sigma",
            "mean_shift_sp",
            "mean_shift_sr",
            "requested_groups",
            "chroma_floor",
            "ramp_steps",
            "ramp_minimum_span",
            "canny_low",
            "canny_high",
            "alpha_threshold",
        ],
    )

    # ── structure 缺失字段 ──
    errors += _collect_missing(
        s,
        "structure",
        [
            "alpha_coverage",
            "edge_coverage",
            "edge_min_length",
            "edge_canny_support_radius",
            "small_cleanup_threshold",
            "small_cleanup_passes",
            "small_tier_smooth_majority",
            "small_skip_canny_under",
        ],
    )

    # ── render 缺失字段 ──
    errors += _collect_missing(
        r,
        "render",
        [
            "dither_method",
            "pattern_style",
            "dither_fraction_min",
            "dither_fraction_max",
            "dither_gradient_min",
            "silhouette_dark_step",
            "silhouette_dark_scale",
            "internal_outline_dark_steps",
            "internal_outline_dark_scale",
        ],
    )

    if errors:
        raise ValueError("配置校验失败：\n  - " + "\n  - ".join(errors))

    # ── perceive 业务规则 ──
    if p.denoise_d <= 0 or p.denoise_d % 2 == 0:
        errors.append(f"perceive.denoise_d={p.denoise_d} 必须是正奇数")
    if p.denoise_sigma <= 0:
        errors.append(f"perceive.denoise_sigma={p.denoise_sigma} 必须 > 0")
    if p.mean_shift_sp <= 0:
        errors.append(f"perceive.mean_shift_sp={p.mean_shift_sp} 必须 > 0")
    if p.mean_shift_sr <= 0:
        errors.append(f"perceive.mean_shift_sr={p.mean_shift_sr} 必须 > 0")
    if p.requested_groups < 1:
        errors.append(f"perceive.requested_groups={p.requested_groups} 必须 >= 1")
    if p.chroma_floor < 0:
        errors.append(f"perceive.chroma_floor={p.chroma_floor} 必须 >= 0")
    if p.ramp_steps < 3:
        errors.append(f"perceive.ramp_steps={p.ramp_steps} 必须 >= 3")
    if p.ramp_minimum_span < 0 or p.ramp_minimum_span > 100:
        errors.append(
            f"perceive.ramp_minimum_span={p.ramp_minimum_span} 必须位于 [0, 100]"
        )
    if p.canny_low < 0 or p.canny_high < 0:
        errors.append("perceive canny 阈值必须 >= 0")
    if p.canny_low > p.canny_high:
        errors.append(
            f"perceive canny_low={p.canny_low} 必须 <= canny_high={p.canny_high}"
        )
    if not 0 <= p.alpha_threshold <= 255:
        errors.append(f"perceive.alpha_threshold={p.alpha_threshold} 必须位于 [0, 255]")

    # ── structure 业务规则 ──
    if not 0.0 < s.alpha_coverage <= 1.0:
        errors.append(f"structure.alpha_coverage={s.alpha_coverage} 必须位于 (0, 1]")
    if not 0.0 <= s.edge_coverage <= 1.0:
        errors.append(f"structure.edge_coverage={s.edge_coverage} 必须位于 [0, 1]")
    if s.edge_min_length < 1:
        errors.append(f"structure.edge_min_length={s.edge_min_length} 必须 >= 1")
    if s.edge_canny_support_radius < 0:
        errors.append(
            f"structure.edge_canny_support_radius={s.edge_canny_support_radius} 必须 >= 0"
        )
    if s.small_cleanup_passes < 0:
        errors.append(
            f"structure.small_cleanup_passes={s.small_cleanup_passes} 必须 >= 0"
        )
    if s.small_tier_smooth_majority < 1 or s.small_tier_smooth_majority > 9:
        errors.append(
            f"structure.small_tier_smooth_majority={s.small_tier_smooth_majority} 必须位于 [1, 9]"
        )

    # ── render 业务规则 ──
    if r.dither_method not in ("none", "bayer", "floyd_steinberg", "pattern"):
        errors.append(
            f"render.dither_method={r.dither_method!r} 必须是 none | bayer | floyd_steinberg | pattern"
        )
    if r.pattern_style not in ("ordered", "diagonal", "clustered"):
        errors.append(
            f"render.pattern_style={r.pattern_style!r} 必须是 ordered | diagonal | clustered"
        )
    if not 0.0 <= r.dither_fraction_min < r.dither_fraction_max <= 1.0:
        errors.append("render dither 分位参数必须满足 0 <= min < max <= 1")
    if r.dither_gradient_min < 0:
        errors.append(f"render.dither_gradient_min={r.dither_gradient_min} 必须 >= 0")
    if r.silhouette_dark_step < 0:
        errors.append(f"render.silhouette_dark_step={r.silhouette_dark_step} 必须 >= 0")
    if not 0.0 <= r.silhouette_dark_scale <= 1.0:
        errors.append(
            f"render.silhouette_dark_scale={r.silhouette_dark_scale} 必须位于 [0, 1]"
        )
    if r.internal_outline_dark_steps < 0:
        errors.append(
            f"render.internal_outline_dark_steps={r.internal_outline_dark_steps} 必须 >= 0"
        )
    if not 0.0 <= r.internal_outline_dark_scale <= 1.0:
        errors.append(
            f"render.internal_outline_dark_scale={r.internal_outline_dark_scale} 必须位于 [0, 1]"
        )

    if errors:
        raise ValueError("配置校验失败：\n  - " + "\n  - ".join(errors))
