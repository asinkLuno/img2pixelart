"""配置 schema 与业务规则校验。

所有参数默认值由 conf/（hydra-core）提供，本模块不设代码层默认值。
"""

from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf


@dataclass
class PerceiveConfig:
    mean_shift_sp: float
    mean_shift_sr: float
    requested_groups: int
    ramp_steps: int


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
    dither_style: str
    silhouette_darkness: float
    internal_darkness: float


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
    """ascii 阶段业务规则校验（cfg 为完整配置，含顶层 alpha_threshold）。"""
    a = cfg.ascii
    errors: list[str] = []

    if not 0 <= cfg.alpha_threshold <= 255:
        errors.append(f"alpha_threshold={cfg.alpha_threshold} 必须位于 [0, 255]")

    if not isinstance(cfg.debug, bool):
        errors.append(f"debug={cfg.debug!r} 必须是布尔值")

    errors += _collect_missing(
        a,
        "ascii",
        [
            "rows",
            "subject_coverage",
            "line_art_white_ratio",
            "denoise_strength",
            "merge_max_gap",
        ],
    )

    if errors:
        raise ValueError("配置校验失败：\n  - " + "\n  - ".join(errors))

    if a.rows < 1:
        errors.append(f"ascii.rows={a.rows} 必须 >= 1")
    if not 0.0 < a.subject_coverage <= 1.0:
        errors.append(f"ascii.subject_coverage={a.subject_coverage} 必须位于 (0, 1]")
    if not 0.0 <= a.line_art_white_ratio <= 1.0:
        errors.append(
            f"ascii.line_art_white_ratio={a.line_art_white_ratio} 必须位于 [0, 1]"
        )
    if a.denoise_strength <= 0:
        errors.append(f"ascii.denoise_strength={a.denoise_strength} 必须 > 0")
    if a.merge_max_gap < 0:
        errors.append(f"ascii.merge_max_gap={a.merge_max_gap} 必须 >= 0")

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
            "mean_shift_sp",
            "mean_shift_sr",
            "requested_groups",
            "ramp_steps",
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
        ["dither_style", "silhouette_darkness", "internal_darkness"],
    )

    if errors:
        raise ValueError("配置校验失败：\n  - " + "\n  - ".join(errors))

    # ── perceive 业务规则 ──
    if p.mean_shift_sp <= 0:
        errors.append(f"perceive.mean_shift_sp={p.mean_shift_sp} 必须 > 0")
    if p.mean_shift_sr <= 0:
        errors.append(f"perceive.mean_shift_sr={p.mean_shift_sr} 必须 > 0")
    if p.requested_groups < 1:
        errors.append(f"perceive.requested_groups={p.requested_groups} 必须 >= 1")
    if p.ramp_steps < 3:
        errors.append(f"perceive.ramp_steps={p.ramp_steps} 必须 >= 3")
    if not 0 <= cfg.alpha_threshold <= 255:
        errors.append(f"alpha_threshold={cfg.alpha_threshold} 必须位于 [0, 255]")

    if not isinstance(cfg.debug, bool):
        errors.append(f"debug={cfg.debug!r} 必须是布尔值")

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
    if r.dither_style not in (
        "none",
        "ordered",
        "diagonal",
        "clustered",
        "floyd_steinberg",
    ):
        errors.append(
            "render.dither_style="
            f"{r.dither_style!r} 必须是 none | ordered | diagonal | clustered | floyd_steinberg"
        )
    if not 0.0 <= r.silhouette_darkness <= 1.0:
        errors.append(
            f"render.silhouette_darkness={r.silhouette_darkness} 必须位于 [0, 1]"
        )
    if not 0.0 <= r.internal_darkness <= 1.0:
        errors.append(f"render.internal_darkness={r.internal_darkness} 必须位于 [0, 1]")

    if errors:
        raise ValueError("配置校验失败：\n  - " + "\n  - ".join(errors))
