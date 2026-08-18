import hashlib
from pathlib import Path

import cv2
from hydra import compose, initialize_config_module

from img2pixelart.ascii import _ALL_EDGE, CELL_W, generate_ascii_art


def _compose(**overrides: object):
    with initialize_config_module(version_base=None, config_module="img2pixelart.conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "img=tests/cup.png",
                *[f"ascii.{k}={v}" for k, v in overrides.items()],
            ],
        )
    return cfg


def test_ascii_art_from_conf(tmp_path: Path) -> None:
    source = cv2.imread("tests/cup.png", cv2.IMREAD_UNCHANGED)
    assert source is not None

    cfg = _compose(rows=40)
    a = cfg.ascii
    lines = generate_ascii_art(
        source,
        rows=a.rows,
        alpha_threshold=cfg.alpha_threshold,
        subject_coverage=a.subject_coverage,
        line_art_white_ratio=a.line_art_white_ratio,
        denoise_strength=a.denoise_strength,
        merge_max_gap=a.merge_max_gap,
        debug=True,
        debug_dir=tmp_path,
    )

    # 行数严格等于 rows（740×740 方图 → 列数恰为 rows 的两倍）
    assert len(lines) == 40
    assert all(len(line.rstrip()) <= 40 * 2 for line in lines)

    # 输出只含空白与边缘字符表中的字符
    used = set("".join(lines))
    assert used <= _ALL_EDGE | {" "}

    # 有实际内容，且 alpha 主体遮罩生效（四角字符格应被置空）
    assert any(line.strip() for line in lines)
    assert not lines[0].startswith(("╌", "─", "━", "│"))

    # 调试产物落盘
    assert (tmp_path / "ascii_gray.png").exists()
    assert (tmp_path / "ascii_edges.png").exists()
    assert (tmp_path / "ascii_subject.png").exists()


def test_ascii_art_rows_control_output_height(tmp_path: Path) -> None:
    source = cv2.imread("tests/cup.png", cv2.IMREAD_UNCHANGED)
    assert source is not None

    cfg = _compose(rows=20)
    a = cfg.ascii
    lines = generate_ascii_art(
        source,
        rows=a.rows,
        alpha_threshold=cfg.alpha_threshold,
        subject_coverage=a.subject_coverage,
        line_art_white_ratio=a.line_art_white_ratio,
        denoise_strength=a.denoise_strength,
        merge_max_gap=a.merge_max_gap,
        debug=True,
        debug_dir=tmp_path,
    )

    assert len(lines) == 20
    assert max(len(line) for line in lines) <= 20 * 2 + CELL_W - 1


def test_default_ascii_output_is_stable(tmp_path: Path) -> None:
    """默认配置的 ASCII 输出哈希锁定（#3 合并后应与改动前一致）。"""
    source = cv2.imread("tests/cup.png", cv2.IMREAD_UNCHANGED)
    assert source is not None

    cfg = _compose()
    a = cfg.ascii
    lines = generate_ascii_art(
        source,
        rows=a.rows,
        alpha_threshold=cfg.alpha_threshold,
        subject_coverage=a.subject_coverage,
        line_art_white_ratio=a.line_art_white_ratio,
        denoise_strength=a.denoise_strength,
        merge_max_gap=a.merge_max_gap,
        debug=True,
        debug_dir=tmp_path,
    )
    digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    assert digest == "fde7ff5750aeb45b441b81b1ed4cbbd3d3e3e5b8ad8b046090c005a0e76c924b"
