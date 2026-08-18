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
    return cfg.ascii


def test_ascii_art_from_conf(tmp_path: Path) -> None:
    source = cv2.imread("tests/cup.png", cv2.IMREAD_UNCHANGED)
    assert source is not None

    a = _compose(rows=40)
    lines = generate_ascii_art(
        source,
        rows=a.rows,
        alpha_threshold=a.alpha_threshold,
        alpha_coverage=a.alpha_coverage,
        line_art_white_ratio=a.line_art_white_ratio,
        bilateral_d=a.bilateral_d,
        bilateral_sigma_color=a.bilateral_sigma_color,
        bilateral_sigma_space=a.bilateral_sigma_space,
        canny_low_ratio=a.canny_low_ratio,
        merge_max_gap=a.merge_max_gap,
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

    a = _compose(rows=20)
    lines = generate_ascii_art(
        source,
        rows=a.rows,
        alpha_threshold=a.alpha_threshold,
        alpha_coverage=a.alpha_coverage,
        line_art_white_ratio=a.line_art_white_ratio,
        bilateral_d=a.bilateral_d,
        bilateral_sigma_color=a.bilateral_sigma_color,
        bilateral_sigma_space=a.bilateral_sigma_space,
        canny_low_ratio=a.canny_low_ratio,
        merge_max_gap=a.merge_max_gap,
        debug_dir=tmp_path,
    )

    assert len(lines) == 20
    assert max(len(line) for line in lines) <= 20 * 2 + CELL_W - 1
