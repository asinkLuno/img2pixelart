from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.visual_regression import (
    EXPERIMENTS,
    PIXEL_SOURCES,
    SIZES,
    _apply_patch,
    _comparison_sheet,
    _hydra_pixel_command,
    _image_metrics,
)


def test_baseline_inputs_and_matrix_are_versioned() -> None:
    assert SIZES == (48, 64, 96)
    for source in PIXEL_SOURCES:
        assert source.is_file(), source


def test_experiments_cover_issue_candidates() -> None:
    assert set(EXPERIMENTS) == {"ab-2", "ab-3"}
    assert EXPERIMENTS["ab-2"][1].patch == "skip-bilateral"
    # AB-3 已落地为默认（Otsu 自适应），A 侧用补丁回退到固定阈值。
    assert EXPERIMENTS["ab-3"][0].patch == "fixed-canny"
    assert EXPERIMENTS["ab-3"][1].patch is None


def test_temporary_candidates_patch_only_copied_package(tmp_path: Path) -> None:
    perceive = tmp_path / "perceive.py"
    perceive_original = (
        "    denoised = cv2.bilateralFilter(\n"
        "        bgr, _BILATERAL_D, _BILATERAL_SIGMA, _BILATERAL_SIGMA\n"
        "    )\n"
    )
    perceive.write_text(perceive_original, encoding="utf-8")
    image = tmp_path / "image.py"
    image_original = (
        "    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)\n"
        "    return cv2.Canny(\n"
        "        gray,\n"
        "        max(float(otsu) * CANNY_LOW_RATIO, CANNY_LOW_FLOOR),\n"
        "        float(otsu),\n"
        "    )\n"
    )
    image.write_text(image_original, encoding="utf-8")

    _apply_patch(tmp_path, "skip-bilateral")
    assert "denoised = bgr.copy()" in perceive.read_text(encoding="utf-8")

    perceive.write_text(perceive_original, encoding="utf-8")
    image.write_text(image_original, encoding="utf-8")
    _apply_patch(tmp_path, "fixed-canny")
    patched = image.read_text(encoding="utf-8")
    assert "return cv2.Canny(gray, 40, 120)" in patched
    assert "denoised = bgr.copy()" not in perceive.read_text(encoding="utf-8")


def test_ab_experiments_run_with_debug_output(tmp_path: Path) -> None:
    """A/B 指标依赖 05_palette / 22_palette_strip，命令必须显式开启 debug。"""
    command = _hydra_pixel_command(
        PIXEL_SOURCES[0], SIZES[0], tmp_path, EXPERIMENTS["ab-3"][0]
    )
    assert "debug=true" in command


def test_image_metrics_and_sheet_include_alpha(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    image_a = np.zeros((2, 2, 4), dtype=np.uint8)
    image_b = image_a.copy()
    image_b[0, 0, 3] = 255
    assert cv2.imwrite(str(a), image_a)
    assert cv2.imwrite(str(b), image_b)

    assert _image_metrics(a, b)["changed_pixel_ratio"] == pytest.approx(0.25)
    sheet = tmp_path / "comparison.png"
    _comparison_sheet([("case", a, b)], sheet)
    assert sheet.is_file()
