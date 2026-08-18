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
    _image_metrics,
)


def test_baseline_inputs_and_matrix_are_versioned() -> None:
    assert SIZES == (48, 64, 96)
    for source in PIXEL_SOURCES:
        assert source.is_file(), source


def test_experiments_cover_issue_candidates() -> None:
    assert set(EXPERIMENTS) == {"ab-1", "ab-2", "ab-3"}
    assert EXPERIMENTS["ab-1"][0].overrides == (
        "render.dither_method=bayer",
        "render.pattern_style=ordered",
    )
    assert EXPERIMENTS["ab-1"][1].overrides == (
        "render.dither_method=pattern",
        "render.pattern_style=ordered",
    )
    assert EXPERIMENTS["ab-2"][1].patch == "skip-bilateral"
    assert EXPERIMENTS["ab-3"][1].patch == "otsu-canny"


def test_temporary_candidates_patch_only_copied_package(tmp_path: Path) -> None:
    perceive = tmp_path / "perceive.py"
    original = (
        "denoised = cv2.bilateralFilter(bgr, denoise_d, denoise_sigma, denoise_sigma)\n"
        "canny = cv2.Canny(gray, canny_low, canny_high)\n"
    )
    perceive.write_text(original, encoding="utf-8")

    _apply_patch(tmp_path, "skip-bilateral")
    assert "denoised = bgr.copy()" in perceive.read_text(encoding="utf-8")

    perceive.write_text(original, encoding="utf-8")
    _apply_patch(tmp_path, "otsu-canny")
    patched = perceive.read_text(encoding="utf-8")
    assert "cv2.THRESH_BINARY + cv2.THRESH_OTSU" in patched
    assert "max(float(otsu) * 0.33, 10.0)" in patched


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
