"""共享图像辅助（image.py）与二值原语（binary_image.py）的行为锁定。"""

import cv2
import numpy as np
import pytest

from img2pixelart import structure  # 兼容再导出
from img2pixelart.binary_image import down_mask, thin
from img2pixelart.image import (
    CANNY_LOW_FLOOR,
    CANNY_LOW_RATIO,
    binary_alpha_u8,
    otsu_canny,
    to_bgra,
    validate_bgr_or_bgra,
)

# ---------------------------------------------------------------------------
# validate_bgr_or_bgra
# ---------------------------------------------------------------------------


def test_validate_accepts_bgr_and_bgra() -> None:
    validate_bgr_or_bgra(np.zeros((2, 3, 3), dtype=np.uint8))
    validate_bgr_or_bgra(np.zeros((2, 3, 4), dtype=np.uint8))


@pytest.mark.parametrize(
    "image",
    [
        np.zeros(4, dtype=np.uint8),  # 一维
        np.zeros((2, 3, 2), dtype=np.uint8),  # 通道数错误
        np.zeros((2, 3, 5), dtype=np.uint8),
        np.zeros((0, 3, 3), dtype=np.uint8),  # 空
    ],
)
def test_validate_rejects_bad_shapes(image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_bgr_or_bgra(image)


def test_validate_rejects_float_dtype() -> None:
    with pytest.raises(TypeError):
        validate_bgr_or_bgra(np.zeros((2, 3, 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# otsu_canny 与原内联实现完全一致
# ---------------------------------------------------------------------------


def test_otsu_canny_matches_inline_expression() -> None:
    gray = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (48, 1))
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    expected = cv2.Canny(
        gray,
        max(float(otsu) * CANNY_LOW_RATIO, CANNY_LOW_FLOOR),
        float(otsu),
    )
    assert np.array_equal(otsu_canny(gray), expected)


def test_otsu_canny_rejects_color_input() -> None:
    with pytest.raises(ValueError):
        otsu_canny(np.zeros((4, 4, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# alpha 合成
# ---------------------------------------------------------------------------


def test_binary_alpha_u8_expands_bool_and_numeric_masks() -> None:
    bool_mask = np.array([[True, False], [False, True]])
    assert np.array_equal(
        binary_alpha_u8(bool_mask), np.array([[255, 0], [0, 255]], dtype=np.uint8)
    )
    # float32 0.0（空流水线返回值）与正常布尔掩码语义一致
    assert np.array_equal(
        binary_alpha_u8(np.zeros((2, 2), dtype=np.float32)),
        np.zeros((2, 2), dtype=np.uint8),
    )


def test_to_bgra_keeps_bgr_channels_and_alpha() -> None:
    bgr = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    alpha = np.array([[True, False]])
    bgra = to_bgra(bgr, alpha)
    assert bgra.shape == (1, 2, 4)
    assert bgra.dtype == np.uint8
    assert np.array_equal(bgra[..., :3], bgr)
    assert np.array_equal(bgra[..., 3], np.array([[255, 0]], dtype=np.uint8))


def test_to_bgra_rejects_mismatched_shapes() -> None:
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        to_bgra(bgr, np.zeros((3, 2)))
    with pytest.raises(ValueError):
        to_bgra(np.zeros((2, 2, 4), dtype=np.uint8), np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# binary_image 原语
# ---------------------------------------------------------------------------


def test_down_mask_coverage_threshold() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[:, :2] = True  # 左半覆盖
    out = down_mask(mask, 2, 1, threshold=0.5)
    assert out.dtype == np.bool_
    assert out.tolist() == [[True, False]]


def test_down_mask_accepts_u8_input() -> None:
    mask = np.zeros((2, 2), dtype=np.uint8)
    mask[0, 0] = 255
    assert down_mask(mask, 2, 2, threshold=0.5).tolist() == [
        [True, False],
        [False, False],
    ]


def test_thin_reduces_thick_line_to_one_pixel() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 1:6] = True  # 3px 粗横线
    skeleton = thin(mask)
    assert skeleton.dtype == np.bool_
    # 骨架是子集且确实变细，且不存在 2x2 实心块（1px 宽不变量）
    assert skeleton.sum() < mask.sum()
    assert not (
        skeleton[:-1, :-1] & skeleton[1:, :-1] & skeleton[:-1, 1:] & skeleton[1:, 1:]
    ).any()


def test_thin_empty_mask_is_unchanged() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    assert np.array_equal(thin(mask), mask)


def test_structure_reexports_binary_primitives() -> None:
    assert structure.thin is thin
    assert structure.down_mask is down_mask
