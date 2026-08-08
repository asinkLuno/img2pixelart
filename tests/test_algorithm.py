import numpy as np

from img2pixelart.fit import fit_ramps_to_palette, quantize_to_ramps
from img2pixelart.perceive import _merge_hue_centers
from img2pixelart.structure import _simplify_small_sprite


def test_hue_merge_does_not_chain_distant_centers() -> None:
    angles = np.deg2rad([0, 9, 18])
    centers = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)

    merged = _merge_hue_centers(centers, angular_threshold_degrees=10)

    assert len(merged) == 2


def test_palette_fit_preserves_ramps_and_exact_colors() -> None:
    palette_bgr = np.array([[0, 0, 255], [0, 0, 128], [255, 0, 0]], dtype=np.uint8)
    image = np.array([[[5, 5, 240], [240, 5, 5]]], dtype=np.uint8)
    fitted = fit_ramps_to_palette(image.astype(np.float32), palette_bgr)
    families = np.array([[0, 0]], dtype=np.int16)

    result = quantize_to_ramps(
        image,
        families,
        fitted,
        np.array([2], dtype=np.int32),
    )

    assert fitted.shape == image.shape
    assert np.array_equal(result[0, 0], palette_bgr[0])
    assert np.array_equal(result[0, 1], palette_bgr[2])


def test_large_sprite_keeps_internal_detail() -> None:
    empty = np.zeros((3, 3), dtype=bool)
    detail = empty.copy()
    detail[1, 1] = True
    labels = np.zeros((3, 3), dtype=np.int16)

    result = _simplify_small_sprite(
        alpha_down=~empty,
        family_down=labels,
        tier_down=labels,
        canny_down=detail,
        internal_detail=detail,
        size=96,
        small_cleanup_threshold=64,
        small_hole_close=True,
        small_cleanup_passes=2,
        small_tier_smooth_majority=5,
        small_skip_canny_under=40,
        edge_canny_support_radius=6,
        edge_min_length=2,
        small_detail_min_length=4,
        edge_open_before_thin=False,
    )

    assert np.array_equal(result["internal_detail"], detail)
