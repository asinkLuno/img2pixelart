import numpy as np

from img2pixelart.fit import fit_ramps_to_palette, quantize_to_ramps
from img2pixelart.perceive import _merge_hue_centers


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
