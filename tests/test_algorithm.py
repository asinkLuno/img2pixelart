import numpy as np

from img2pixelart.perceive import _merge_hue_centers


def test_hue_merge_does_not_chain_distant_centers() -> None:
    angles = np.deg2rad([0, 9, 18])
    centers = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)

    merged = _merge_hue_centers(centers, angular_threshold_degrees=10)

    assert len(merged) == 2
