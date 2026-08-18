import hashlib
from pathlib import Path

import cv2
import numpy as np
from hydra import compose, initialize_config_module

from img2pixelart import structure as structure_module
from img2pixelart.cli import run_pipeline


def test_rectangular_output(tmp_path: Path) -> None:
    source = cv2.imread("tests/cup.png", cv2.IMREAD_UNCHANGED)
    assert source is not None

    with initialize_config_module(version_base=None, config_module="img2pixelart.conf"):
        cfg = compose(
            config_name="config",
            overrides=["img=tests/cup.png", "width=80", "height=45"],
        )

    bgr, alpha = run_pipeline(source, cfg, tmp_path)

    assert bgr.shape == (45, 80, 3)
    assert alpha.shape == (45, 80)

    encoded = cv2.imencode(".png", np.dstack([bgr, alpha.astype(np.uint8) * 255]))[
        1
    ].tobytes()
    assert hashlib.sha256(encoded).hexdigest() == (
        "932ab5da77ee20121d89dd1cd3e4935b7d701d29e3b34a40ef6084528a107977"
    )


def test_small_sprite_detail_length_has_a_fixed_floor(monkeypatch) -> None:
    observed_lengths: list[int] = []
    original = structure_module.remove_short_components

    def record_length(mask: np.ndarray, min_length: int) -> np.ndarray:
        observed_lengths.append(min_length)
        return original(mask, min_length)

    monkeypatch.setattr(structure_module, "remove_short_components", record_length)
    shape = (8, 8)
    structure_module._simplify_small_sprite(
        alpha_down=np.ones(shape, dtype=bool),
        family_down=np.zeros(shape, dtype=np.int16),
        tier_down=np.zeros(shape, dtype=np.int16),
        canny_down=np.zeros(shape, dtype=bool),
        internal_detail=np.zeros(shape, dtype=bool),
        width=8,
        height=8,
        small_cleanup_threshold=64,
        small_cleanup_passes=0,
        small_tier_smooth_majority=5,
        small_skip_canny_under=40,
        edge_canny_support_radius=0,
        edge_min_length=1,
    )

    assert observed_lengths == [4]
