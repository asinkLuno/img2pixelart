from pathlib import Path

import cv2
from hydra import compose, initialize_config_module

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
