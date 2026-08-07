from pathlib import Path
from typing import Annotated

import cv2
import cyclopts
from loguru import logger
import numpy as np
from omegaconf import DictConfig, OmegaConf
from matplotlib import pyplot as plt

from .config import load_settings
from .perceive import perceive

app = cyclopts.App(name="img2pixelart")


@app.default
def main(
    img: Path,
    auto_background: bool = False,
    setting: Annotated[list[str] | None, cyclopts.Parameter(allow_repeating=True)] = None,
) -> None:
    """将图片转换为像素画风格。

    算法 settings 由 conf/config.yaml（hydra-core）提供，可用 --setting key=value 覆盖。
    """
    cfg = load_settings(setting)
    logger.info("settings loaded:\n{}", OmegaConf.to_yaml(cfg))

    if auto_background:
        cfg.auto_background = True
    logger.info("auto_background = {}", cfg.auto_background)
    if cfg.auto_background:
        logger.warning("auto_background 开关尚未接入像素画管线，当前为 no-op")

    bgra = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
    if bgra.ndim != 3 or bgra.shape[2] not in (3, 4):
        raise ValueError(f"cannot read image: {img}")

    p = cfg.perceive
    blocks, source_alpha = perceive(
        bgra,
        denoise_d=p.denoise_d,
        denoise_sigma=p.denoise_sigma,
        mean_shift_sp=p.mean_shift_sp,
        mean_shift_sr=p.mean_shift_sr,
    )
    logger.info("perceive: blocks={} alpha={}", blocks.shape, source_alpha is not None)


if __name__ == "__main__":
    app()
