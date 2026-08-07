from pathlib import Path
import cv2
import cyclopts
from loguru import logger
import numpy as np
from matplotlib import pyplot as plt

app = cyclopts.App(name="img2pixelart")


def estimate_foreground_mask(img,):
    # TODO: 主体检测
    return no_background_img


def perceive(bgra:np.ndarray,denoise_d,denoise_sigma,mean_shift_sp,mean_shift_sr):
    has_alpha=bgra.ndim==3 and bgra.shape[-1]==4
    bgr=bgra[...,:3].copy() if has_alpha else bgra.copy()
    source_alpha=bgra[...,3] if has_alpha else None

    denoised=cv2.bilateralFilter(bgr,denoise_d, denoise_sigma,denoise_sigma)
    blocks=cv2.pyrMeanShiftFiltering(denoised,mean_shift_sp,mean_shift_sr)

    h,w=bgr.shape[:2]
    side=max(h,w)


@app.default
def main(img: Path, auto_background: bool = False) -> None:
    """将图片转换为像素画风格。"""
    img=cv2.imread(str(img),cv2.IMREAD_UNCHANGED)
    if img.ndim!=3 or img.shape[2] not in (3,4):
        raise ValueError(f"cannot read image: {img}")

    if auto_background:
        img=estimate_foreground_mask(img)


if __name__=="__main__":
    app()