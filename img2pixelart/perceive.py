import numpy as np
import cv2


def perceive(
    bgra: np.ndarray,
    denoise_d: int,
    denoise_sigma: float,
    mean_shift_sp: float,
    mean_shift_sr: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """感知阶段：双边滤波去噪 + 均值漂移分块。

    参数来自 hydra settings（conf/perceive/default.yaml），由 cli.py 从
    cfg.perceive.* 取值传入，可用 --setting perceive.xxx=yyy 在命令行覆盖。

    返回 (blocks, source_alpha)：blocks 为分块后的 BGR（若输入带 alpha 则合并回
    4 通道），source_alpha 为原始 alpha 通道（无 alpha 时为 None）。
    """
    has_alpha = bgra.ndim == 3 and bgra.shape[-1] == 4
    bgr = bgra[..., :3].copy() if has_alpha else bgra.copy()
    source_alpha = bgra[..., 3] if has_alpha else None

    denoised = cv2.bilateralFilter(bgr, denoise_d, denoise_sigma, denoise_sigma)
    blocks = cv2.pyrMeanShiftFiltering(denoised, mean_shift_sp, mean_shift_sr)

    if source_alpha is not None:
        blocks = np.dstack([blocks, source_alpha])

    return blocks, source_alpha
