"""OpenCV 图像格式校验、alpha 合成与统一的 Otsu-Canny 边缘检测。"""

from typing import Final

import cv2
import numpy as np

# Canny 低阈值 = Otsu × 比例，且不低于下限（AB-3 结论，见 issue #1）。
# perceive 与 ascii 照片边缘策略共用，勿在调用方各自维护第二套阈值。
CANNY_LOW_RATIO: Final = 0.33
CANNY_LOW_FLOOR: Final = 10.0


def validate_bgr_or_bgra(image: np.ndarray, *, name: str = "image") -> None:
    """校验 OpenCV 图像为非空 uint8 BGR / BGRA 数组，否则抛 ValueError / TypeError。"""
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(
            f"{name} 必须为 (H, W, 3) 或 (H, W, 4)，实际 shape={image.shape}"
        )
    if image.size == 0:
        raise ValueError(f"{name} 不能为空")
    if image.dtype != np.uint8:
        raise TypeError(f"{name} 必须为 uint8，实际为 {image.dtype}")


def otsu_canny(gray: np.ndarray) -> np.ndarray:
    """项目统一的 Canny 策略：low = max(Otsu × 0.33, 10)，high = Otsu。"""
    if gray.ndim != 2:
        raise ValueError(f"gray 必须为二维灰度图，实际 shape={gray.shape}")
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.Canny(
        gray,
        max(float(otsu) * CANNY_LOW_RATIO, CANNY_LOW_FLOOR),
        float(otsu),
    )


def binary_alpha_u8(alpha: np.ndarray) -> np.ndarray:
    """把布尔 / 数值前景掩码转换为 PNG 透明通道所需的 0/255 uint8。

    语义与下游各调用点原 ``alpha.astype(np.uint8) * 255`` 一致：
    布尔掩码按真值展开为 0/255，数值掩码先归零到布尔再展开，
    任何非零前景（含空流水线的 float32 0.0）都映射正确。
    """
    return np.asarray(alpha, dtype=bool).astype(np.uint8) * 255


def to_bgra(bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """BGR 颜色 + 独立前景掩码 → BGRA（保持 OpenCV 通道顺序约定）。"""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"bgr 必须为 (H, W, 3)，实际 shape={bgr.shape}")
    if alpha.shape != bgr.shape[:2]:
        raise ValueError(
            f"alpha 形状 {alpha.shape} 必须与 bgr 前两维 {bgr.shape[:2]} 一致"
        )
    return np.dstack([bgr, binary_alpha_u8(alpha)])
