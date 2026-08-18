"""调试中间图的条件写入：debug=false 时完全不触碰文件系统。"""

from pathlib import Path

import cv2
import numpy as np


class DebugImageWriter:
    """仅在启用时创建输出目录，并按 ``{name}.png`` 写入调试图。

    bool 掩码自动转换为 0/255 uint8，与原有各阶段 ``_save`` 行为一致。
    """

    def __init__(self, enabled: bool, directory: Path) -> None:
        self.enabled = enabled
        self.directory = directory
        if enabled:
            directory.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, image: np.ndarray) -> None:
        """写入 ``directory/{name}.png``；未启用时直接返回。"""
        if not self.enabled:
            return
        if image.dtype == bool:
            image = image.astype(np.uint8) * 255
        cv2.imwrite(str(self.directory / f"{name}.png"), image)
