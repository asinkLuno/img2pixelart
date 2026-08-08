import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from hydra import compose, initialize_config_module
from omegaconf import DictConfig, OmegaConf
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .cli import run_pipeline
from .config import validate_settings

Control = QCheckBox | QComboBox | QDoubleSpinBox | QSpinBox
CHOICES = {
    "render.dither_method": ["none", "bayer", "floyd_steinberg", "pattern"],
    "render.pattern_style": ["ordered", "diagonal", "clustered"],
}
NUMERIC_LIMITS: dict[str, tuple[float, float]] = {
    "perceive.denoise_d": (1, 1_000_000),
    "perceive.denoise_sigma": (0.000001, 1_000_000),
    "perceive.mean_shift_sp": (0.000001, 1_000_000),
    "perceive.mean_shift_sr": (0.000001, 1_000_000),
    "perceive.requested_groups": (1, 1_000_000),
    "perceive.chroma_floor": (0, 1_000_000),
    "perceive.minimum_fit_pixels": (1, 1_000_000),
    "perceive.spherical_kmeans_iterations": (1, 1_000_000),
    "perceive.ramp_steps": (3, 1_000_000),
    "perceive.ramp_minimum_span": (0, 100),
    "perceive.ramp_low_quantile": (0, 1),
    "perceive.ramp_high_quantile": (0, 1),
    "perceive.ramp_minimum_family_pixels": (1, 1_000_000),
    "perceive.ramp_chroma_quantile": (0, 1),
    "perceive.ramp_endpoint_chroma_scale": (0, 1),
    "perceive.ramp_maximum_chroma": (0.000001, 1_000_000),
    "perceive.canny_low": (0, 1_000_000),
    "perceive.canny_high": (0, 1_000_000),
    "perceive.alpha_threshold": (0, 255),
    "structure.alpha_coverage": (0.000001, 1),
    "structure.edge_coverage": (0, 1),
    "structure.edge_min_length": (1, 1_000_000),
    "structure.edge_canny_support_radius": (0, 1_000_000),
    "structure.small_cleanup_passes": (0, 1_000_000),
    "structure.small_tier_smooth_majority": (1, 9),
    "render.dither_fraction_min": (0, 1),
    "render.dither_fraction_max": (0, 1),
    "render.dither_gradient_min": (0, 1_000_000),
    "render.silhouette_dark_step": (0, 1_000_000),
    "render.silhouette_dark_scale": (0, 1),
    "render.internal_outline_dark_steps": (0, 1_000_000),
    "render.internal_outline_dark_scale": (0, 1),
}


class MainWindow(QMainWindow):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.source: np.ndarray | None = None
        self.result: np.ndarray | None = None
        self.controls: dict[str, Control] = {}
        self.debug_dir = tempfile.TemporaryDirectory(prefix="img2pixelart-ui-")
        self.setWindowTitle(cfg.ui.title)
        self.resize(cfg.ui.window_width, cfg.ui.window_height)

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.addWidget(self._control_panel())
        layout.addWidget(self._preview_panel(), 1)
        self.setCentralWidget(root)
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(cfg.ui.render_delay_ms)
        self.timer.timeout.connect(self.render_preview)

    def _control_panel(self) -> QScrollArea:
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)

        open_button = QPushButton("打开图片")
        open_button.clicked.connect(self.open_image)
        panel_layout.addWidget(open_button)

        palette_buttons = QHBoxLayout()
        load_palette_button = QPushButton("加载调色板")
        load_palette_button.clicked.connect(self.open_palette)
        clear_palette_button = QPushButton("清除")
        clear_palette_button.clicked.connect(self.clear_palette)
        palette_buttons.addWidget(load_palette_button)
        palette_buttons.addWidget(clear_palette_button)
        panel_layout.addLayout(palette_buttons)
        self.palette_label = QLabel("未使用调色板")
        self.palette_label.setWordWrap(True)
        panel_layout.addWidget(self.palette_label)

        self._add_group(panel_layout, "全局", {"size": self.cfg.size})
        for section, title in (
            ("perceive", "感知"),
            ("structure", "结构"),
            ("render", "渲染"),
        ):
            self._add_group(panel_layout, title, self.cfg[section], section)

        self.save_button = QPushButton("保存结果")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_result)
        panel_layout.addWidget(self.save_button)
        panel_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(self.cfg.ui.controls_width)
        scroll.setWidget(panel)
        return scroll

    def _add_group(
        self,
        layout: QVBoxLayout,
        title: str,
        values: DictConfig | dict[str, Any],
        prefix: str = "",
    ) -> None:
        group = QGroupBox(title)
        form = QFormLayout(group)
        for raw_name, value in values.items():
            name = str(raw_name)
            path = f"{prefix}.{name}" if prefix else name
            widget = self._control(path, value)
            self.controls[path] = widget
            form.addRow(name, widget)
        layout.addWidget(group)

    def _control(self, path: str, value: Any) -> Control:
        if path in CHOICES:
            widget = QComboBox()
            widget.addItems(CHOICES[path])
            widget.setCurrentText(value)
            widget.currentIndexChanged.connect(self.schedule_render)
            return widget
        if isinstance(value, bool):
            checkbox = QCheckBox()
            checkbox.setChecked(value)
            checkbox.checkStateChanged.connect(self.schedule_render)
            return checkbox
        if isinstance(value, int):
            integer = QSpinBox()
            minimum, maximum = NUMERIC_LIMITS.get(path, (-1_000_000, 1_000_000))
            integer.setRange(int(minimum), int(maximum))
            integer.setSingleStep(
                self.cfg.ui.size_step
                if path == "size"
                else 2
                if path == "perceive.denoise_d"
                else 1
            )
            integer.setValue(value)
            integer.valueChanged.connect(self.schedule_render)
            return integer
        decimal = QDoubleSpinBox()
        minimum, maximum = NUMERIC_LIMITS.get(path, (-1_000_000, 1_000_000))
        decimal.setRange(minimum, maximum)
        decimal.setDecimals(6)
        decimal.setSingleStep(self.cfg.ui.decimal_step)
        decimal.setValue(value)
        decimal.valueChanged.connect(self.schedule_render)
        return decimal

    def _preview_panel(self) -> QScrollArea:
        previews = QHBoxLayout()
        self.original_label = QLabel("请选择图片")
        self.result_label = QLabel("调整参数后将在这里预览")
        for title, label in (
            ("原图", self.original_label),
            ("像素画", self.result_label),
        ):
            column = QVBoxLayout()
            heading = QLabel(title)
            heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            column.addWidget(heading)
            column.addWidget(label, 1)
            previews.addLayout(column, 1)
        content = QWidget()
        content.setLayout(previews)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def open_image(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "打开图片", "", "图片 (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if not filename:
            return
        source = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if source is None or source.ndim != 3 or source.shape[2] not in (3, 4):
            QMessageBox.warning(self, "无法打开", "请选择 RGB 或 RGBA 图片。")
            return
        self.source = source
        self.original_label.setPixmap(self._pixmap(source, smooth=True))
        self.render_preview()

    def open_palette(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "加载调色板", "", "调色板 (*.txt);;所有文件 (*)"
        )
        if filename:
            self.cfg.palette = filename
            self.palette_label.setText(filename)
            self.schedule_render()

    def clear_palette(self) -> None:
        self.cfg.palette = None
        self.palette_label.setText("未使用调色板")
        self.schedule_render()

    def schedule_render(self) -> None:
        if self.source is not None:
            self.timer.start()

    def render_preview(self) -> None:
        if self.source is None:
            return
        self._sync_config()
        if not self._validate_controls():
            return
        try:
            final_bgr, alpha = run_pipeline(
                self.source, self.cfg, Path(self.debug_dir.name)
            )
        except (ValueError, cv2.error) as error:
            self.status.showMessage(str(error).replace("\n", " "))
            return
        self.result = np.dstack([final_bgr, alpha.astype(np.uint8) * 255])
        self.result_label.setPixmap(self._pixmap(self.result, smooth=False))
        self.save_button.setEnabled(True)
        self.status.showMessage("预览已更新")

    def _sync_config(self) -> None:
        for path, widget in self.controls.items():
            if isinstance(widget, QComboBox):
                value: str | bool | float | int = widget.currentText()
            elif isinstance(widget, QCheckBox):
                value = widget.isChecked()
            else:
                value = widget.value()
            OmegaConf.update(self.cfg, path, value)

    def _validate_controls(self) -> bool:
        for widget in self.controls.values():
            widget.setStyleSheet("")
        try:
            validate_settings(self.cfg)
        except ValueError as error:
            message = str(error)
            invalid = {path for path in self.controls if path in message}
            if "perceive ramp 分位" in message:
                invalid.update(
                    ("perceive.ramp_low_quantile", "perceive.ramp_high_quantile")
                )
            if "render dither 分位" in message:
                invalid.update(
                    ("render.dither_fraction_min", "render.dither_fraction_max")
                )
            for path in invalid:
                self.controls[path].setStyleSheet("border: 1px solid #d33")
            self.status.showMessage(message.replace("\n", " "))
            return False
        return True

    def save_result(self) -> None:
        if self.result is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "result.png", "PNG 图片 (*.png)"
        )
        if filename and not cv2.imwrite(filename, self.result):
            QMessageBox.warning(self, "保存失败", filename)

    def _pixmap(self, image: np.ndarray, *, smooth: bool) -> QPixmap:
        rgba = cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2RGBA if image.shape[2] == 4 else cv2.COLOR_BGR2RGB,
        )
        qimage = QImage(
            rgba.tobytes(),
            rgba.shape[1],
            rgba.shape[0],
            rgba.strides[0],
            QImage.Format.Format_RGBA8888
            if rgba.shape[2] == 4
            else QImage.Format.Format_RGB888,
        ).copy()
        return QPixmap.fromImage(qimage).scaled(
            self.cfg.ui.preview_size,
            self.cfg.ui.preview_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
            if smooth
            else Qt.TransformationMode.FastTransformation,
        )


def main() -> None:
    with initialize_config_module(config_module="img2pixelart.conf", version_base=None):
        cfg = compose(config_name="config")
    app = QApplication(sys.argv)
    window = MainWindow(cfg)
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
