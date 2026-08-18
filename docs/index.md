# img2pixelart

图片转像素画 CLI。

## 安装

```bash
pip install img2pixelart
pip install hydra-joblib-launcher   # 多线程 multirun 需要
```

或从源码安装：

```bash
uv sync
uv pip install hydra-joblib-launcher
```

## 用法

### 单张转换

```bash
uv run img2pixelart img=tests/cup.png width=96 height=96
```

输出在当前目录的 `result.png`。

### 图片转 ASCII 字符画

```bash
uv run img2pixelart ascii img=tests/cup.png ascii.rows=60
# → outputs/single/<ts>/result_ascii.txt
```

输出行数由 `ascii.rows` 控制，列数按图片宽高比自动推导（字符栅格按 8×16
半角等宽比例）。流程与像素画共享边缘算法：线条画走 Otsu 二值化 +
Zhang-Suen 细化，照片走双边滤波 Canny 后细化；每个字符格用 Sobel 梯度
方向选择横 / 竖 / 斜字符，边缘密度决定线宽。带 alpha 通道的源图用 alpha
做主体遮罩，透明背景自动置空。

### 裁边

去掉图片四周空白区域：

```bash
uv run img2pixelart crop-padding tests/cup.png
# → tests/cup_no_padding.png
```

### 参数扫网（multirun）

多个参数值用逗号分隔，`-m` 开启 multirun，结果自动拼成 `combined.png`：

```bash
# 渐变层次 × 抖动风格 × 描边，81 种组合
uv run img2pixelart -m \
  img=tests/cup.png \
  perceive.ramp_steps=5,7,9 \
  render.silhouette_dark_step=0,1,2 \
  render.internal_outline_dark_steps=0,1,2 \
  render.pattern_style=ordered,diagonal,clustered \
  width=96 height=96
```

### 并行扫网

加 Joblib launcher，`n_jobs=-1` 用满所有 CPU：

```bash
uv run img2pixelart -m \
  img=tests/cup.png \
  perceive.ramp_steps=5,7,9 \
  render.silhouette_dark_step=0,1,2 \
  render.internal_outline_dark_steps=0,1,2 \
  render.pattern_style=ordered,diagonal,clustered \
  width=96 height=96 \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=-1
```

## 关键参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `width` | 输出宽度 | 64 |
| `height` | 输出高度 | 64 |
| `perceive.ramp_steps` | 每色相的明度阶梯数 | 7 |
| `perceive.requested_groups` | 色相族数量 | 3 |
| `render.pattern_style` | 抖动风格：`ordered` / `diagonal` / `clustered` | ordered |
| `render.silhouette_dark_step` | 外轮廓暗化级数（0=无） | 0 |
| `render.silhouette_dark_scale` | 外轮廓亮度缩放（越小越深） | 0.75 |
| `render.internal_outline_dark_steps` | 内部描边暗化级数（0=无） | 2 |
| `render.internal_outline_dark_scale` | 内部描边亮度缩放（越小越深） | 0.6 |
| `render.dither_fraction_min` | 抖动区域下限 | 0.18 |
| `render.dither_fraction_max` | 抖动区域上限 | 0.82 |

完整参数见 `img2pixelart/conf/` 目录下各阶段 YAML 文件。
