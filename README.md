# img2pixelart

图片转像素画 CLI。

## 安装

```bash
pip install img2pixelart
```

或从源码安装：

```bash
uv sync
```

## 效果

<p align="center">
  <img src="docs/assets/banana_orig.png" width="128"> →
  <img src="docs/assets/banana_96.png" width="96">
  <img src="docs/assets/banana_64.png" width="64">
  <img src="docs/assets/banana_48.png" width="48">
  <img src="docs/assets/banana_32.png" width="32">
  <br>
  <img src="docs/assets/sakura_orig.png" width="128"> →
  <img src="docs/assets/sakura_96.png" width="96">
  <img src="docs/assets/sakura_64.png" width="64">
  <img src="docs/assets/sakura_48.png" width="48">
  <img src="docs/assets/sakura_32.png" width="32">
  <br>
  <img src="docs/assets/chair_orig.png" width="128"> →
  <img src="docs/assets/chair_96.png" width="96">
  <img src="docs/assets/chair_64.png" width="64">
  <img src="docs/assets/chair_48.png" width="48">
  <img src="docs/assets/chair_32.png" width="32">
</p>

不同色相族数量（`requested_groups`），尺寸固定 96px：

<p align="center">
  <img src="docs/assets/banana_orig.png" width="128"> →
  <img src="docs/assets/banana_g1.png" width="96">
  <img src="docs/assets/banana_g2.png" width="96">
  <img src="docs/assets/banana_g3.png" width="96">
  <img src="docs/assets/banana_g4.png" width="96">
  <br>
  <img src="docs/assets/sakura_orig.png" width="128"> →
  <img src="docs/assets/sakura_g1.png" width="96">
  <img src="docs/assets/sakura_g2.png" width="96">
  <img src="docs/assets/sakura_g3.png" width="96">
  <img src="docs/assets/sakura_g4.png" width="96">
  <br>
  <img src="docs/assets/chair_orig.png" width="128"> →
  <img src="docs/assets/chair_g1.png" width="96">
  <img src="docs/assets/chair_g2.png" width="96">
  <img src="docs/assets/chair_g3.png" width="96">
  <img src="docs/assets/chair_g4.png" width="96">
</p>

## 用法

### 单张转换

```bash
img2pixelart img=tests/apple.jpg size=96
```

输出在当前目录的 `result.png`。

### 裁边

去掉图片四周空白区域：

```bash
img2pixelart crop-padding tests/apple.jpg
# → tests/apple_no_padding.jpg
```

### 参数扫网（multirun）

多个参数值用逗号分隔，`-m` 开启 multirun，结果自动拼成 `combined.png`：

```bash
# 渐变层次 × 抖动风格 × 描边，81 种组合
img2pixelart -m \
  img=tests/apple.jpg \
  perceive.ramp_steps=5,7,9 \
  render.silhouette_dark_step=0,1,2 \
  render.internal_outline_dark_steps=0,1,2 \
  render.pattern_style=ordered,diagonal,clustered \
  size=96
```

### 并行扫网

加 Joblib launcher，`n_jobs=-1` 用满所有 CPU：

```bash
img2pixelart -m \
  img=tests/apple.jpg \
  perceive.ramp_steps=5,7,9 \
  render.silhouette_dark_step=0,1,2 \
  render.internal_outline_dark_steps=0,1,2 \
  render.pattern_style=ordered,diagonal,clustered \
  size=96 \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=-1
```

## 关键参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `size` | 输出像素画尺寸 | 96 |
| `perceive.ramp_steps` | 每色相的明度阶梯数 | 5 |
| `perceive.requested_groups` | 色相族数量 | 2 |
| `render.pattern_style` | 抖动风格：`ordered` / `diagonal` / `clustered` | ordered |
| `render.silhouette_dark_step` | 外轮廓暗化级数（0=无） | 0 |
| `render.internal_outline_dark_steps` | 内部描边暗化级数（0=无） | 1 |
| `render.dither_fraction_min` | 抖动区域下限 | 0.18 |
| `render.dither_fraction_max` | 抖动区域上限 | 0.82 |

完整参数见 `img2pixelart/conf/` 目录下各阶段 YAML 文件。
