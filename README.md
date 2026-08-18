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

| 原图 | 96px | 64px | 48px | 32px |
|---|---|---|---|---|
| ![banana](docs/assets/banana_orig.png) | ![banana-96](docs/assets/banana_96.png) | ![banana-64](docs/assets/banana_64.png) | ![banana-48](docs/assets/banana_48.png) | ![banana-32](docs/assets/banana_32.png) |
| ![chair](docs/assets/chair_orig.png) | ![chair-96](docs/assets/chair_96.png) | ![chair-64](docs/assets/chair_64.png) | ![chair-48](docs/assets/chair_48.png) | ![chair-32](docs/assets/chair_32.png) |

## 用法

### 图形界面

```bash
img2pixelart-ui
```

打开图片后，可调整完整的感知、结构和渲染参数，加载 `.txt` 调色板，
自动刷新预览并将结果保存为 PNG。

### 单张转换

```bash
img2pixelart img=docs/assets/banana_orig.png size=96
```

输出在当前目录的 `result.png`。

指定调色板后，各色相族的明度 ramp 会先匹配到调色板最近色；抖动、阴影和轮廓
都沿匹配后的 ramp 渲染，因此所有非透明像素都来自调色板：

```bash
img2pixelart img=docs/assets/banana_orig.png palette=palette/resurrect-64.txt size=96
```

### 裁边

去掉图片四周空白区域：

```bash
img2pixelart crop-padding docs/assets/banana_orig.png
# → docs/assets/banana_no_padding.png
```

### 参数扫网（multirun）

多个参数值用逗号分隔，`-m` 开启 multirun，结果自动拼成 `combined.png`：

```bash
# 渐变层次 × 抖动风格 × 描边，81 种组合
img2pixelart -m \
  img=docs/assets/banana_orig.png \
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
  img=docs/assets/banana_orig.png \
  perceive.ramp_steps=5,7,9 \
  render.silhouette_dark_step=0,1,2 \
  render.internal_outline_dark_steps=0,1,2 \
  render.pattern_style=ordered,diagonal,clustered \
  size=96 \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=-1
```

### 视觉 A/B 基线（行为敏感变更）

Issue [#1](https://github.com/asinkLuno/img2pixelart/issues/1) 的基准由受版本控制的
`tools/visual_regression.py` 固化。它覆盖 `tests/cup.png`、
`tests/cup_no_padding.png`、`docs/assets/banana_orig.png` 与
`docs/assets/chair_orig.png`，像素画在 `48 / 64 / 96` 方格运行，ASCII 同时覆盖
`rows=40 / 60`。生成物仅落在（并被 Git 忽略的）
`outputs/ab/<topic>/<UTC timestamp>/`。

```bash
# 当前实现的完整像素画 + ASCII 基线矩阵
uv run python tools/visual_regression.py baseline

# 每个命令生成 A/B 的 result.png、调试 palette、comparison.png 和 metrics.json
uv run python tools/visual_regression.py ab-1  # bayer vs pattern/ordered
uv run python tools/visual_regression.py ab-2  # bilateral → mean-shift vs mean-shift only
uv run python tools/visual_regression.py ab-3  # fixed Canny vs ratio×Otsu Canny
```

P2 的 B 侧在临时包副本中应用严格的实验补丁，不修改工作树；若目标实现移动，补丁会
显式失败而不是悄悄比较了错误的算法。评审时打开 `comparison.png`，再读取
`metrics.json` 的逐像素差异比例、平均通道差异和 `05_palette` /
`22_palette_strip` 一致性。数值用于定位差异，结论仍以 **B 相对 A 没有可感知质量
回退** 的人工评审为准；将 `AB-N: 结论 / 倾向 / 证据（输出路径或贴图）` 作为 issue
评论。ASCII 矩阵是基线覆盖，当前三个 AB 仅比较像素画流水线。

## 关键参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `size` | 输出像素画尺寸 | 96 |
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
