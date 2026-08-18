# AGENTS.md

`img2pixelart` 是一个 Python 3.12+ 图像处理工具：将图片转换为像素画或 ASCII
字符画，并提供 PyQt6 图形界面。包使用 `uv` 管理依赖，以 Hydra 管理运行配置。
远程仓库是 [`asinkLuno/img2pixelart`](https://github.com/asinkLuno/img2pixelart)，
默认分支为 `master`。

> `AGENTS.md` 是本文件（`CLAUDE.md`）的符号链接。修改项目协作指引时只编辑本文件，
> 两个入口会自动保持一致。

## 开发环境与常用命令

- 运行环境：Python `>=3.12`（版本见 `.python-version`）。
- 依赖锁定：`uv.lock`；新增、升级或删除依赖后必须用 `uv lock` 更新它。
- 初次安装开发环境：`uv sync --group dev`。
- 不要手动编辑 `.venv/`、`uv.lock`，也不要以 `pip install` 绕过 uv 的锁定环境。

```bash
# 运行全部测试
uv run pytest

# 静态质量检查（提交前至少运行与改动相关的检查）
uv run ruff check .
uv run ruff format --check .
uv run ty check

# 自动格式化 / 修复；先检查 git diff，再决定是否保留改动
uv run ruff check --fix .
uv run ruff format .

# 执行仓库定义的全部提交钩子
uv run pre-commit run --all-files

# 构建发行包；发布前必须执行
uv build
```

测试使用 `pytest`，其图片夹具位于 `tests/cup.png` 与
`tests/cup_no_padding.png`。涉及像素处理、alpha、尺寸、配置或 CLI 行为的改动，必须
补充可重复的测试；不要仅凭人工查看 PNG 判断正确性。

## 项目结构与架构边界

| 路径 | 职责 |
| --- | --- |
| `img2pixelart/cli.py` | CLI 入口、Hydra 调度、单图 / ASCII / 裁边命令及 multirun 拼图。 |
| `img2pixelart/perceive.py` | 感知阶段：去噪、色相族、明度 ramp、边缘等源图分析。 |
| `img2pixelart/structure.py` | 结构阶段：目标网格、前景 alpha、边缘与小尺寸清理。 |
| `img2pixelart/render.py` | 渲染阶段：抖动、轮廓与最终 BGR 像素画。 |
| `img2pixelart/fit.py` | 调色板加载、颜色 / ramp 匹配与量化。 |
| `img2pixelart/ascii.py` | ASCII 字符画生成。 |
| `img2pixelart/ui.py` | PyQt6 桌面界面和预览逻辑。 |
| `img2pixelart/config.py` | 配置 schema 辅助结构及跨字段业务校验。 |
| `img2pixelart/conf/` | Hydra 默认配置；参数默认值的唯一来源。 |
| `tests/` | 自动化回归测试及小型受版本控制的测试图片。 |
| `palette/` | 可由 CLI / UI 加载的文本调色板。 |
| `docs/`、`mkdocs.yml` | MkDocs 文档及站点导航。 |
| `.github/workflows/` | `master` 推送时部署文档；推送 `v*` tag 时发布 PyPI。 |

流水线的稳定顺序是 **perceive → structure → render**。各阶段应只消费前一阶段的
明确产物，不要为了快捷把后续渲染决策塞回感知或结构阶段。共享数据的新增与变更要同时
更新调用方、类型 / 文档说明和测试。

## 配置、CLI 与图像约定

- 默认参数放在 `img2pixelart/conf/**/default.yaml`；`config.py` 负责验证，而不是
  在 Python 代码中再维护第二套默认值。
- 新增配置项时，完成以下闭环：默认 YAML、`config.py` 的缺失项与取值校验、使用该项的
  调用链、README / 文档中的公开说明，以及覆盖默认值的测试。
- CLI 用 Hydra 解析覆写参数；从代码取得文件路径时使用
  `hydra.utils.to_absolute_path()`，因为 Hydra 会将进程目录切换到输出目录。
- 普通像素画转换将 `result.png` 和调试中间产物写入
  `outputs/single/<timestamp>/`；`-m` 参数扫网写入
  `outputs/multirun/<timestamp>/`，并由回调生成 `combined.png`。ASCII 输出为
  `result_ascii.txt`。这些运行产物不得提交。
- OpenCV 图像数组使用 **BGR / BGRA** 通道顺序。输出透明度必须以独立 alpha 通道保留；
  不能将 RGBA 与 BGRA 混用，也不能把透明背景错误填充为黑色主体。
- 颜色量化、抖动、连通域处理等需要确定性结果的算法应显式传递随机种子，避免引入无种子
  的随机行为。

常用手工冒烟命令：

```bash
uv run img2pixelart img=tests/cup.png width=80 height=45
uv run img2pixelart ascii img=tests/cup.png ascii.rows=40
uv run img2pixelart crop-padding tests/cup.png
uv run img2pixelart-ui
```

GUI 改动至少要完成一次打开测试图片、调整参数、刷新预览及保存 PNG 的人工验证；算法改动
应优先使用无头 `pytest` 覆盖其核心行为。

## 文档、资产与生成文件

- 用户可见的 CLI、配置、调色板或行为变化须同步更新 `README.md`；算法原理和示例更新
  放在 `docs/`，新增页面还要更新 `mkdocs.yml` 导航。
- 文档图片仅提交能稳定复现且确有说明价值的资产；不要把临时实验图片、截图或 Hydra 输出
  放进仓库。
- `outputs/`、`tmp/`、`site/`、`dist/`、`.venv/`、缓存目录及 `tests/*.png` 默认被
  忽略。若需要新增测试图，先通过 `git add -f tests/<name>.png` 明确纳入版本控制，并在
  测试中引用它。
- 不编辑 `dist/` 中已有 wheel / sdist；由 `uv build` 重新生成。不要提交 `__pycache__`、
  工具缓存或本机 IDE 状态。

## Git 与 GitHub 协作

当前工作树可能包含其他任务的未提交改动。开始工作和提交前均执行
`git status --short`；只暂存本任务文件，禁止用 `git add -A`、`git reset --hard`、
`git clean -fd` 或覆盖他人未提交内容。

仓库仍沿用双账号职责：

| 账号 | 职责 |
| --- | --- |
| `asinkLuno` | 开发、提交、推送功能分支、创建 PR、处理 review。 |
| `starmountain1997` | 创建和管理 issue、review、批准与 squash 合并 PR。 |

- 代码提交、推送和 `gh pr create` 必须使用 `asinkLuno`。
- issue、review、approve / request changes、合并必须使用 `starmountain1997`。
- 任何 GitHub 操作前先确认身份：`gh api user --jq .login`。
- 不直接向 `master` 提交。功能分支从最新 `origin/master` 创建，PR 的 base 必须为
  `master`，仓库为 `asinkLuno/img2pixelart`。
- Git commit 使用仓库现有身份；若尚未配置，提交前确认 `git config user.name` 与
  `git config user.email`，不要擅自改写仓库级身份。
- 提交信息使用清晰的 Conventional Commit 风格，例如
  `feat(ascii): improve diagonal edge selection`、
  `fix(render): preserve alpha on palette quantization`。

推荐开发与提交流程：

```bash
# 1. 以开发者账号从干净的 master 开始
gh auth switch --user asinkLuno
gh api user --jq .login
git fetch origin
git switch master
git pull --ff-only origin master
git switch -c feat/short-description

# 2. 实现、验证，只暂存本任务文件
git status --short
uv run pytest
git add img2pixelart/... tests/... README.md
git commit -m "feat(scope): concise description"

# 3. 推送并请求 review
git push -u origin feat/short-description
gh pr create --repo asinkLuno/img2pixelart --base master \
  --head feat/short-description --title "feat(scope): concise description" \
  --body "Closes #N\n\nValidation: uv run pytest" \
  --reviewer starmountain1997
```

`starmountain1997` 审核通过后使用 squash 合并并删除分支：

```bash
gh auth switch --user starmountain1997
gh pr view <number> --repo asinkLuno/img2pixelart --json reviewDecision --jq .reviewDecision
gh pr merge <number> --repo asinkLuno/img2pixelart --squash --delete-branch
```

发布使用版本标签：先在目标提交上完成 `uv build` 和测试，再推送形如 `v0.1.3` 的标签；
`.github/workflows/publish.yml` 会通过 PyPI Trusted Publishing 自动发布。不要手动上传
本地 `dist/` 文件，也不要为了发布修改 `master` 上的工作流触发条件。
