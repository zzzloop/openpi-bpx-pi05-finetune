# BPX π0.5 真机数据微调工程

两张 48 GiB GPU（物理 GPU `2,3`）的简洁训练命令见
[`TRAINING_2X48GB_QUICKSTART.md`](TRAINING_2X48GB_QUICKSTART.md)。

多任务 V2 的去冗余重建、任务均衡归一化、训练及分阶段磁盘清理见
[`BPX_MULTITASK_V2_REBUILD.md`](BPX_MULTITASK_V2_REBUILD.md)。

本目录把已经对齐的 BPX 遥操 HDF5 数据转换为 OpenPI 使用的 LeRobot v2.1 数据集，并提供 π0.5 全参数微调与 JAX LoRA 微调配置。范围只包括数据检查、转换、训练和模型输出适配；不包含实机控制接口，也不包含仿真。

实现基于 OpenPI 提交 `15a9616a00943ada6c20a0f158e3adb39df2ccac`，遵循官方的“转换为 LeRobot → 定义机器人输入/输出变换 → 计算 norm stats → 微调 → policy server”流程：

- [OpenPI README：自定义数据微调](../../README.md)
- [OpenPI UR5 自定义机器人示例](../ur5/README.md)
- [服务器数据、清洗、归一化与训练控制台](SERVER_DATA_TRAINING_STUDIO.md)
- [OpenPI 归一化说明](../../docs/norm_stats.md)
- [OpenPI 远程推理说明](../../docs/remote_inference.md)
- [OpenPI 固定的 LeRobot v2.1 实现](https://github.com/huggingface/lerobot/blob/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5/lerobot/common/datasets/lerobot_dataset.py)

## 1. 已确认的训练定义

### 1.1 原始 23 维关节顺序

HDF5 中 `observations/qpos` 和 `action` 必须严格使用下列顺序：

| 原始索引 | 关节 |
|---:|---|
| 0 | `FoldingModularJoint02_Joint` |
| 1 | `FoldingModularJoint03_Joint` |
| 2 | `Trunk_Joint` |
| 3–9 | `ArmL02_Joint` … `ArmL08_Joint` |
| 10–11 | `JawBlock01_Joint`, `JawBlock02_Joint` |
| 12–18 | `ArmR02_Joint` … `ArmR08_Joint` |
| 19–20 | `JawBlock03_Joint`, `JawBlock04_Joint` |
| 21–22 | `Head02_Joint`, `Head03_Joint` |

语义：

- `observations/qpos[t]`：机器人在时刻 `t` 的实际反馈位置。
- `action[t]`：遥操系统在时刻 `t` 下发的绝对目标位置。
- 四路图像、qpos、action 已在采集端按固定 30 Hz 对齐。
- 两个腰部关节和躯干在本阶段冻结。
- 头部由策略主动控制；左眼图像会随头部动作变化。
- 每个实机夹爪只有一个物理自由度。HDF5 将该物理值平均分在两个 JawBlock 槽位中：例如左夹爪真机值为 `0.84` 时，`raw[10]=0.42`、`raw[11]=0.42`，两者之和才是真机的一维值。

### 1.2 模型使用的 18 维有效向量

转换器删除冻结的原始索引 `0:3`，并把每个重复夹爪对求和为一个物理自由度：

```text
effective[0:7]    = raw[3:10]          # 左臂 7
effective[7]      = raw[10] + raw[11]  # 左夹爪物理 1-DoF
effective[8:15]   = raw[12:19]         # 右臂 7
effective[15]     = raw[19] + raw[20]  # 右夹爪物理 1-DoF
effective[16:18]  = raw[21:23]         # 主动头部 2
```

π0.5 内部动作维度仍是 32，OpenPI 的 `PadStatesAndActions` 会把 18 维补到 32；不修改基础模型结构。

推理输出经过 `BpxOutputs` 后：

1. 截取模型前 18 维；
2. 将每个物理夹爪输出除以 2 后分别写入对应两个 JawBlock 槽，保证两槽之和等于模型输出；
3. 在最前面插入固定腰部值 `[0, 0, 0]`；
4. 返回原始顺序的 `(action_horizon, 23)`。

如果后续部署使用的腰部保持姿态不是零位，只需给 `BpxOutputs.waist_position` 传入对应的三个固定值；不需要重新训练。

### 1.3 动作表示

原始 HDF5 保存绝对目标位置。训练数据管线对以下维度转换为相对当前 qpos 的 delta action：

- 左臂 7 维；
- 右臂 7 维；
- 头部 2 维。

两个物理夹爪维度保持绝对值。policy server 输出时按相反顺序恢复成绝对目标，再拆分并恢复为 23 维。

30 Hz 下使用 `action_horizon=30`，即每个训练目标覆盖约 1 秒。部署端可以只执行 chunk 的前若干步后重新规划，不要求开环执行完 30 步。

## 2. 图像定义

原始 HDF5 含四路 `(T, 360, 640, 3) uint8` 图像：

```text
observations/images/left_eye
observations/images/right_eye
observations/images/left_wrist
observations/images/right_wrist
```

训练使用：

| HDF5 | π0.5 图像槽 |
|---|---|
| `left_eye` | `base_0_rgb` |
| `left_wrist` | `left_wrist_0_rgb` |
| `right_wrist` | `right_wrist_0_rgb` |
| `right_eye` | 丢弃 |

LeRobot 保留完整 `360×640` RGB 图像，不在数据转换阶段裁剪。OpenPI 使用官方 `resize_with_pad` 等比例缩放到 `224×224`：`360×640` 的有效图像约缩放为 `126×224`，再在上下补边，不会裁掉左右视野。训练与推理必须采用同一处理。

当前默认按 RGB 解释。转换前应导出预览：

```bash
uv run examples/bpx/preview_bpx_images.py \
  --episode-path datasets/cleaned_bpx/episode_0.hdf5 \
  --output artifacts/bpx_camera_preview_rgb.png \
  --image-color-order rgb
```

如果红蓝明显颠倒，再生成 BGR 版本比较：

```bash
uv run examples/bpx/preview_bpx_images.py \
  --episode-path datasets/cleaned_bpx/episode_0.hdf5 \
  --output artifacts/bpx_camera_preview_bgr.png \
  --image-color-order bgr
```

最终转换时必须使用确认后的同一颜色选项。

## 3. 数据采集与筛选要求

### 3.1 HDF5 schema

每条成功示范一个 `episode_*.hdf5`：

```text
action                                      (T, 23) float32
observations/qpos                           (T, 23) float32
observations/images/left_eye                (T, 360, 640, 3) uint8
observations/images/right_eye               (T, 360, 640, 3) uint8
observations/images/left_wrist              (T, 360, 640, 3) uint8
observations/images/right_wrist             (T, 360, 640, 3) uint8
```

可选字段：

```text
task       UTF-8 string
success    bool
```

如果 `task` 不存在，转换器注入：

```text
Place all the colored blocks on the table into the bucket.
```

### 3.2 成功与失败数据

首版训练只使用成功 episode。OpenPI 当前微调目标是动作模仿损失，没有“这是失败动作”的监督信号；未标注失败轨迹会被当成正确行为学习。

处理规则：

- 完整失败且未恢复：不要放进训练目录，单独归档用于评测和故障分析。
- 中途出现偏差、随后通过合理动作恢复并最终完成任务：可以作为成功示范保留。
- 人工急停、通信异常、相机掉帧、动作越界：直接剔除。
- 不要为了增加数量重复完全相同的轨迹。

### 3.3 建议覆盖范围

几百条成功示范应覆盖变化，而不只是重复同一摆放：

- 方块颜色、数量、位置、朝向和相互遮挡；
- 桶在允许工作区内的位置和朝向；
- 左右夹爪选择、跨手操作和双臂避让；
- 初始双臂与头部姿态；
- 正常照明变化、桌面背景和相机轻微扰动；
- 抓空后重新定位、物体滑落后再次抓取等最终成功的恢复动作。

当前已确认的 59 条成功 episode 全部用于首轮训练闭环，不自行删除或划出其中任何 episode。独立评测应使用另行采集且不参加训练的真实场景；如果以后确需从示教中留出 episode，必须先确认，且绝不能随机拆分同一轨迹的帧。

## 4. 服务器环境

OpenPI 官方主要测试 Ubuntu 22.04。本项目按 Ubuntu 22.04、256 GB 内存和约 196 GB 总显存设计。

服务器用户名固定为 `kemove`，工作区固定为 `~/zzk_data`，仓库根目录固定为：

```text
/home/kemove/zzk_data/openpi-bpx-pi05-finetune
```

若服务器尚未安装基础工具且当前账号具有 `sudo` 权限，先执行：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs curl ffmpeg
git lfs install
nvidia-smi
```

`nvidia-smi` 必须能正常列出 GPU；不要在没有确认服务器驱动/CUDA 管理策略的情况下自行更换 NVIDIA 驱动。

先在服务器上完成 GitHub 私密仓库认证，然后连同子模块一起克隆并进入仓库根目录：

```bash
mkdir -p ~/zzk_data
cd ~/zzk_data
git clone --recurse-submodules https://github.com/zzzloop/openpi-bpx-pi05-finetune.git
cd openpi-bpx-pi05-finetune
REPO_ROOT=~/zzk_data/openpi-bpx-pi05-finetune
test "$(pwd -P)" = "$(realpath "$REPO_ROOT")"
git submodule update --init --recursive
```

后续所有相对路径命令都必须从该仓库根目录执行。先按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/) 安装 `uv`；Linux 可使用官方安装器：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

显式使用仓库固定的 Python 3.11，并按锁文件安装依赖：

```bash
uv python install 3.11
uv venv --python 3.11
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv run python -c "import sys; assert sys.version_info[:2] == (3, 11); print(sys.version)"
```

训练使用 JAX。全参数微调是主方案；LoRA 用作低成本对照和快速排错。OpenPI 的 PyTorch 路径需要先转换 checkpoint，且当前不支持 JAX 路径已有的 LoRA、FSDP 和 EMA 组合，因此本 BPX 首版不把 PyTorch 作为主训练路径。

## 5. 放置 π0.5 base

`pi05_base` 不进入 Git。当前本机源目录约为 11.59 GiB、33 个文件；必须把整个目录（包括 `params/`、`assets/`、根目录隐藏元数据和 README）手工复制到服务器：

```text
/home/kemove/zzk_data/openpi-bpx-pi05-finetune/checkpoints/pretrained/pi05_base/
├── params/
├── assets/
├── .mdl
├── .msc
├── .mv
└── README.md
```

两个 BPX 配置都使用相对路径：

```text
./checkpoints/pretrained/pi05_base/params
```

先在仓库根目录创建父目录：

```bash
mkdir -p checkpoints/pretrained
```

复制完成后的正确结构是 `checkpoints/pretrained/pi05_base/params`，不能出现错误的双层结构 `checkpoints/pretrained/pi05_base/pi05_base/params`。复制后执行：

```bash
test -d checkpoints/pretrained/pi05_base/params
test -d checkpoints/pretrained/pi05_base/assets
test -f checkpoints/pretrained/pi05_base/params/_CHECKPOINT_METADATA
test -f checkpoints/pretrained/pi05_base/params/manifest.ocdbt
du -sh checkpoints/pretrained/pi05_base
```

不要只复制 checkpoint 中的某几个文件，也不要把训练输出写入 `checkpoints/pretrained/`。BPX 训练 checkpoint 会写到独立的：

```text
checkpoints/pi05_bpx_full/...
checkpoints/pi05_bpx_lora/...
```

## 6. 放置并验证已清洗数据

当前 59 条训练轨迹已经完成清洗和逐元素审计，不要在服务器重新清洗。当前本机清洗目录约为 106.30 GiB、120 个文件；必须把整个 `bpx_real_hdf5_cleaned` 目录手工复制到服务器，并在目标端命名为 `cleaned_bpx`：

```text
/home/kemove/zzk_data/openpi-bpx-pi05-finetune/datasets/cleaned_bpx/
├── episode_0.hdf5 ... episode_58.hdf5
├── cleaning_audit.csv
├── cleaning_summary.json
└── .cleaning_records/
```

先在仓库根目录创建父目录：

```bash
mkdir -p datasets
```

复制完成后的正确结构是 `datasets/cleaned_bpx/episode_0.hdf5`，不能出现错误的双层结构 `datasets/cleaned_bpx/bpx_real_hdf5_cleaned/episode_0.hdf5`。以下内容必须全部存在：

```text
datasets/cleaned_bpx/cleaning_audit.csv
datasets/cleaned_bpx/cleaning_summary.json
datasets/cleaned_bpx/.cleaning_records/
datasets/cleaned_bpx/episode_*.hdf5
```

复制后在仓库根目录执行完整性验收：

```bash
test "$(find datasets/cleaned_bpx -maxdepth 1 -type f -name 'episode_*.hdf5' | wc -l)" -eq 59
test "$(find datasets/cleaned_bpx/.cleaning_records -maxdepth 1 -type f -name 'episode_*.json' | wc -l)" -eq 59
test -f datasets/cleaned_bpx/cleaning_audit.csv
test -f datasets/cleaned_bpx/cleaning_summary.json
du -sh datasets/cleaned_bpx
uv run python - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("datasets/cleaned_bpx/cleaning_summary.json").read_text(encoding="utf-8"))
assert summary["episode_count"] == 59, summary["episode_count"]
assert summary["frame_count"] == 67_455, summary["frame_count"]
assert summary["repair_count"] == 439, summary["repair_count"]
assert summary["rejected_candidate_count"] == 0, summary["rejected_candidate_count"]
print("PASS: 59 episodes, 67,455 frames, 439 audited qpos repairs")
PY
```

只需要手工复制上述 `pi05_base` 和 `cleaned_bpx` 两个大目录。以下内容不要从本机复制：原始 E 盘数据、URDF 资源、LeRobot 转换结果、norm stats、训练 checkpoint 和 WandB 日志；这些内容不属于服务器初始化输入，或会在服务器训练流程中重新生成。

`clean_bpx_hdf5.py` 只保留用于追溯现有数据是怎样生成的；当前批次不得再次执行它，除非先确认发现了新的数据错误并重新审计。

在转换前运行服务器总门禁。默认要求仓库所在文件系统至少还剩 500 GiB，用于 LeRobot 视频、环境缓存以及带优化器和 EMA 的多个 full checkpoint：

```bash
df -hT .
df -ih .
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
ffmpeg -version | head -n 1
uv run examples/bpx/preflight_bpx_server.py --repo-root . --min-free-gib 500
```

该门禁要求 Ubuntu/Linux、Python 3.11、完整子模块、FFmpeg、NVIDIA 驱动、JAX CUDA backend、可用磁盘、完整 base 模型以及 59 条/67,455 帧清洗数据全部同时通过。若磁盘确实无法提供 500 GiB，不要直接降低阈值；应先根据 LeRobot 实际输出和 checkpoint 保留策略重新核算容量。

先运行仓库门禁和 BPX 配置加载检查：

```bash
uv run ruff check examples/bpx src/openpi/policies/bpx_constants.py \
  src/openpi/policies/bpx_policy.py src/openpi/policies/bpx_policy_test.py \
  src/openpi/training/bpx_config_test.py src/openpi/training/config.py
uv run ruff format --check examples/bpx src/openpi/policies/bpx_constants.py \
  src/openpi/policies/bpx_policy.py src/openpi/policies/bpx_policy_test.py \
  src/openpi/training/bpx_config_test.py src/openpi/training/config.py
uv run pytest examples/bpx src/openpi/policies/bpx_policy_test.py \
  src/openpi/training/bpx_config_test.py -q
```

再运行严格 HDF5 校验：

```bash
uv run examples/bpx/validate_bpx_hdf5.py \
  --raw-dir datasets/cleaned_bpx \
  --action-horizon 30
```

校验器严格检查：

- 所有模态帧数一致；
- qpos/action 均为 `(T,23)` 浮点数且无 NaN/Inf；
- 四路图像均为 `(T,360,640,3) uint8`；
- 腰部 action 为零；
- 每侧夹爪的两个 action 严格相等，转换时将两值求和为一个物理夹爪维度；
- 每条 episode 至少有 30 帧；
- 报告 action 到 qpos 的最佳反馈延迟和每一有效维度的范围、标准差。

当前 59 条实机数据均为最终成功示教。清洗前共 67,455 帧、约 37.48 分钟。本批次默认参数共修复 439 个 `qpos` 元素、拒绝候选数为 0；对清洗结果再次只读扫描时修复候选数为 0。全量 `action` 与原始数据严格一致，原始/清洗 `qpos` 的差异集合与 `cleaning_audit.csv` 逐点一致。额外只读统计显示 action 完全重复的相邻帧比例为 0，按各维 `q01–q99` 范围归一化后步长不超过 `1e-3` 的近静止转移约占 0.14%，且没有 episode 存在超过 1 秒的首尾近静止段，因此当前不做 idle 过滤。action 到 qpos 的反馈延迟只用于诊断；训练标签仍使用采集端已对齐的同帧 `observation[t]` 与遥操目标 `action[t]`，不额外平移。

## 7. 转换为 LeRobot v2.1

在仓库根目录设置相对数据缓存：

```bash
export HF_LEROBOT_HOME=./datasets/lerobot
```

先做单条 episode pilot，确认异步图像写入、LeRobot v2.1 数据落盘和读取均正常：

```bash
uv run examples/bpx/convert_bpx_data_to_lerobot.py \
  --raw-dir datasets/cleaned_bpx \
  --repo-id bpx/bpx_blocks_pilot \
  --episodes 0 \
  --image-color-order rgb
uv run examples/bpx/validate_bpx_lerobot.py \
  --raw-dir datasets/cleaned_bpx \
  --repo-id bpx/bpx_blocks_pilot \
  --episodes 0
```

pilot 通过后进行首次全量转换：

```bash
uv run examples/bpx/convert_bpx_data_to_lerobot.py \
  --raw-dir datasets/cleaned_bpx \
  --repo-id bpx/bpx_blocks \
  --image-color-order rgb
```

输出：

```text
datasets/lerobot/bpx/bpx_blocks/
```

只有在明确需要覆盖已有转换结果时才使用：

```bash
uv run examples/bpx/convert_bpx_data_to_lerobot.py \
  --raw-dir datasets/cleaned_bpx \
  --repo-id bpx/bpx_blocks \
  --image-color-order rgb \
  --overwrite
```

转换器会流式读取每帧，不会把数百条 episode 的全部图像一次性载入内存。默认只保存在本地，不上传 Hugging Face Hub。若确需上传，应使用私有数据集并确认数据合规后再加 `--push-to-hub`；默认 `private=true`。

全量转换完成后执行源数据对照验收：

```bash
uv run examples/bpx/validate_bpx_lerobot.py \
  --raw-dir datasets/cleaned_bpx \
  --repo-id bpx/bpx_blocks \
  --expected-episodes 59 \
  --expected-frames 67455
```

该命令逐帧比较 18D state/action 与清洗 HDF5，检查 episode/frame/index/timestamp/task 顺序，并为每条 episode 抽取首、中、末帧的三路 RGB 图像与源图比较。它同时确认右眼不存在、腰部已删除、两侧夹爪等于原始 pair 之和、头部两维仍在。

如果修改 `repo-id`，必须同步修改 `src/openpi/training/config.py` 中两个 BPX 配置的 `repo_id` 和 `asset_id`。

## 8. 训练配置

配置位于 `src/openpi/training/config.py`：

### `pi05_bpx_full`

- π0.5 flow-matching；
- action horizon 30；
- 18 维有效状态/动作，模型内部 pad 到 32；
- 全参数微调；
- batch size 64；
- 20,000 steps；
- EMA 0.999；
- 本地持久化监控开启：每 100 step 记录标量、每 500 step 绘制训练总览、每 1,000 step 保存三相机和动作目标图；
- 标准全参数方案；OpenPI 官方要求单卡显存大于约 70 GB，必须先通过目标服务器显存门禁。

### `pi05_bpx_full_low_mem`

- 模型、数据、AdamW、学习率、action horizon 和全参数训练范围均与 `pi05_bpx_full` 相同；
- 默认 global batch size 2，供两张 48 GiB GPU 以 FSDP 2 运行；
- 关闭 EMA，不在训练状态中维护额外的完整参数副本；
- checkpoint 自动保存当前训练参数，推理和 23D 输出恢复流程不变；
- 这是当前两张 48 GiB GPU 上的最后一级 full-finetune 显存试验，仍必须先通过 1-step 门禁，不能保证必然可用。

### `pi05_bpx_lora`

- PaliGemma 与 action expert 均使用 LoRA；
- action horizon 和数据变换与 full 完全一致；
- batch size 128；
- 30,000 steps；
- 不使用 EMA；
- 使用与 full 相同的本地持久化监控；
- 用于快速验证、消融和低成本对照。

三个配置使用同一个 LeRobot 数据集。low-memory full 和 LoRA 均复用标准 full 配置计算出的 norm stats，避免重复解码全部视频。

## 9. 计算归一化统计

每次改变以下任一项后都必须重新计算：

- 训练 episode 集合；
- 关节顺序或维度；
- action 定义；
- delta-action mask；
- 夹爪转换；
- 数据采样频率。

命令：

```bash
export HF_LEROBOT_HOME=./datasets/lerobot
uv run scripts/compute_norm_stats.py --config-name pi05_bpx_full
```

输出位于：

```text
assets/pi05_bpx_full/bpx/bpx_blocks/norm_stats.json
```

π0.5 使用分位数归一化。生成后打印并检查全部 18 维 state/action 的 `q01`、`q99`、`mean` 和 `std`：

```bash
uv run examples/bpx/inspect_bpx_norm_stats.py \
  --norm-stats-path assets/pi05_bpx_full/bpx/bpx_blocks/norm_stats.json
```

- `q01` 与 `q99` 几乎相同通常表示该维度没有被充分示教；
- 极小范围可能放大噪声并导致 loss 发散；
- 不应复用其他机器人（ALOHA、DROID、ARX）的 norm stats；
- 不应为冻结腰部计算 norm stats，这也是将其删除的原因。

## 10. 全参数微调

建议先进行 1,000–5,000 step 的工程验证，再运行完整训练。

### 10.1 训练前确定 JAX 设备、FSDP 与 WandB

任何 full pilot 之前都必须先执行：

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
uv run python - <<'PY'
import jax

print("backend:", jax.default_backend())
print("device_count:", jax.device_count())
print("devices:", jax.devices())
assert jax.default_backend() == "gpu"
assert 64 % jax.device_count() == 0, "global batch 64 must be divisible by JAX device count"
PY
```

OpenPI 官方估计全参数微调单卡需要超过 70 GB 显存。约 196 GB 是总显存，并不能证明单张 GPU 足够：

- 若每张可见 GPU 均有至少约 80 GB，可先设 `FSDP_DEVICES=1`，此时多卡之间做数据并行；
- 若单卡不足约 80 GB，应选择 `FSDP_DEVICES=2`、`4` 等有效分片数，使一个 FSDP 组的总显存足够；
- `FSDP_DEVICES` 必须是 `jax.device_count()` 的正因数，且不得大于设备数；
- global batch 64 必须能被 `jax.device_count()` 整除。若不整除，应通过 `CUDA_VISIBLE_DEVICES` 选择合适数量的 GPU；不要未经重新审计直接修改 batch size。

例如 4 张约 48 GB GPU 可先尝试：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FSDP_DEVICES=2
```

根据服务器真实 GPU 调整上述值，并再次运行 `preflight_bpx_server.py`。不要机械照抄示例。

```bash
uv run examples/bpx/preflight_bpx_server.py \
  --repo-root . \
  --min-free-gib 500 \
  --fsdp-devices "$FSDP_DEVICES" \
  --batch-size 64
```

这次门禁会同时验证所选 `FSDP_DEVICES` 是当前 JAX device count 的正因数。

训练默认启用 Weights & Biases。二选一完成配置：

```bash
# 方案 A：在线记录，推荐
uv run wandb login
uv run wandb status

# 方案 B：无外网时离线记录
export WANDB_MODE=offline
```

如果明确完全不需要 WandB，则给每条训练命令追加 `--no-wandb-enabled`；否则无头服务器可能在 `wandb.init` 阶段阻塞。以下命令默认已经完成在线登录或设置了 `WANDB_MODE=offline`。

### 10.2 两张 48 GiB GPU：关闭 EMA 的 full 显存门禁

标准 `pi05_bpx_full` 已在两张 48 GiB GPU、FSDP 2、global batch 2、显存比例 0.99 下 OOM 时，改用独立配置
`pi05_bpx_full_low_mem`。该配置仍更新全部参数，只关闭 EMA。先确认两张卡没有其他计算进程，然后只运行 1 step：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export HF_LEROBOT_HOME=./datasets/lerobot
export WANDB_MODE=offline

uv run examples/bpx/preflight_bpx_server.py \
  --repo-root . \
  --min-free-gib 200 \
  --fsdp-devices 2 \
  --batch-size 2

XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 \
uv run scripts/train.py pi05_bpx_full_low_mem \
  --exp-name=bpx_full_low_mem_gate_bs2 \
  --batch-size=2 \
  --num-train-steps=1 \
  --fsdp-devices=2 \
  --overwrite
```

门禁要求：完成第 1 step，且无 `RESOURCE_EXHAUSTED`、NaN 或 Inf。通过后才运行 300-step 稳定性 pilot：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 \
uv run scripts/train.py pi05_bpx_full_low_mem \
  --exp-name=bpx_full_low_mem_pilot_bs2 \
  --batch-size=2 \
  --num-train-steps=300 \
  --fsdp-devices=2 \
  --overwrite
```

global batch 从标准配置的 64 降到 2 后，20,000 step 的样本量和学习率日程不再与标准 full 等价。300-step pilot
只验证显存、数值稳定性、吞吐、监控和 checkpoint；进入长训前必须依据 pilot 重新审定步数及学习率，不得直接把
20,000 step 当作最终训练量。如果 1-step 仍然 OOM，则当前两张 48 GiB GPU 不支持本工程的全参数训练，转用 LoRA
或更大显存硬件，不再尝试 global batch 1。

### 10.3 运行 5,000-step 标准 full pilot

```bash
export HF_LEROBOT_HOME=./datasets/lerobot
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_bpx_full \
  --exp-name=bpx_blocks_full_pilot \
  --num-train-steps=5000 \
  --fsdp-devices="$FSDP_DEVICES" \
  --overwrite
```

pilot 验收必须同时满足：loss 全程有限且总体下降，无 NaN/Inf；日志中的 global batch 为 64；显存和数据吞吐稳定；checkpoint 能保存；使用第 13 节的真实 HDF5 离线 smoke 能重新加载 checkpoint 并得到正确的 23D 输出。任何一项不通过都先排查，不进入长训。

### 10.4 运行标准 full 长训练

pilot 和真实数据离线 smoke 通过后，沿用已经验证的 `CUDA_VISIBLE_DEVICES` 与 `FSDP_DEVICES`：

```bash
export HF_LEROBOT_HOME=./datasets/lerobot
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_bpx_full \
  --exp-name=bpx_blocks_full \
  --fsdp-devices="$FSDP_DEVICES" \
  --overwrite
```

要求：

- `jax.device_count() % FSDP_DEVICES == 0`；
- `64 % jax.device_count() == 0`；
- `--overwrite` 会覆盖同名实验目录，只能用于明确重跑；
- 续训使用 `--resume`，不要同时使用 `--overwrite`。

断点续训命令：

```bash
export HF_LEROBOT_HOME=./datasets/lerobot
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_bpx_full \
  --exp-name=bpx_blocks_full \
  --fsdp-devices="$FSDP_DEVICES" \
  --resume
```

完整训练输出：

```text
checkpoints/pi05_bpx_full/bpx_blocks_full/<step>/
```

## 11. LoRA 微调

先完成第 9 节的 full norm stats，然后：

```bash
export HF_LEROBOT_HOME=./datasets/lerobot
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_bpx_lora \
  --exp-name=bpx_blocks_lora \
  --fsdp-devices="$FSDP_DEVICES" \
  --overwrite
```

LoRA 适合快速确认数据链路，但 BPX 与预训练本体在 7-DoF 双臂、双 JawBlock 输出和主动头部动作上存在差异。最终模型优先以全参数微调的实机成功率为准，不应只根据 LoRA 的训练 loss 决定方案。

## 12. 训练监控与 checkpoint 选择

### 12.1 自动保存的监控文件

BPX standard full、low-memory full 和 LoRA 配置默认开启本地持久化监控。即使设置了 `WANDB_MODE=offline` 或使用
`--no-wandb-enabled`，以下内容仍会写入当前实验 checkpoint 目录：

```text
checkpoints/<config>/<exp-name>/monitoring/
├── metrics.jsonl
├── metrics.csv
├── training_overview_latest.png
├── plots/
│   └── training_overview_step_*.png
├── camera_views/
│   └── camera_step_*.png
└── action_targets/
    └── action_target_step_*.png
```

- `metrics.jsonl`：完整机器可读记录，包含 loss、区间 loss 标准差/最小/最大值、30 个 horizon 的 loss、
  gradient/update/parameter norm、学习率、训练吞吐和 UTC 时间；续训时会继续追加。
- `metrics.csv`：便于直接用表格软件检查的标量记录。
- `training_overview_step_*.png`：每 500 step 保存一次六联图，包括 loss 及移动均值、loss 区间、
  gradient/update norm、parameter norm、学习率、吞吐和各 action-horizon 位置的 loss。
- `camera_step_*.png`：每 1,000 step 保存进入模型随机增强前的三路 resize-with-pad 图像，用于发现相机错位、
  黑屏、颜色错误和 padding 异常；训练时模型内部仍会执行随机裁剪/旋转（非腕部）和颜色增强。
- `action_target_step_*.png`：每 1,000 step 分三组保存模型空间中前 18 维动作标签曲线，并使用 BPX 有效关节名标注。
  这里展示的是归一化后的 delta/absolute 混合训练标签，不是模型预测，也不是可直接下发真机的 23D 动作。

同一批图像会同步到 W&B；W&B 还会自动采集 GPU 利用率、显存、温度和系统资源。离线训练结束且恢复网络后可执行：

```bash
uv run wandb sync wandb/offline-run-*
```

默认间隔已经避免在 JIT 热路径中绘图。若确需调整，可在训练命令中显式覆盖：

```bash
--monitoring-plot-interval=500 \
--monitoring-visualization-interval=1000 \
--monitoring-max-samples=2
```

不建议把图像间隔降低到 100 step：频繁把分片 batch 从 GPU 取回并编码 PNG 会降低训练吞吐。

### 12.2 判读要求

至少监控：

- training loss 是否持续下降且无 NaN/Inf；
- 梯度范数是否频繁触发裁剪；
- 数据加载吞吐和 GPU 利用率；
- norm stats 是否存在近零范围维度；
- 生成动作是否在训练分布内；
- 两个夹爪输出在 `BpxOutputs` 后是否严格相同；
- 腰部输出是否为固定保持值；
- 头部动作与左眼画面是否形成合理闭环。

训练循环每 100 step 检查一次全部指标，发现 NaN 或 Inf 会在写入本地/W&B 记录后自动抛错停止。出现以下任一情况也应
立即停止训练并排查：loss 长时间突然抬升且无法恢复；
gradient norm 持续异常增大；吞吐突然明显下降；三相机出现黑屏、串流、颜色反转或视角不一致。

本地曲线和 W&B 只能判断训练健康度。动作目标图不能证明策略已学会任务；checkpoint 的实际能力仍必须通过第 13 节的
真实 HDF5 smoke、离线预测误差以及独立实机场景成功率验证。

不要仅选择最低训练 loss。每隔固定 checkpoint 在未进入训练的场景中评测，记录：

- 完整任务成功率；
- 单块抓取成功率；
- 掉落率、碰撞率和超时率；
- 平均完成时间；
- 不同颜色、位置、数量条件下的分项成功率。

使用另行采集的固定真实场景评测各 checkpoint，并保持场景与重复次数一致。当前 59 条训练 episode 不自行留出或重分组。

## 13. 启动 policy server

先使用清洗 HDF5 中的真实 RGB、qpos 和任务提示加载 checkpoint 并执行一次推理；该 smoke test 会检查 `(30,23)`、有限值、腰部保持值和两对相等的夹爪槽位：

```bash
uv run examples/bpx/smoke_test_bpx_checkpoint.py \
  --config-name pi05_bpx_full \
  --checkpoint-dir checkpoints/pi05_bpx_full/bpx_blocks_full_pilot/<step> \
  --episode-path datasets/cleaned_bpx/episode_0.hdf5 \
  --frame-index 0
```

关闭 EMA 的 full checkpoint 将配置名和目录对应替换为：

```bash
uv run examples/bpx/smoke_test_bpx_checkpoint.py \
  --config-name pi05_bpx_full_low_mem \
  --checkpoint-dir checkpoints/pi05_bpx_full_low_mem/bpx_full_low_mem_pilot_bs2/<step> \
  --episode-path datasets/cleaned_bpx/episode_0.hdf5 \
  --frame-index 0
```

再选择另一条 episode 和中间帧重复一次，防止只验证单一观测。`<step>` 必须替换为 checkpoint 目录中实际保存的数字步骤，不能保留尖括号占位符。

全参数 checkpoint：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_bpx_full \
  --policy.dir=checkpoints/pi05_bpx_full/bpx_blocks_full/<step>
```

关闭 EMA 的全参数 checkpoint：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_bpx_full_low_mem \
  --policy.dir=checkpoints/pi05_bpx_full_low_mem/bpx_full_low_mem_pilot_bs2/<step>
```

LoRA checkpoint：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_bpx_lora \
  --policy.dir=checkpoints/pi05_bpx_lora/bpx_blocks_lora/<step>
```

本工程不实现机器人侧控制。部署项目向 policy 发送：

```text
qpos:   23 维原始顺序，或已经完成腰部删除和夹爪求和的 18 维顺序
images:
  left_eye
  left_wrist
  right_wrist
prompt: 任务指令
```

policy 返回：

```text
actions: (30, 23)
```

其中原始索引 `0:3` 为腰部保持值；每个物理夹爪值已除以 2 并写入两个原始槽位，因此对应两槽之和等于策略的一维夹爪输出；头部动作位于原始索引 `21:23`。

## 14. 文件说明

| 文件 | 作用 |
|---|---|
| `src/openpi/policies/bpx_constants.py` | 唯一的关节顺序、维度、相机、频率和 18↔23 维规则 |
| `src/openpi/policies/bpx_policy.py` | 三路图像映射、输入压缩和输出恢复 |
| `src/openpi/policies/bpx_policy_test.py` | 关节顺序、腰部恢复、夹爪成对和图像映射测试 |
| `src/openpi/training/bpx_config_test.py` | standard full/low-memory full/LoRA 配置加载、transform 顺序、delta 往返和 Pad32 测试 |
| `src/openpi/training/config.py` | `LeRobotBpxDataConfig`、standard full、low-memory full 和 LoRA 配置 |
| `src/openpi/training/monitoring.py` | 本地 JSONL/CSV、训练曲线、三相机和动作目标持久化监控 |
| `src/openpi/training/monitoring_test.py` | 监控写入、无头绘图、续训截断和可视化测试 |
| `examples/bpx/bpx_data.py` | HDF5 schema 与严格校验公共逻辑 |
| `examples/bpx/clean_bpx_hdf5.py` | 不覆盖原始文件的 qpos 毛刺清洗与审计报告 |
| `examples/bpx/BPX_SPIKE_CLEANING_GUIDE.md` | ROS bag/HDF5毛刺审计、清洗、复核与可追溯操作规范 |
| `examples/bpx/evaluate_bpx_bag_server.py` | 通过已启动策略服务对ROS bag做逐关节、逐horizon离线评估和可视化 |
| `examples/bpx/BAG_CHECKPOINT_EVALUATION.md` | combined checkpoint的ROS bag离线评估与真机服务命令 |
| `examples/bpx/build_bpx_multitask_v2_manifest.py` | 从旧固定轨迹中确定性选择代表样本并生成多任务 V2 manifest |
| `examples/bpx/BPX_MULTITASK_V2_REBUILD.md` | V2 去冗余、任务均衡采样、归一化、训练、回退与磁盘清理手册 |
| `examples/bpx/preflight_bpx_server.py` | 服务器软件、GPU/JAX、磁盘、base 与数据总门禁 |
| `examples/bpx/validate_bpx_hdf5.py` | 转换前数据质量检查 |
| `examples/bpx/preview_bpx_images.py` | RGB/BGR 四相机预览 |
| `examples/bpx/convert_bpx_data_to_lerobot.py` | HDF5 → LeRobot v2.1 |
| `examples/bpx/validate_bpx_lerobot.py` | 转换后计数、字段、顺序、18D 数值与 RGB 抽检 |
| `examples/bpx/inspect_bpx_norm_stats.py` | 打印并检查全部 18 维归一化统计 |
| `examples/bpx/smoke_test_bpx_checkpoint.py` | 真实 HDF5 checkpoint 离线推理与 23D 输出契约检查 |
| `examples/bpx/evaluate_bpx_episode_server.py` | 通过已启动policy server回放HDF5，输出真实/预测动作指标、CSV、NPZ和可视化 |

### 14.1 回放 episode 并比较真实动作

policy server保持运行，在服务器另一个终端直接读取清洗HDF5；不需要再次转成LeRobot：

```bash
uv run examples/bpx/evaluate_bpx_episode_server.py \
  --episode-path datasets/cleaned_bpx/episode_0.hdf5 \
  --host 127.0.0.1 \
  --port 8000 \
  --stride 30 \
  --samples-per-frame 1 \
  --output-dir eval_outputs/full_60000_episode0
```

快速检查通过后，用与实机一致的5步重规划间隔做更密集评估：

```bash
uv run examples/bpx/evaluate_bpx_episode_server.py \
  --episode-path datasets/cleaned_bpx/episode_0.hdf5 \
  --host 127.0.0.1 \
  --port 8000 \
  --stride 5 \
  --samples-per-frame 3 \
  --output-dir eval_outputs/full_60000_episode0_stride5
```

输出包括 `metrics.json`、逐关节/逐horizon CSV、原始 `predictions.npz`、三相机抽帧、18维轨迹、单个30步动作块对齐图和总览图。同一路径已存在时脚本默认拒绝覆盖；明确需要重跑时添加 `--overwrite`。

机械臂和头部动作使用弧度，两个夹爪使用线性坐标，因此 `aggregate_mae` 混合了不同单位，只适合做同一数据链的粗略检查。Full/LoRA 横向比较应主要查看 `per_joint_metrics.csv` 的 `normalized_mae`、首步相关性、远期 horizon 误差和动作曲线，不应仅凭总 MAE 选模型。

`episode_0` 已参与训练，因此该结果只能验证数据链、颜色、关节顺序和训练集拟合，不能作为泛化结论。正式比较full与LoRA时应对二者使用完全相同的episode、stride和samples，并分别指定输出目录；实机泛化必须使用从未参与训练的新episode和独立场景成功率。

## 15. 长训练前验收清单

- [ ] OpenPI 提交与本工程记录一致。
- [ ] 子模块、Python 3.11、锁文件依赖、FFmpeg 和 NVIDIA 驱动完整。
- [ ] 服务器 preflight 通过，JAX backend 为 GPU，磁盘剩余不少于审计阈值。
- [ ] `pi05_base/params` 已放在相对目录且未被训练输出覆盖。
- [ ] 训练目录只含成功或最终成功恢复的 episode。
- [ ] 训练使用清洗副本，`cleaning_audit.csv` 已人工抽查且原始数据仍保留。
- [ ] 四路图像、qpos、action 均为 30 Hz 对齐。
- [ ] RGB/BGR 已通过预览人工确认。
- [ ] HDF5 校验器全部通过。
- [ ] Ruff、BPX pytest 和配置加载测试全部通过。
- [ ] 单 episode LeRobot pilot 转换与验收通过。
- [ ] LeRobot `repo_id` 与两个训练配置一致。
- [ ] 全量转换严格验收为 59 条、67,455 帧，18D 数值和 RGB 抽检通过。
- [ ] norm stats 在最终训练数据上重新计算。
- [ ] 18 维 state/action 的 q01、q99、mean、std 已逐维检查。
- [ ] 在 pilot 前根据单卡显存和 JAX device count 确定了有效的 FSDP 分片数。
- [ ] WandB 已登录、设为 offline，或训练命令明确禁用。
- [ ] pilot 的 `monitoring/` 已生成 JSONL、CSV、总览、三相机和动作目标图，并人工检查无异常。
- [ ] 先完成短训练并验证能正常保存、加载和 serve。
- [ ] checkpoint 使用至少两条真实 HDF5 观测完成离线 smoke，输出为 `(30,23)`，腰部、夹爪和头部契约正确。
- [ ] 再运行全参数长训练。
- [ ] checkpoint 通过独立场景实机成功率选择。
