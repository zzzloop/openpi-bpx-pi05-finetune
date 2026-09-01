# BPX 新旧数据统一转换与训练（服务器）

所有命令都从仓库根目录执行。流程不会修改 `/scratch/bpx_datasets/new_datasets_cleaned`，也不会删除旧 59 条数据。

## 1. 环境变量

```bash
cd ~/zzk_data/openpi-bpx-pi05-finetune
uv sync

export REPO_ROOT="$PWD"
export NEW_ROOT=/scratch/bpx_datasets/new_datasets_cleaned
export COMBINED_ROOT=/scratch/bpx_datasets/bpx_pi05_combined_source
export HF_LEROBOT_HOME=/scratch/bpx_datasets/lerobot
export MANIFEST="$COMBINED_ROOT/manifests/combined_episodes.json"
export LEGACY_LEROBOT_ROOT="$REPO_ROOT/datasets/lerobot/bpx/bpx_blocks"
```

## 2. 严格检查新复制的约 200GB 数据

先做快速结构门禁（读取全部 324 个 bag 的索引）：

```bash
uv run --with rosbags==0.11.0 examples/bpx/prepare_bpx_combined_source.py \
  --new-bag-root "$NEW_ROOT"
```

必须看到：

```text
NEW DATASET PASS: bags=324, frames=107586, ... deep=False
```

可选的完整内容门禁会顺序解压全部约 200GB 数据，耗时较长，但不会写入或修改数据：

```bash
uv run --with rosbags==0.11.0 examples/bpx/prepare_bpx_combined_source.py \
  --new-bag-root "$NEW_ROOT" \
  --deep
```

时间紧张时可以跳过这次独立 deep：后续全量 LeRobot 转换会使用同一逐帧解码器检查 8 个 ROS topic、逐帧同时间戳、25D 有限数值、轮子为零、图像 360×640、RGB/BGR 编码和完整载荷；转换后验证再核对全部 18D state/action。独立 deep 额外验证 original/dark/bright 的逐消息亮度变换关系。

## 3. 以已验收的旧 LeRobot 59 条作为前缀

原始 HDF5 只保留了 episode 0 和 50，因此不能作为 59 条来源。先确认旧 LeRobot 数据集存在；全量转换器会把它完整复制到 scratch 的新目标，再追加新 324 条，不会二次编码旧视频：

```bash
test -d "$LEGACY_LEROBOT_ROOT"
mkdir -p "$COMBINED_ROOT/manifests"
```

新数据不再复制一份；在统一源目录内建立只读入口，并核对链接目标：

```bash
if [[ ! -e "$COMBINED_ROOT/new_rosbags" ]]; then
  ln -s "$NEW_ROOT" "$COMBINED_ROOT/new_rosbags"
fi

test "$(readlink -f "$COMBINED_ROOT/new_rosbags")" = "$(readlink -f "$NEW_ROOT")"
export UNIFIED_NEW_ROOT="$COMBINED_ROOT/new_rosbags"
```

生成统一且确定的清单：

```bash
uv run --with rosbags==0.11.0 examples/bpx/prepare_bpx_combined_source.py \
  --new-bag-root "$UNIFIED_NEW_ROOT" \
  --legacy-lerobot-root "$LEGACY_LEROBOT_ROOT" \
  --output-manifest "$MANIFEST"
```

必须看到：

```text
COMBINED DATASET PASS: episodes=383, frames=175041
```

编号固定为：旧数据 `0..58`，新数据 `59..382`；每个新任务按源编号升序，再按 `original, dark, bright` 排列。`暗光混合方块` 使用与旧 59 条完全相同的提示词。

## 4. 新数据 pilot 转换和验收

旧59条已经验收过；pilot 只转换第一条新数据（统一清单 source episode 59）：

```bash
uv run --with rosbags==0.11.0 examples/bpx/convert_bpx_combined_to_lerobot.py \
  --manifest-path "$MANIFEST" \
  --repo-id bpx/bpx_blocks_combined_pilot \
  --source-episodes 59 \
  --overwrite

uv run --with rosbags==0.11.0 examples/bpx/validate_bpx_combined_lerobot.py \
  --manifest-path "$MANIFEST" \
  --repo-id bpx/bpx_blocks_combined_pilot \
  --source-episodes 59
```

只有出现 `VALIDATION PASS` 才继续。

## 5. 全量转换和逐帧验收

```bash
uv run --with rosbags==0.11.0 examples/bpx/convert_bpx_combined_to_lerobot.py \
  --manifest-path "$MANIFEST" \
  --repo-id bpx/bpx_blocks_combined

uv run --with rosbags==0.11.0 examples/bpx/validate_bpx_combined_lerobot.py \
  --manifest-path "$MANIFEST" \
  --repo-id bpx/bpx_blocks_combined
```

目标目录是 `$HF_LEROBOT_HOME/bpx/bpx_blocks_combined`。验收会比较全部 175,041 帧的 18D state/action，并抽检每条轨迹首/中/末三相机 RGB。若目标目录已由失败的转换留下，先人工确认该目录确实是失败产物，再重新运行转换并显式加 `--overwrite`；不要对旧 `bpx/bpx_blocks` 使用该参数。

## 6. 重新计算并检查归一化统计

旧统计不能复用：

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_bpx_full_low_mem_combined \
  --num-workers 8

uv run examples/bpx/inspect_bpx_norm_stats.py \
  --norm-stats-path assets/pi05_bpx_full_combined/bpx/bpx_blocks_combined/norm_stats.json
```

`--num-workers 8` 只并行化数据读取，仍使用配置中的 `batch_size=2`、全部帧、相同变换和相同统计算法；不使用 `--max-frames`。确认 18 个 state 和 18 个 action 维度全部有限，且没有未经解释的 `LOW_STD` / `LOW_QRANGE`。

## 7. 全参数训练

下面仅把配置和实验名切换到新数据集；用户已经验证可训练的显存、batch、FSDP、保存参数完全不变：

```bash
export CUDA_VISIBLE_DEVICES=2,3

XLA_PYTHON_CLIENT_MEM_FRACTION=0.96 \
uv run scripts/train.py pi05_bpx_full_low_mem_combined \
  --exp-name=bpx_full_low_mem_combined_bs2 \
  --checkpoint-base-dir=/scratch/bpx_datasets/checkpoints \
  --batch-size=2 \
  --num-train-steps=100000 \
  --fsdp-devices=2 \
  --save-interval=5000 \
  --keep-period=25000 \
  --overwrite
```

中断后续训，所有参数和实验名必须一致，只把 `--overwrite` 换成 `--resume`：

```bash
export CUDA_VISIBLE_DEVICES=2,3

XLA_PYTHON_CLIENT_MEM_FRACTION=0.96 \
uv run scripts/train.py pi05_bpx_full_low_mem_combined \
  --exp-name=bpx_full_low_mem_combined_bs2 \
  --checkpoint-base-dir=/scratch/bpx_datasets/checkpoints \
  --batch-size=2 \
  --num-train-steps=100000 \
  --fsdp-devices=2 \
  --save-interval=5000 \
  --keep-period=25000 \
  --resume
```

`--num-train-steps` 是最终总步数，不是本次额外训练步数；需要延长时只提高该值，其他参数保持不变。
