# BPX 多任务 V2 数据重建与训练

目标：保留现有 V1 80k 模型随时可推理；从 59 条固定布局旧轨迹中客观选出 8 条代表轨迹，保留全部 324 条新多任务轨迹，并用任务等权采样重新训练。V2 从 `pi05_base` 开始，禁止从 V1 80k `--resume`。

## 1. 保护 V1 并释放可再生空间

先确认训练、转换和归一化进程均已停止；不要在仍有训练进程读取数据时删除缓存或 checkpoint。

```bash
cd ~/zzk_data/openpi-bpx-pi05-finetune

export V1_CKPT=/scratch/bpx_datasets/checkpoints/pi05_bpx_full_low_mem_combined/bpx_full_low_mem_combined_bs2/80000
test -d "$V1_CKPT/params" \
  && test -d "$V1_CKPT/assets" \
  && test -d "$V1_CKPT/train_state" \
  && find "$V1_CKPT/assets" -name norm_stats.json -print \
  && echo "V1 80k checkpoint PASS"
```

只有上一条输出 `PASS` 后，才删除两个旧 checkpoint 和可再生 HF 数据缓存。它们预计释放约 258 GiB；不会删除 80k checkpoint、原始 BAG、V1 LeRobot 数据或归一化资产。

```bash
rm -rf -- \
  /scratch/bpx_datasets/checkpoints/pi05_bpx_full_low_mem_combined/bpx_full_low_mem_combined_bs2/50000 \
  /scratch/bpx_datasets/checkpoints/pi05_bpx_full_low_mem_combined/bpx_full_low_mem_combined_bs2/75000 \
  /scratch/bpx_datasets/cache/huggingface/datasets

df -h /scratch "$HOME"
```

## 2. 构建 V2 manifest

该步骤只读取旧 manifest 和旧轨迹，不复制图像。筛选使用 32 个归一化时间点的 `state + action + action-state` 轨迹签名，并进行确定性的 k-medoids 代表性选择；输出会记录旧 manifest SHA256 和入选编号。

```bash
cd ~/zzk_data/openpi-bpx-pi05-finetune

export V1_MANIFEST=/scratch/bpx_datasets/bpx_pi05_combined_source/manifests/combined_episodes.json
export V2_MANIFEST=/scratch/bpx_datasets/bpx_pi05_multitask_v2/manifests/multitask_v2.json

uv run examples/bpx/build_bpx_multitask_v2_manifest.py \
  --source-manifest-path "$V1_MANIFEST" \
  --output-manifest-path "$V2_MANIFEST" \
  --legacy-keep 8 \
  --signature-points 32
```

必须看到 `V2 MANIFEST PASS`。V2 应为 332 条 episode（8 条旧轨迹 + 324 条新轨迹）。

## 3. 转换并严格验收 V2 LeRobot

```bash
export HF_LEROBOT_HOME=/scratch/bpx_datasets/lerobot
export HF_HOME=/scratch/bpx_datasets/cache/huggingface
export HF_DATASETS_CACHE=/scratch/bpx_datasets/cache/huggingface/datasets
export UV_CACHE_DIR=/scratch/bpx_datasets/cache/uv
export TMPDIR=/scratch/bpx_datasets/tmp

mkdir -p "$HF_DATASETS_CACHE" "$UV_CACHE_DIR" "$TMPDIR" logs

uv run --with rosbags==0.11.0 \
  examples/bpx/convert_bpx_combined_to_lerobot.py \
  --manifest-path "$V2_MANIFEST" \
  --repo-id bpx/bpx_blocks_multitask_v2 \
  2>&1 | tee logs/convert_bpx_multitask_v2.log

uv run --with rosbags==0.11.0 \
  examples/bpx/validate_bpx_combined_lerobot.py \
  --manifest-path "$V2_MANIFEST" \
  --repo-id bpx/bpx_blocks_multitask_v2 \
  2>&1 | tee logs/validate_bpx_multitask_v2.log
```

转换中断且确认目标只是未完成的 V2 时，才在转换命令末尾添加 `--overwrite` 重建。验收必须输出 `VALIDATION PASS`。

## 4. 重新计算并验收 V2 归一化

V2 配置会在统计和训练中使用相同的任务等权采样。每个语言任务获得相同总采样概率，任务内部按帧均匀采样，从而防止旧固定轨迹或蓝色任务因帧数较多而主导优化。

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_bpx_full_low_mem_multitask_v2 \
  --num-workers 32

uv run examples/bpx/inspect_bpx_norm_stats.py \
  --norm-stats-path \
  assets/pi05_bpx_full_low_mem_multitask_v2/bpx/bpx_blocks_multitask_v2/norm_stats.json

uv run python -c 'from openpi.training import config; c=config.get_config("pi05_bpx_full_low_mem_multitask_v2"); d=c.data.create(c.assets_dirs,c.model); assert d.norm_stats is not None; assert d.balance_by_task; print("V2 NORM STATS LOAD PASS")'
```

必须同时看到 18 维检查 `PASS` 和 `V2 NORM STATS LOAD PASS`。

## 5. 从 base 启动独立 V2 训练

```bash
mkdir -p /scratch/bpx_datasets/checkpoints

export CUDA_VISIBLE_DEVICES=2,3
export HF_LEROBOT_HOME=/scratch/bpx_datasets/lerobot
export HF_HOME=/scratch/bpx_datasets/cache/huggingface
export HF_DATASETS_CACHE=/scratch/bpx_datasets/cache/huggingface/datasets
export TMPDIR=/scratch/bpx_datasets/tmp

XLA_PYTHON_CLIENT_MEM_FRACTION=0.96 \
uv run scripts/train.py pi05_bpx_full_low_mem_multitask_v2 \
  --exp-name=bpx_multitask_v2_bs2 \
  --checkpoint-base-dir=/scratch/bpx_datasets/checkpoints \
  --batch-size=2 \
  --num-train-steps=100000 \
  --fsdp-devices=2 \
  --save-interval=5000 \
  --keep-period=25000 \
  --overwrite
```

先在 100k 做离线 BAG 与真机分任务评测。若仍持续改善，再保持所有参数不变，把目标步数改为 150k 或 200k，并用 `--resume` 替换 `--overwrite`：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.96 \
uv run scripts/train.py pi05_bpx_full_low_mem_multitask_v2 \
  --exp-name=bpx_multitask_v2_bs2 \
  --checkpoint-base-dir=/scratch/bpx_datasets/checkpoints \
  --batch-size=2 \
  --num-train-steps=150000 \
  --fsdp-devices=2 \
  --save-interval=5000 \
  --keep-period=25000 \
  --resume
```

## 6. V2 推理与回退 V1

V2：

```bash
export CUDA_VISIBLE_DEVICES=2,3
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_bpx_full_low_mem_multitask_v2 \
  --policy.dir=/scratch/bpx_datasets/checkpoints/pi05_bpx_full_low_mem_multitask_v2/bpx_multitask_v2_bs2/100000
```

随时回退到保留的 V1 80k：

```bash
export CUDA_VISIBLE_DEVICES=2,3
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_bpx_full_low_mem_combined \
  --policy.dir=/scratch/bpx_datasets/checkpoints/pi05_bpx_full_low_mem_combined/bpx_full_low_mem_combined_bs2/80000
```

## 7. V2 验收后的第二阶段清理

只有当 V2 LeRobot 已 `VALIDATION PASS`、归一化已通过且 V1 80k checkpoint 完整时，才可删除约 105 GiB 的 V1 合并 LeRobot 数据和约 764 MiB 的 pilot。删除后 V1 80k 仍可推理，但不能再依赖该数据继续训练或做数据集离线评测。

```bash
rm -rf -- \
  /scratch/bpx_datasets/lerobot/bpx/bpx_blocks_combined \
  /scratch/bpx_datasets/lerobot/bpx/bpx_blocks_combined_pilot
```

禁止删除：

- `/scratch/bpx_datasets/new_datasets_cleaned`：不可重新采集的原始 BAG；
- V1 的 `80000` checkpoint；
- V2 LeRobot、V2 norm stats 和 V2 checkpoint；
- V2 验收前所依赖的旧 59 条 LeRobot 源数据。

训练完成后，HF datasets cache 仍可再次删除；首次重新加载数据时会自动重建。
