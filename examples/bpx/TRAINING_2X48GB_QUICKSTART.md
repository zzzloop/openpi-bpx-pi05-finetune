# BPX π0.5 双 48 GiB GPU 训练速查

适用环境：物理 GPU `2,3`，Ubuntu 22.04，仓库根目录执行。全参数训练使用已通过 300-step pilot 的
`pi05_bpx_full_low_mem`（全参数 AdamW、关闭 EMA、global batch 2）；LoRA 使用 `pi05_bpx_lora`。

## 1. 公共环境

```bash
cd ~/zzk_data/openpi-bpx-pi05-finetune
deactivate 2>/dev/null || true

git pull --ff-only origin main
uv sync

export CUDA_VISIBLE_DEVICES=2,3
export HF_LEROBOT_HOME=./datasets/lerobot
export WANDB_MODE=offline

uv run python - <<'PY'
import jax

print("backend:", jax.default_backend())
print("devices:", jax.devices())
assert jax.default_backend() == "gpu"
assert jax.device_count() == 2
PY
```

训练时另开终端监控物理 GPU：

```bash
watch -n 1 nvidia-smi -i 2,3
```

## 2. 全参数训练（关闭 EMA）

正式训练约采样 `100000 × 2 = 200000` 个训练帧，约为当前 67,455 帧数据集的 3 次采样覆盖。首次启动：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.96 \
uv run scripts/train.py pi05_bpx_full_low_mem \
  --exp-name=bpx_full_low_mem_bs2 \
  --batch-size=2 \
  --num-train-steps=100000 \
  --fsdp-devices=2 \
  --save-interval=5000 \
  --keep-period=25000 \
  --overwrite
```

中断后续训，参数必须保持一致，并用 `--resume` 替换 `--overwrite`：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.96 \
uv run scripts/train.py pi05_bpx_full_low_mem \
  --exp-name=bpx_full_low_mem_bs2 \
  --batch-size=2 \
  --num-train-steps=100000 \
  --fsdp-devices=2 \
  --save-interval=5000 \
  --keep-period=25000 \
  --resume
```

## 3. LoRA 训练

先验证 global batch 8 能通过一次完整更新：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.96 \
uv run scripts/train.py pi05_bpx_lora \
  --exp-name=bpx_lora_bs8_gate \
  --batch-size=8 \
  --num-train-steps=1 \
  --fsdp-devices=2 \
  --overwrite
```

门禁通过后正式训练，约采样 `30000 × 8 = 240000` 个训练帧：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.96 \
uv run scripts/train.py pi05_bpx_lora \
  --exp-name=bpx_lora_bs8 \
  --batch-size=8 \
  --num-train-steps=30000 \
  --fsdp-devices=2 \
  --save-interval=5000 \
  --keep-period=10000 \
  --overwrite
```

LoRA 续训同样保持参数不变，并用 `--resume` 替换 `--overwrite`。若 batch 8 的 1-step 门禁 OOM，不运行正式命令；
改用 batch 2，并将正式训练步数改为 120,000，以保持约 240,000 个训练样本。

## 4. 监控与 checkpoint 验收

训练曲线、三相机图像和动作标签自动保存在：

```text
checkpoints/<config>/<exp-name>/monitoring/
```

必须确认 loss 总体下降，且无 `RESOURCE_EXHAUSTED`、NaN、Inf、相机黑屏或动作标签异常。

查找最新全参数 checkpoint 并使用真实 HDF5 验证：

```bash
FULL_STEP=$(find checkpoints/pi05_bpx_full_low_mem/bpx_full_low_mem_bs2 \
  -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | grep -E '^[0-9]+$' | sort -n | tail -1)

uv run examples/bpx/smoke_test_bpx_checkpoint.py \
  --config-name pi05_bpx_full_low_mem \
  --checkpoint-dir "checkpoints/pi05_bpx_full_low_mem/bpx_full_low_mem_bs2/$FULL_STEP" \
  --episode-path datasets/cleaned_bpx/episode_0.hdf5 \
  --frame-index 0
```

LoRA 验证：

```bash
LORA_STEP=$(find checkpoints/pi05_bpx_lora/bpx_lora_bs8 \
  -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | grep -E '^[0-9]+$' | sort -n | tail -1)

uv run examples/bpx/smoke_test_bpx_checkpoint.py \
  --config-name pi05_bpx_lora \
  --checkpoint-dir "checkpoints/pi05_bpx_lora/bpx_lora_bs8/$LORA_STEP" \
  --episode-path datasets/cleaned_bpx/episode_0.hdf5 \
  --frame-index 0
```

smoke 必须输出 `PASS`、`output_shape=(30, 23)`、`effective_shape=(30, 18)` 和 `finite=True`。smoke 只验证推理与
23D 输出契约；最终 checkpoint 必须按独立真机场景成功率选择，不能只选训练 loss 最低者。
