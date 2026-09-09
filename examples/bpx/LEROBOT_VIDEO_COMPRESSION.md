# BPX LeRobot v2.1 视频化压缩

## 结论

旧版 BPX 转换器把三路相机显式声明为 `dtype=image`。因此图像被逐帧嵌入
Parquet；`use_videos=True` 并不会覆盖显式 feature 类型。数据本身仍是合法的
LeRobot v2.1，但不是节省空间的 MP4-backed 布局。

`transcode_bpx_lerobot_images_to_video.py` 只读现有 LeRobot 数据集，另建一份
`dtype=video` 数据集。每个 episode、每个相机生成一个 MP4；state、action、
task、时间戳和索引会做一致性检查，并抽样检查视频解码后的 RGB 误差。

## 删除原始 bag 的影响

删除 `/scratch/bpx_datasets/new_datasets_cleaned` 不影响 checkpoint 推理。推理只需要：

- OpenPI 仓库代码和对应 config；
- checkpoint 的 `params/` 与 `assets/`；
- checkpoint 对应的 norm stats（checkpoint 的 `assets/` 中应已有副本）。

删除 bag 会失去从 ROS 原始消息重新审计、重新清洗和重新转换的能力，也不能再运行
基于这些 bag 的离线评测。由于 bag 已删除，在视频化数据完整验收以前，不要删除当前
LeRobot 数据集。

## 1. 查看实际占用和存储类型

```bash
du -h --max-depth=2 /scratch/bpx_datasets/lerobot | sort -h

find /scratch/bpx_datasets/lerobot -type f -name '*.parquet' -printf '%s\n' \
  | awk '{s+=$1} END {printf "parquet: %.2f GiB\n", s/1024/1024/1024}'

find /scratch/bpx_datasets/lerobot -type f -name '*.mp4' -printf '%s\n' \
  | awk '{s+=$1} END {printf "mp4: %.2f GiB\n", s/1024/1024/1024}'
```

查看某个数据集的相机 feature：

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("/scratch/bpx_datasets/lerobot/bpx/bpx_blocks_multitask_v2/meta/info.json")
info = json.loads(path.read_text())
print({key: value["dtype"] for key, value in info["features"].items() if key.startswith("observation.images.")})
PY
```

旧数据会显示三个 `image`；正确的视频数据应显示三个 `video`。

## 2. 先做单条 pilot，不覆盖源数据

```bash
cd ~/zzk_data/openpi-bpx-pi05-finetune

export HF_LEROBOT_HOME=/scratch/bpx_datasets/lerobot
export HF_HOME=/scratch/bpx_datasets/cache/huggingface
export HF_DATASETS_CACHE=/scratch/bpx_datasets/cache/huggingface/datasets
export TMPDIR=/scratch/bpx_datasets/tmp

mkdir -p "$HF_DATASETS_CACHE" "$TMPDIR" logs

uv run examples/bpx/transcode_bpx_lerobot_images_to_video.py \
  --source-repo-id bpx/bpx_blocks_multitask_v2 \
  --output-repo-id bpx/bpx_blocks_multitask_v2_video_pilot \
  --episode-limit 1
```

结尾出现 `TRANSCODE PASS` 后，再转换完整 multitask v2：

```bash
uv run examples/bpx/transcode_bpx_lerobot_images_to_video.py \
  --source-repo-id bpx/bpx_blocks_multitask_v2 \
  --output-repo-id bpx/bpx_blocks_multitask_v2_video \
  2>&1 | tee logs/transcode_bpx_blocks_multitask_v2_video.log
```

编码使用仓库锁定的 LeRobot v2.1 默认参数：AV1、YUV420p、CRF 30、30 Hz。
这是有损视频压缩，但脚本会逐帧保持数值模态完全一致，并对每条 episode 的首、中、
末 RGB 帧执行 MAE 门禁。转换期间需要源和目标同时存在。

成功结尾必须包含：

```text
TRANSCODE PASS: episodes=332, frames=119057, videos=996, ...
```

## 3. 验收目标数据

```bash
du -sh \
  /scratch/bpx_datasets/lerobot/bpx/bpx_blocks_multitask_v2 \
  /scratch/bpx_datasets/lerobot/bpx/bpx_blocks_multitask_v2_video

find /scratch/bpx_datasets/lerobot/bpx/bpx_blocks_multitask_v2_video \
  -type f -name '*.mp4' | wc -l

uv run python - <<'PY'
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

root = Path("/scratch/bpx_datasets/lerobot/bpx/bpx_blocks_multitask_v2_video")
dataset = LeRobotDataset("bpx/bpx_blocks_multitask_v2_video", root=root)
assert dataset.num_episodes == 332
assert dataset.num_frames == 119057
assert len(dataset.meta.video_keys) == 3
assert not dataset.meta.image_keys
for index in (0, len(dataset) // 2, len(dataset) - 1):
    item = dataset[index]
    assert item["observation.images.left_eye"].shape == (3, 360, 640)
print("VIDEO DATASET LOAD PASS")
PY
```

只有 `TRANSCODE PASS`、MP4 数量、随机读取和体积检查全部通过后，才考虑移除旧副本。
如果还要训练且不想修改 config，可在停训后把两个目录分别重命名，让视频版占用原来的
`bpx_blocks_multitask_v2` 目录名；重命名后再次执行上面的加载测试。

## 4. 其他数据集

同一脚本可转换旧的 combined 数据集：

```bash
uv run examples/bpx/transcode_bpx_lerobot_images_to_video.py \
  --source-repo-id bpx/bpx_blocks_combined \
  --output-repo-id bpx/bpx_blocks_combined_video \
  2>&1 | tee logs/transcode_bpx_blocks_combined_video.log
```

不要盲目转换 pilot、中间产物和已经不会再训练的数据集。纯 checkpoint 推理完全不读取
LeRobot 数据；确认 checkpoint 与 norm stats 已备份后，删除不再需要的数据集比重新编码
它们更节省空间。
