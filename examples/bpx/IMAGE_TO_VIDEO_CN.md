# 一代 BPX：Parquet 图像转独立 MP4 的 LeRobot v2.1

原先 `dtype=image` 本身是合法 LeRobot 存储形式，并非损坏数据。视频版使用相同的 LeRobot v2.1 数值与任务契约，独立 MP4 更方便播放、共享和检查，也通常节省空间；但不保证对所有解码器和随机读训练场景更快。不要为转换而升级服务器当前锁定的 LeRobot 版本。

## 服务器执行

在原项目中拉取更新后，使用已安装的项目 `.venv` 离线执行，不运行 `uv sync`，不改全局环境：

```bash
cd ~/zzk_data/openpi-bpx-pi05-finetune
bash scripts/transcode_bpx_video.sh
```

默认读取 `/scratch/bpx_datasets/lerobot/bpx/bpx_blocks_combined`，输出到 `/scratch/bpx_datasets/lerobot/bpx/bpx_blocks_combined_video`。如果前者已替换成清洗结果，此命令转换的就是该路径下当前的数据，不会重新读取旧 bag。源保持只读，输出已存在则拒绝启动；不要加 `--overwrite`，也不要把输入输出设成相同 ID。

若先试一条，使用单独的测试 ID，测试通过后再执行上面的全量命令：

```bash
bash scripts/transcode_bpx_video.sh \
  --output-repo-id bpx/bpx_blocks_combined_video_pilot \
  --episode-limit 1
```

自定义源或输出：

```bash
bash scripts/transcode_bpx_video.sh \
  --source-repo-id bpx/bpx_blocks_combined \
  --output-repo-id bpx/bpx_blocks_combined_video
```

脚本参数只适用于一代 30 FPS、18 维、left_eye/left_wrist/right_wrist 三路相机的数据，不适用于二代四相机数据。

## 压缩与校验

- 32 个图片写入线程，三路相机最多同时编码；episode 依次处理，不是 32 个 episode 同时编码。CPU 编码，不占用训练 GPU。
- 显式 H.264、CRF 18、yuv420p、GOP 2；不裁剪、不缩放、不增广，不做关节清洗，也不改夹爪。
- H.264/yuv420p 是有损编码，即使 CRF=0 也不能据此保证 RGB 像素无损。这里对每条轨迹抽样校验 RGB MAE，默认每条每相机抽 3 帧，不是全部像素等价证明。
- 每条轨迹全量检查 state/action、索引和任务编号不变，时间戳在原脚本规定容差内保持一致；最后所有 MP4 全帧解码，核对形状、FPS 与轨迹帧数。
- 正常结束显示 `TRANSCODE PASS`，并写入输出目录 `transcode_validation.json`。中断或失败的目录不是合格的最终结果，不要拿它训练，也不要删除源数据；保留报错排查。暂未实现断点续转。
- 检查磁盘空间；转换期间原数据和新数据同时存在，单条编码还会临时写 PNG。输出体积必须以实际结果为准，不能预先保证压缩比例。

对于原合并集 383 个 episode，预期 1149 个独立 MP4，目录为：

```text
bpx_blocks_combined_video/
├── data/chunk-000/episode_000000.parquet
├── meta/info.json                 # 相机 dtype=video，含视频路径模板
├── meta/episodes.jsonl
├── meta/episodes_stats.jsonl
├── meta/tasks.jsonl
├── videos/chunk-000/
│   ├── observation.images.left_eye/episode_000000.mp4
│   ├── observation.images.left_wrist/episode_000000.mp4
│   └── observation.images.right_wrist/episode_000000.mp4
└── transcode_validation.json
```

转换不会改旧模型、推理服务或 OpenPI Config。新目录使用新 repo_id，旧训练 Config 不会自动指向它。后续训练需显式确认 Config 的 repo_id 与数据路径匹配。只做图像存储转换、数值完全不变，不因这个转换本身要求重算 state/action 归一化，但若 repo_id 改变仍需正确绑定统计资产；已有推理不受影响。确认转换通过且能够被服务器原 LeRobot 环境加载之前，不要删除或覆盖源数据。
