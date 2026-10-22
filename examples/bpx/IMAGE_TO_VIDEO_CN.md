# 一代 BPX：Parquet 图像转独立 MP4 的 LeRobot v2.1

新版启动器改用 `openpi.bpx_studio.video_transcoder`：直接逐批读取 Parquet，多个 episode 并行，不导入 LeRobot、Torch、JAX 或 Hugging Face datasets，不联网、不加载整套数据集、不写 PNG 或 Arrow 数据缓存。旧 `examples/bpx/transcode_bpx_lerobot_images_to_video.py` 仍保留供兼容，但不再由启动器调用。

原先 `dtype=image` 本身是合法 LeRobot 存储形式，并非损坏数据。视频版使用相同的 LeRobot v2.1 数值与任务契约，独立 MP4 更方便播放、共享和检查，也通常节省空间；但不保证对所有解码器和随机读训练场景更快。不要为转换而升级服务器当前锁定的 LeRobot 版本。

## 服务器执行

在原项目中拉取更新后，使用已安装的项目 `.venv` 离线执行，不运行 `uv sync`，不改全局环境：

```bash
cd ~/zzk_data/openpi-bpx-pi05-finetune
bash scripts/transcode_bpx_video.sh --workers 64 --encoder-threads 2
```

默认读取 `/scratch/bpx_datasets/lerobot/bpx/bpx_blocks_combined`，输出到 `/scratch/bpx_datasets/lerobot/bpx/bpx_blocks_combined_video`。如果前者已替换成清洗结果，此命令转换的就是该路径下当前的数据，不会重新读取旧 bag。源保持只读，输出已存在则拒绝启动；没有覆盖选项，也不要把输入输出设成相同目录或互相嵌套的目录。

上面的高并行参数面向 112 核 / 224 线程服务器，最多同时转换 64 个 episode，每个工作的编码器设置 2 个线程，另有图像解码与 I/O 开销。它不是 64×3 路编码器同时无限开线程。不传 workers 时默认 32；SATA/共享存储可能先达到 I/O 瓶颈，64 不保证比 32 更快，可调整为 32 或 48。

建议先在服务器试一条，使用单独的输出目录，确认当前服务器环境可运行后再执行上面的全量命令：

```bash
bash scripts/transcode_bpx_video.sh \
  --output-root /scratch/bpx_datasets/lerobot/bpx/bpx_blocks_combined_video_pilot \
  --episode-limit 1 --workers 1 --encoder-threads 2
```

自定义源或输出：

```bash
bash scripts/transcode_bpx_video.sh \
  --source-root /scratch/bpx_datasets/lerobot/bpx/bpx_blocks_combined \
  --output-root /scratch/bpx_datasets/lerobot/bpx/bpx_blocks_combined_video \
  --workers 64 --encoder-threads 2
```

新版按 LeRobot v2.1 的本地 features 解析全部 image 相机、数值维度及 FPS，不猜测关节单位，也不扩展或重排关节。源必须是一条 episode 对应一个 Parquet、episode 编号从 0 连续的 v2.1 数据，且无已经压缩过的 video 特征。图像宽高需为偶数；不满足时直接报错而不自动缩放。

## 压缩与校验

- 多个 episode 进程并行，每个进程顺序处理各相机；限制内部 Arrow/BLAS/视频解码线程，编码线程由 encoder-threads 指定。CPU 编码，不占用训练 GPU。
- 显式 H.264、CRF 18、yuv420p、GOP 2；不裁剪、不缩放、不增广，不做关节清洗，也不改夹爪。
- H.264/yuv420p 是有损编码，即使 CRF=0 也不能据此保证 RGB 像素无损。这里对每条轨迹抽样校验 RGB MAE，默认每条每相机抽 3 帧，不是全部像素等价证明。
- 每条轨迹全量比较所有非图像列的数值和 Arrow 类型，包括未知附加字段、夹爪、语言编号和时间戳；不把 state/action 转成训练增量或归一化值。要求源时间戳满足固定 FPS，拒绝悄悄重采样。
- 所有 MP4 全帧解码，核对每帧尺寸、FPS、PTS 时间和帧数。派生的 LeRobot episode 统计重新计算，图像统计来自实际解码视频；不覆盖 OpenPI 的 norm_stats.json。
- 正常结束显示 `TRANSCODE PASS`，并写入输出目录 `transcode_validation.json`。中断或失败保留明确命名的暂存目录，不要拿半成品训练或删除源数据。
- 默认每个相机 8 帧一批读取，可用 `--batch-frames` 调整；单 worker 内存还受原始 Parquet row group/page 大小影响，不是严格只占 8 帧。转换期间原数据和新数据同时存在，检查磁盘空间；不再临时写 PNG。输出体积以实际结果为准。

## 日志、暂存与断点续转

启动后立即显示 BOOT、SOURCE、OUTPUT、PREFLIGHT 等阶段。工作进程显示 episode、相机、编码或校验进度，父进程在等待任务时每 15 秒输出状态。

**正式输出目录只在全部完成后出现。** 运行中实际写入：

```text
/scratch/bpx_datasets/lerobot/bpx/.bpx_blocks_combined_video.transcoding
```

中断后原参数不变，加 `--resume`：

```bash
bash scripts/transcode_bpx_video.sh --workers 64 --encoder-threads 2 --resume
```

已完成轨迹的 Parquet/视频会核验 SHA-256，通过后跳过；未完成或校验失败的轨迹重新生成。源文件身份检查采用大小与 mtime，不是对所有原图内容做密码学哈希。可调整 workers，但编码参数、batch-frames 或源元数据改变时不能复用原暂存身份。同一输出有活动任务时拒绝第二份转换；不要在运行中改动源数据。

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

本次流式并行新版遵照用户要求，未在本机安装依赖、执行转换或运行测试；实际运行验证由服务器进行。
