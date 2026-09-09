# OpenPI BPX 服务器数据与训练控制台

本控制台属于 **OpenPI 服务器仓库**，不是 ROS2 工控机采集仓库。它在训练服务器上编排 LeRobot v2.1 数据审计、毛刺扫描/清洗、HDF5 派生导出、归一化统计和 OpenPI 训练。

## 路径边界

- 代码、虚拟环境和小型归一化资产：`~/zzk_data/openpi-bpx-pi05-finetune`
- 大型数据、缓存、报告和 checkpoint：`/scratch/bpx_datasets`
- 默认不会把数据集、checkpoint 或缓存写入 `/home`。

推荐目录：

```text
/scratch/bpx_datasets/
├── lerobot/             # LeRobot 权威数据集
├── cleaned/             # 独立清洗版本
├── exports/             # HDF5 等派生格式
├── checkpoints/         # OpenPI checkpoint
├── reports/             # 审计与毛刺报告
├── cache/
│   ├── huggingface/
│   └── uv/
├── tmp/
└── dashboard_state/     # 任务状态、日志和访问令牌
```

归一化统计仍由 OpenPI 按配置写入仓库的 `assets/<config>/.../norm_stats.json`。它很小，而且必须和训练 config 对应，不需要迁到 `/scratch`。

## 首次启动

```bash
cd ~/zzk_data/openpi-bpx-pi05-finetune

export BPX_DATA_ROOT=/scratch/bpx_datasets
mkdir -p \
  "$BPX_DATA_ROOT"/lerobot \
  "$BPX_DATA_ROOT"/cleaned \
  "$BPX_DATA_ROOT"/exports \
  "$BPX_DATA_ROOT"/checkpoints \
  "$BPX_DATA_ROOT"/reports \
  "$BPX_DATA_ROOT"/cache/huggingface/datasets \
  "$BPX_DATA_ROOT"/cache/uv \
  "$BPX_DATA_ROOT"/tmp \
  "$BPX_DATA_ROOT"/dashboard_state

uv sync
uv run --with 'fastapi>=0.115' --with 'uvicorn>=0.34' \
  python -m openpi.bpx_studio.api \
  --host 127.0.0.1 \
  --port 8866 \
  --data-root "$BPX_DATA_ROOT"
```

终端会打印带随机控制令牌的 URL。控制台仅监听服务器本机；从工作电脑访问时使用 SSH 隧道：

```bash
ssh -L 8866:127.0.0.1:8866 kemove@训练服务器IP
```

然后在本机浏览器打开终端打印的 `http://127.0.0.1:8866/?token=...`。不要把令牌提交到 Git。

## 推荐操作顺序

1. 在“数据审计”中选择 `/scratch/bpx_datasets/lerobot/<组织>/<数据集>`。`headers` 会逐个核对 MP4 容器、尺寸和 FPS；正式冻结数据集前再执行一次 `full` 全帧解码。
2. 先执行“仅扫描毛刺”，人工审核 JSON 报告。逐维阈值必须与当前机器人的 state/action 维度和单位匹配。
3. 确认后再“生成清洗版本”。工具用硬链接快速复制，只有发生修复的 Parquet 会被原子替换；源数据集保持不变，并对输出重新审计。
4. 需要兼容其他工具时再从 LeRobot 导出 HDF5。导出目录不可已存在，结果附带 SHA-256 manifest。
5. 选择与数据集对应的 OpenPI config，使用 32 workers 计算全部归一化统计。
6. 检查 `norm_stats.json` 后启动训练。两张 48 GiB GPU 的已验证基线为 CUDA `2,3`、显存比例 `0.96`、global batch `2`、FSDP `2`。

## 毛刺阈值文件

阈值文件必须为每个 state/action 维度分别提供跳变阈值和邻点一致阈值：

```json
{
  "observation.state": {
    "jump_thresholds": [0.25, 0.25],
    "neighbor_thresholds": [0.25, 0.25]
  },
  "action": {
    "jump_thresholds": [0.25, 0.25],
    "neighbor_thresholds": [0.25, 0.25]
  }
}
```

上例只有两维，只说明格式，不能直接用于真实机器人。实际数组长度必须严格等于对应特征维度。算法仅在中心点同时远离前后两点、且前后两点彼此接近时，用前后均值替换中心点；不会平滑整段轨迹。

## 训练与续训约束

控制台最终调用仓库原生 `scripts/train.py`，不会另写训练实现。新实验使用 `overwrite`；中断续训使用 `resume`。续训时 config、实验名、checkpoint 根目录、batch、FSDP 设备数、保存间隔和长期保留间隔必须与原实验一致，训练总 steps 可以增加。

任务在独立进程中运行，关闭网页不会停止任务；任务命令、环境变量、日志和训练指标保存在 `/scratch/bpx_datasets/dashboard_state/server_jobs`。同一时间只允许一个重任务，避免两次训练或数据写入互相争抢资源。
