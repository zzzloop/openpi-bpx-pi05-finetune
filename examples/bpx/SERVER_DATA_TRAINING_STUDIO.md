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

mkdir -p \
  /scratch/bpx_datasets/lerobot \
  /scratch/bpx_datasets/cleaned \
  /scratch/bpx_datasets/exports \
  /scratch/bpx_datasets/checkpoints \
  /scratch/bpx_datasets/reports \
  /scratch/bpx_datasets/cache/huggingface/datasets \
  /scratch/bpx_datasets/cache/uv \
  /scratch/bpx_datasets/tmp \
  /scratch/bpx_datasets/dashboard_state

uv sync
uv run --with 'fastapi>=0.115' --with 'uvicorn>=0.34' \
  python -m openpi.bpx_studio.api \
  --host 127.0.0.1 \
  --port 8866
```

`/scratch/bpx_datasets` 已经是程序内置默认值，因此日常启动不需要重复 `export BPX_DATA_ROOT`，也不需要传 `--data-root`；只有迁移数据盘时才使用这两个覆盖入口。

终端会打印带随机控制令牌的 URL。控制台仅监听服务器本机；从工作电脑访问时使用 SSH 隧道：

```bash
ssh -L 8866:127.0.0.1:8866 kemove@训练服务器IP
```

然后在本机浏览器打开终端打印的 `http://127.0.0.1:8866/?token=...`。不要把令牌提交到 Git。

## 推荐操作顺序

约定的数据流如下；仪表盘会自动填入这些路径：

```text
采集仓库 recordings/lerobot/bpx_v2_humanoid_raw
  → /scratch/bpx_datasets/lerobot/bpx/bpx_v2_humanoid_raw
  → 数据审计与毛刺扫描
  → /scratch/bpx_datasets/lerobot/bpx/bpx_v2_humanoid
  → 归一化
  → OpenPI 训练
  → /scratch/bpx_datasets/checkpoints/<config>/<experiment>/<step>
  → checkpoint 推理服务
```

1. 将工控机采集仓库 `recordings/lerobot/bpx_v2_humanoid_raw` 中冻结的数据集同步到服务器对应的 `_raw` 目录。
2. `headers` 审计会逐个核对 MP4 容器、尺寸和 FPS；正式冻结数据集前再执行一次 `full` 全帧解码。
3. 先执行“仅扫描毛刺”，人工审核 JSON 报告。逐维阈值必须与当前机器人的 state/action 维度和单位匹配。
4. 确认后再“生成清洗版本”。工具用硬链接快速复制，只有发生修复的 Parquet 会被原子替换；源数据集保持不变，并对输出重新审计。
5. 需要兼容其他工具时再从清洗后的 LeRobot 导出 HDF5。导出目录不可已存在，结果附带 SHA-256 manifest。
6. 归一化和训练默认共用 `pi05_bpx_full_low_mem_multitask_v2`。这是当前已有模型的可运行默认值；二代机器人 OpenPI config 完成后，应统一换成二代 config 名，不能混用旧机器人的特征契约和归一化统计。
7. 默认使用 32 workers 计算全部归一化统计。检查 `norm_stats.json` 后启动训练。两张 48 GiB GPU 的已验证基线为 CUDA `2,3`、显存比例 `0.96`、global batch `2`、FSDP `2`。
8. Experiment name 会检查当前 config 的已有目录并从 `bpx_v2_humanoid_bs2_v001` 开始递增，防止覆盖已有实验；仍允许人工改名。
9. “模型推理”会扫描 `/scratch/bpx_datasets/checkpoints` 下所有 config 和 experiment 中同时具有 `params/` 和 `assets/` 的完整 checkpoint，忽略 `.orbax-checkpoint-tmp-*`。列表完整显示 `Config / Experiment / Step`，选择后自动匹配推理 Config，再启动仓库原生 `serve_policy.py`。停止按钮会终止该推理任务。

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
