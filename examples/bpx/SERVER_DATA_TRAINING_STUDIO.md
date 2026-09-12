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
└── dashboard_state/     # 任务状态与日志
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

uv sync --frozen
```

上面的 `uv sync` 是完整 OpenPI 环境的首次部署步骤。对于已经能够训练和推理的服务器，更新仪表盘版本后不需要再次运行它。BPX Studio 的小型纯 Python Web 运行时已随 Git 仓库放在 `third_party/bpx_studio_wheels/` 中，不依赖 PyPI 或 `uv` 缓存。

## 日常离线启动

```bash
cd ~/zzk_data/openpi-bpx-pi05-finetune
bash scripts/start_bpx_studio.sh
```

启动脚本直接使用项目自己的 `.venv/bin/python`，并从仓库内的 wheel 文件导入 FastAPI/Uvicorn；不会调用 `uv`、不会安装或解析依赖、也不会访问 PyPI。仓库文件缺失时脚本会明确退出，不会在后台下载。需要修改监听地址或端口时再传参数：

```bash
bash scripts/start_bpx_studio.sh --host 127.0.0.1 --port 8866
```

`/scratch/bpx_datasets` 已经是程序内置默认值，因此日常启动不需要重复 `export BPX_DATA_ROOT`，也不需要传 `--data-root`；只有迁移数据盘时才使用这两个覆盖入口。

默认监听 `0.0.0.0:8866`，无需 token。在局域网浏览器打开 `http://服务器IP:8866/`。旧 `.server-token` 文件不再读取，无需删除。该入口允许访问者启动和停止服务器任务，按局域网共享控制台使用。

更新后重启原启动命令，浏览器强制刷新。运行只读检查：

```bash
python scripts/verify_bpx_studio.py
```

检查 API、默认数据链、磁盘和 GPU 状态，不会启动训练。GPU 查询超时、驱动缺失或异常行现在会显示具体诊断，并自动重试。

## 推荐操作顺序

### 毛刺算法与复核

`isolated-excursion-v2` 使用逐维、原始单位的相邻帧跳变量，不按所有关节共用一个阈值。
自动修复要求：中心帧相对前后帧都超过 `jump_thresholds`，前后帧差不超过
`neighbor_thresholds`，并且中心值是局部峰或谷。修复值取两侧均值。
相邻候选相互依赖时不修改，避免把交替振荡翻转成新的异常。

持续阶跃、多个连续异常点、首尾跳变不会直接平滑。报告记录修复后仍超过阈值的
相邻帧跳变数量与位置；页面显示前 100 个位置，JSON 每文件每特征最多保留 200 个示例。
这些跳变可能是真实快速动作，也可能是异常，需人工查看原轨迹。
结构审计 PASS 只表示结构和数值有效，不代表轨迹已没有异常。
旧报告不含剩余跳变统计，更新算法后请重新扫描。

单帧真实快速往返动作也可能满足孤立峰值条件，算法无法只靠三个采样点判断动作意图。
因此关节弧度、夹爪开合量和其他维度必须分别设置阈值；不要直接照抄示例阈值。
工具拒绝跨 episode、缺帧或乱序帧插值，保留未修改数据的原列精度，且不改写源数据。

本地回归包含小型真实 MP4/Parquet 的审计 → 清洗 → 复审 → HDF5 导出，
以及归一化/训练/推理的命令与路径绑定。它不替代服务器真实数据训练和模型效果验证。

### 工作流程

约定的数据流如下；仪表盘会自动填入这些路径：

```text
采集仓库 recordings/lerobot/bpx_v2_humanoid_raw
  → /scratch/bpx_datasets/lerobot/bpx/bpx_v2_humanoid_raw
  → 数据审计与毛刺扫描
  → /scratch/bpx_datasets/lerobot/bpx/bpx_blocks_multitask_v2
  → 归一化
  → OpenPI 训练
  → /scratch/bpx_datasets/checkpoints/<config>/<experiment>/<step>
  → checkpoint 推理服务
```

1. 将工控机采集仓库 `recordings/lerobot/bpx_v2_humanoid_raw` 中冻结的数据集同步到服务器对应的 `_raw` 目录。
2. `headers` 审计会逐个核对 MP4 容器、尺寸和 FPS；正式冻结数据集前再执行一次 `full` 全帧解码。
3. 先执行“扫描毛刺”。扫描报告与清洗报告分别保存，不会相互覆盖；逐维阈值必须与当前机器人的 state/action 维度和单位匹配。
4. 点击“查看扫描报告”，在浏览器内按 feature/维度筛选命中点。每个命中点都能查看 episode、帧号、关节名、前后值、修复候选值、可调半径的前后帧曲线，以及命中帧全部数值维度。
5. 确认候选合理后再“生成清洗版本”。工具用硬链接快速复制，只有发生修复的 Parquet 会被原子替换；源数据集保持不变，并对输出重新审计。
6. 点击“查看清洗报告”复核清洗前后曲线。只有报告标记为已实际修复、清洗后数据审计通过，且操作者点击“确认清洗结果”后，页面才允许对该清洗数据集计算归一化；报告文件发生变化会自动要求重新确认。
7. 需要兼容其他工具时再从清洗后的 LeRobot 导出 HDF5。导出目录不可已存在，结果附带 SHA-256 manifest。
8. 归一化和训练始终共用同一个 Config。页面会读取该 Config 的 `repo_id`，明确展示实际 LeRobot 数据目录、归一化写入路径和训练加载路径，并检查它们是否与毛刺清洗输出贯通。
9. 默认使用 32 workers 计算全部归一化统计。检查 `norm_stats.json` 后启动训练。两张 48 GiB GPU 的已验证基线为 CUDA `2,3`、显存比例 `0.96`、global batch `2`、FSDP `2`。
10. Experiment name 会检查当前 config 的已有目录并从 `bpx_v2_humanoid_bs2_v001` 开始递增，防止覆盖已有实验；仍允许人工改名。
11. “模型推理”会扫描 `/scratch/bpx_datasets/checkpoints` 下所有 config 和 experiment 中同时具有 `params/` 和 `assets/` 的完整 checkpoint，忽略 `.orbax-checkpoint-tmp-*`。列表完整显示 `Config / Experiment / Step`，选择后自动匹配推理 Config，再启动仓库原生 `serve_policy.py`。停止按钮会终止该推理任务。

## Config 与数据源的严格关系

`scripts/compute_norm_stats.py` 不接受任意数据集目录。它只接受 `--config-name`，再从该 OpenPI Config 的 `data.repo_id` 解析数据源：

```text
HF_LEROBOT_HOME/<repo_id>
```

训练也通过同一个 Config 读取同一 `repo_id`，并加载该配置指定的 `norm_stats.json`。因此“清洗输出目录”和“Config 实际数据集”相同，才表示数据链真正贯通。控制台会显示绿色、黄色或红色状态，并在不一致时要求二次确认；归一化统计写入路径与训练加载路径不一致时则直接阻止操作。

当前默认 `pi05_bpx_full_low_mem_multitask_v2` 是**第一代机器人已有模型的兼容入口**，实际读取 `bpx/bpx_blocks_multitask_v2`，并不等于尚未定型的二代清洗目录 `bpx/bpx_v2_humanoid`。在二代机器人的关节顺序、动作维度、相机键和末端执行器契约确定前，不能安全地伪造一个二代 Config。契约完成后必须新增二代专用 Config，使其 `repo_id` 指向 `bpx/bpx_v2_humanoid`，届时页面会自动显示全链路贯通。

视频检查下拉框只改变 MP4 检查强度；无论选择哪一项，Parquet 中的 state/action 有限值与维度、episode/frame/task 索引以及时间戳都会完整检查：`none` 只核对 MP4 路径集合，`headers` 还逐个打开容器检查视频流、尺寸和 FPS，`full` 再逐帧解码并核对精确帧数。

毛刺修复只处理按阈值识别出的 `observation.state` 与 `action` 孤立单帧异常，以前后帧均值替换。它不会对整段轨迹做低通平滑，也不会修改连续异常、首尾帧、正常快速动作或视频。清洗输出使用硬链接节省空间：在同一文件系统上删除原始目录后，清洗目录中的链接及已重写 Parquet 仍然有效；但会失去重新调整阈值并复现清洗的原始基线，因此正式论文数据建议冷备原始集、毛刺报告和阈值文件。

`norm_stats.json` 仅保存各数值维度的均值、标准差和分位数等统计量，不复制视频或 episode，通常只有数 KB。放在仓库 `assets/<config>/<repo_id>/` 下是 OpenPI 的既定绑定方式；控制台会同时显示实际大小，并验证计算写入位置与训练加载位置完全相同。

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
