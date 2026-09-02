# BPX ROS bag checkpoint离线评估

该流程把一条同步ROS1 bag的实测反馈、三路RGB和任务语言逐帧发给已经启动的OpenPI策略服务，并将模型30步预测与示教action进行开环对比。

这能验证checkpoint加载、语言、图像、关节顺序、输出维度和训练拟合情况，但不是闭环真机成功率。因为每次推理看到的是专家示教轨迹中的真实观测，而不是模型动作执行后产生的新观测。

## 1. 检查checkpoint

```bash
export CKPT=/scratch/bpx_datasets/checkpoints/pi05_bpx_full_low_mem_combined/EXPERIMENT/STEP

test -d "$CKPT/params"
test -d "$CKPT/assets"
test -d "$CKPT/train_state"
```

## 2. 启动combined策略服务

训练进程和服务进程不要同时占用同一组GPU。停止训练或为服务选择空闲GPU后运行：

```bash
export CUDA_VISIBLE_DEVICES=2,3

uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_bpx_full_low_mem_combined \
  --policy.dir="$CKPT"
```

另开终端确认：

```bash
curl http://127.0.0.1:8000/healthz
```

必须返回 `OK`。

## 3. 快速评估一条bag

优先选择未做亮度增强的原始bag。若数据根目录存在 `_bpx_audit/task_language.json`，脚本会按第一级任务目录自动读取精确训练language：

```bash
uv run --with rosbags==0.11.0 \
  examples/bpx/evaluate_bpx_bag_server.py \
  --bag-path path/to/任务目录/episode.bag \
  --host 127.0.0.1 \
  --port 8000 \
  --output-dir eval_outputs/combined_80000/task_episode_quick \
  --stride 30 \
  --max-queries 40 \
  --samples-per-frame 1
```

若bag不在带审计映射的原目录中，必须显式传入与训练一致的完整英文语言：

```bash
  --prompt "Pick up the red block and place it in the bucket."
```

脚本会在最终 `PASS` 行和 `metrics.json` 中记录实际使用的prompt；每次评估都必须核对。

## 4. 部署节奏详细评估

快速评估通过后，使用与当前异步控制重规划间隔一致的5帧步长：

```bash
uv run --with rosbags==0.11.0 \
  examples/bpx/evaluate_bpx_bag_server.py \
  --bag-path path/to/任务目录/episode.bag \
  --host 127.0.0.1 \
  --port 8000 \
  --output-dir eval_outputs/combined_80000/task_episode_stride5 \
  --stride 5 \
  --samples-per-frame 1
```

多次采样可粗看flow-matching随机性，但会成倍增加推理时间：

```text
--samples-per-frame 3
```

## 5. 输出文件

| 文件 | 含义 |
|---|---|
| `metrics.json` | 总体、逐关节、逐horizon指标及prompt、延迟和评估范围 |
| `per_joint_metrics.csv` | 18维MAE、RMSE、归一化MAE、首步相关系数 |
| `per_horizon_metrics.csv` | 30步动作块每个horizon的误差 |
| `summary.png` | horizon误差、逐关节归一化误差、推理往返延迟 |
| `joint_curves.png` | 整条示教action与模型首步预测曲线 |
| `demonstration_state_action.png` | 实测qpos与遥操action目标曲线 |
| `chunk_overlay.png` | 轨迹中点处完整30步预测与示教目标对比 |
| `camera_contact_sheet.png` | 实际送入策略的三路RGB抽帧 |
| `predictions.npz` | 原始预测、目标、状态、action及延迟数组 |

原始总体MAE混合了弧度与夹爪线性坐标，不适合单独比较checkpoint。跨关节、跨checkpoint优先查看 `aggregate_normalized_mae`、`per_joint_normalized_mae`，同时结合曲线和真机独立成功率评测。

## 6. 建议评估矩阵

至少对每种任务选择一条未增强bag。若要判断泛化能力，必须预留从未参与训练的独立bag；使用训练集bag只能说明拟合和数据路径是否正常。
