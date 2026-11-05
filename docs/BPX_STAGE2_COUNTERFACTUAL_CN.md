# BPX 第二阶段：四组同起点实验

原有仪表盘 Config、两卡启动方式和旧实验路径保留。第二阶段在「模型训练 → 训练流程」显式选择；仅切换表单，不提交任务，也不改第一阶段 checkpoint。

| 流程 | 更新参数 | 训练目标 / 推理 |
|---|---|---|
| 1 · 原始 CFM | 全量 | OpenPI 原 CFM / 常规去噪 |
| 2 · CFG 对照 | 全量 | CFM，逐样本 15% 置空任务文本，图像和归一化后的离散状态保留 / CFG 默认强度 2，OpenPI 时间 `t≥0.7` 时使用 |
| 3 · 冻结 VLM | 动作专家和动作/时间投影 | 原 CFM / 常规去噪。不使用 LoRA |
| 4 · CCFA | 全量 | 四角色共享同一高斯噪声和时间，CFM + 1.0×NC-DFM + 0.1×(语义余弦 + 对应位置对称 KL) / 常规去噪 |

四组用同一份四元组数据、划分、归一化和采样顺序，默认从原 18000 权重开新优化器。四张显卡的设备编号是 `0,1,2,3`，不是四元组的角色号。Global BS=4、FSDP=4，每卡一个样本，一个完整四元组只做一次优化器更新。四组用同一组内图像增强随机参数以保留 patch 空间对应；第 1–3 组的 CFM 噪声和时间仍按样本独立采样。本版未引入草稿中的旧数据回放、分支梯度缩放或跨本体混训。

## 1. 放置数据与四元组清单

默认 LeRobot 路径为 `/scratch/bpx_datasets/lerobot/bpx/bpx_v1_counterfactual_v001`，须使用现有一代 BPX 18 维状态/动作语义、30 Hz、左眼和两腕相机，右眼视频可以在数据中保留。只有与 18000 兼容的一代新示教进入本轮。不要把已有估计反馈集、未配对的可选端篮任务或另一机器人数据拼进这个目录。工具检查字段和元数据结构，但不能验证机械关节顺序和单位；交付时需人工核对。

数据到位后，在服务器 OpenPI 代码目录取元数据指纹：

```bash
.venv/bin/python -m openpi.training.bpx_counterfactual \
  --dataset-root /scratch/bpx_datasets/lerobot/bpx/bpx_v1_counterfactual_v001 \
  --fingerprint
```

按实际录制及人工复核填写 `/scratch/bpx_datasets/reports/bpx_v1_counterfactual_v001/quadruples.json`。下面仅展示一组，实际清单必须覆盖目标数据集的每个 episode，每个任务族至少两个独立布局组。

```json
{
  "schema_version": 1,
  "metadata_sha256": "用上面命令输出替换",
  "groups": [{
    "group_id": "task3_g01",
    "family": "task3",
    "split_group": "task3_layout01",
    "episodes": [3, 0, 2, 1],
    "windows": [[180, 192], [194, 208], [201, 216], [188, 200]],
    "attention_semantic": [[0, 1], [2, 3]],
    "attention_swap": [[1, 2]]
  }]
}
```

- `episodes` 始终按 `[Oa/I1, Oa/I2, Ob/I1, Ob/I2]` 排列，不按录制顺序。第 0、2 条任务文本必须一致，第 1、3 条也必须一致，两条指令不同。
- `windows` 是四个 episode 各自的**起始参考帧闭区间**。每个参考帧都必须能取出真实 30 帧动作。先按同一物理阶段标注，例如任务 3 移走纸巾盒、左臂退出、右臂尚未伸向杯子；不可用整段轨迹百分比替代。训练在四个已标注窗口里采样相同相对位置。
- 复用同一布局或有增强副本的组必须共享 `split_group`，不能跨训练和验证。清单按组划分，禁止同组泄漏。
- `attention_semantic` 只填同布局、需要看不同目标的有效边；`attention_swap` 只填实际复核目标空间位置对应的边。任务 3 只允许对角 `[1,2]`，不能加 `[0,3]`。任务 1 篮筐遮挡或视角不对应的对角边留空。边未填写即不参与注意力监督；第 4 组要求整个训练划分中两类注意力边都至少存在一个。
- NC-DFM 用有效 18 维，在 OpenPI 的 `t=1` 噪声端对 `v_i-v_j` 监督 `A_j-A_i`；CFM 仍按原 32 维计算。独立验证显示的是带完整文本的 CFM，并非第 4 组训练总损失。

## 2. 仪表盘操作

1. 「模型训练」选择任一第二阶段流程，确认显示的数据目录、四元组清单和 18000 初始化路径。
2. 点「按四元组生成验证清单」。输出不可覆盖；四组应使用同一份。如果已生成，直接填写已有清单，避免重新划分。
3. 「归一化」用该划分运行一次。四组共用 `assets/holdout-<digest>/pi05_bpx_cf_naive_full/bpx/bpx_v1_counterfactual_v001/norm_stats.json`，只统计训练 episode。
4. 可选：每组用同一份人工复核案例规范准备选模套件，输出目录和 `suite.json` 按组区分。缺省案例会选同一布局的配对指令，但仍标成未经人工复核。第 2 组轨迹评估使用 CFG，零步敏感度用未放大的条件速度，便于比较。
5. 给每组新实验名。四组都用四张卡，逐组运行。建议先做小步试跑（另用实验名，注意会保存 checkpoint），确认首批数据、损失与显存，再把同一正式步数用于四组。切组后请确认验证清单、初始化权重、步数和选模套件。断点续训清空初始化字段，保持原实验名、数据和统计不变。
6. 「模型推理」选对应 checkpoint，Config 会决定是否开启 CFG。比较第 2 组强度 1/2 时，可在脚本中调用 `create_trained_policy(..., sample_kwargs={"cfg_scale": 1.0})` 禁用放大；仪表盘默认 2.0。

第二阶段保留本地 Loss、验证和选模诊断，额外记录 CFM、NC-DFM、两项注意力损失或 dropout 比例。不同组的训练总 loss 不同口径，请用共同 CFM 验证、固定反事实评估和独立真机成功率比较。四卡真实显存和吞吐以服务器试跑为准。

## 3. 服务器合成回归（不需要真实数据或权重）

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q \
  src/openpi/training/bpx_counterfactual_test.py \
  src/openpi/training/bpx_stage2_jax_test.py
```
