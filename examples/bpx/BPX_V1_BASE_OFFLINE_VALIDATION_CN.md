# 一代基础集：三个 Checkpoint 的离线动作验证

这是**只读示教观测重放**：每个时刻把录制图像、状态和原英文指令交给模型，比较输出绝对动作与记录的遥操 `action`。不接机器人、不发布ROS指令、不模拟物体运动、不覆盖数据或模型，也不重新归一化。

## 本次默认路径

- Config：`pi05_bpx_full_low_mem_v1_base_estimated`。
- LeRobot：`/scratch/bpx_datasets/lerobot/bpx/bpx_v1_base_estimated_v001`，有效18维，RGB四路640×360；模型仍只使用左眼、左右腕三路。
- 实验：`/scratch/bpx_datasets/checkpoints/pi05_bpx_full_low_mem_v1_base_estimated/bpx_v2_humanoid_bs2_v001_new`。实验名的v2不代表二代本体。
- 候选：`selected_checkpoints/10000`、`selected_checkpoints/18000`；最后完整模型：`19999`。
- 划分：`/scratch/bpx_datasets/reports/bpx_v1_base_recovery_v001/validation_split.json`。
- 输出：`/scratch/bpx_datasets/eval_outputs/bpx_v1_base_20k/replay_<时间戳>/`，每次新建；已有目录拒绝覆盖。修改时仅指定新路径，不删除旧输出。

## 一条命令：先左/右各一条

先确认GPU2/3没有其他训练占用；本脚本不会停止任何其他进程。使用已有服务器环境，不执行安装/uv sync：

```bash
cd /home/kemove/zzk_data/openpi-bpx-pi05-finetune
bash scripts/evaluate_bpx_v1_base_checkpoints.sh
```

默认选episode20（左臂蓝块）和61（右臂蓝块）。每个Checkpoint是单独进程，完成释放后再加载下一份，不同时驻留三套模型。CPU1包装器保留`HWLOC_COMPONENTS=-x86`；默认可见GPU2、3，不预分配96%显存。推理沿用现有Policy实现，GPU可见数量不等于自动FSDP分片；实际推理显存以服务器运行结果为准。不会改动原训练batch/FSDP设置。

通过后可对原8条验证集执行同样的比较：

```bash
bash scripts/evaluate_bpx_v1_base_checkpoints.sh 18 20 25 31 40 41 51 61
```

这仍是**参与过选模的验证集诊断**，不是新的独立测试集。协同任务64–79都在训练集内，若用来重放只能检查拟合/流程，不能称独立泛化能力。其他运行可通过单次命令的`BPX_EVAL_RUN`、`BPX_EVAL_OUTPUT`、`BPX_DATA_ROOT`、`BPX_EVAL_CUDA`覆盖包装器默认路径/可见GPU，不修改全局环境。

## 结果看哪里

每个step/episode目录：

| 文件 | 含义 |
|---|---|
| `joint_curves.png` | 整条示教action与每个查询时刻的首步预测，18维分别显示 |
| `chunk_overlay.png` | 中间查询时刻的完整30步预测与示教动作块 |
| `camera_contact_sheet.png` | 实际送给Policy的原RGB观察预览，尚未做Policy内的224×224预处理 |
| `per_joint_metrics.csv` | 每维原单位MAE、RMSE、首步MAE；按每个seed评分再平均 |
| `predictions.npz` | 所有查询/seed的原始绝对动作、目标及整条示教动作，供逐帧复核 |
| `metrics.json` | 精确模型/数据/统计路径与身份、帧索引、提示词、seed、去噪参数及评估边界 |

最后的`comparison/`：

- `episode_000020_first_step.png`、`episode_000061_first_step.png`：同一张18维图上叠加10000/18000/19999与黑色示教action。
- 对应`*_chunk.png`：三模型30步动作块叠加。
- `paired_per_joint.csv`：三模型各episode/关节的误差明细，不把夹爪旧坐标与关节rad混成一个总物理分数。

如果设置不同数据、查询帧、提示词、噪声、归一化或去噪参数，比较器拒绝当作配对结果；缺一个模型的episode也拒绝，不静默丢失困难样本。若中途失败，保留已生成目录用于排查，重试使用新输出目录。

## 公平性与不能误读的地方

1. 每5帧查询一次（30Hz录制对应约6个起点/秒）；每次预测30步。只评估完整目标时域，显式包含最后一个完整动作块起点，不通过重复尾帧来打分。首步曲线不是闭环机器人轨迹，观察来自示教，不是模型之前动作产生的观察。
2. 三模型采用相同种子101/202/303，噪声由`SeedSequence([seed, episode_index, frame_index])`确定，不受模型加载顺序影响；每次10步去噪，和本次选模配置一致。新重放覆盖更多时间点，噪声规则不同于原Suite，不应将两份报告的分数直接混为同一评估。
3. 从每份Checkpoint自己的`assets/.../norm_stats.json`读取统计；通过现有输入/输出变换还原delta，模型补32维、输出23维再转有效18维。直接比较未归一化的**绝对控制目标**。不对预测做均值平滑/限速/限位/时移来降低误差。
4. 图上的均值/带状区域描述固定噪声样本的均值±标准差，**不是置信区间**，也不是机器人执行误差。分数按每个seed的误差计算，而不是先把正负错误平均抵消。不同起点的30步chunk重叠，不能把所有动作点当独立实验样本。
5. 原`action`没有因飞特恢复而改变，是遥操下发目标；恢复后的`observation.state`可能含估计反馈，不是实测运动真值。因此本工具不画“预测 vs 真实机器人已执行位置”来制造跟踪性能证据，也不证明控制稳定/抓取成功/完整VLM能力保留。
6. 重点检查当前执行侧手臂、夹爪闭合/释放、偏差累积与输出是否缩幅/停住。仅不动关节拟合很好、或动作变平滑，都不足以说明任务完成更好。原有同图换颜色/同义指令指标仍应结合画面阅读，不只追求CF差异大。

## 单独验证一个模型或一条数据

可直接使用`examples/bpx/evaluate_bpx_lerobot_checkpoint.py --help`查看参数。给`--checkpoint-dir`具体完整目录，或推荐`recommended.json`；`--episode-indices`选择数据，`--output-dir`必须新建且不在数据或Checkpoint内部。无需启动WebSocket推理服务或工控机。CPU1机器仍用`python -m openpi.bpx_studio.cpu_affinity --socket 1 -- <原命令>`包装；原HDF5/bag评估脚本不修改、不混用。

这里使用项目锁定的[LeRobot v2.1本地读取API](https://github.com/huggingface/lerobot/blob/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5/lerobot/common/datasets/lerobot_dataset.py)，不是当前v3迁移接口；Hub/datasets均设置为离线，缺失源文件时报错而不下载补数据。HF缓存、临时文件由包装器落到scratch；原始LeRobot和权重只读。

## 验证状态

本机按用户约定只检查Python AST、Bash语法、代码不变量与Git差异，不安装环境，不运行模型、转换数据或渲染实验图。回归规格覆盖查询边界、同噪声配对、非有限值、保留每个seed的误差、候选路径以及比较身份不一致；需由服务器运行：

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
.venv/bin/python -u -m openpi.bpx_studio.cpu_affinity --socket 1 -- \
  .venv/bin/python -m pytest examples/bpx/evaluate_bpx_lerobot_checkpoint_test.py -q
```

这些单元规格不替代真实checkpoint加载、视频解码和显存验收。不要在未执行时称“已得到对比图/推理通过”。
