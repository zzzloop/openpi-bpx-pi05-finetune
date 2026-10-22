# 训练 / 验证 loss 对比与趋势去毛刺

## 增加了什么

训练页新增“训练 / 验证对比与过拟合观察”。启用后记录：

- `loss`：原有训练优化过程的 loss，包含训练时图像增强等因素。
- `train_eval_loss`：固定训练参考样本，在 eval 模式下计算的 loss。
- `val_loss`：固定、未参与训练的验证样本，在相同 eval 模式下计算的 loss。
- `generalization_gap = val_loss - train_eval_loss`。
- 最佳验证 step、验证 batch 均值的标准误差估计、每次评估耗时。

**重点比较 `train_eval_loss` 和 `val_loss`，而不是要求带训练增强的 `loss` 与验证值相等。** 两侧按任务分层选固定帧、无放回抽样；每个任务至少一个参考帧。它们使用同样的 eval 模式和固定噪声/时间采样种子，减少测量随机性。每次只做前向计算，不更新参数、梯度或优化器，不另外复制一份完整模型。

当前针对 JAX + LeRobot v2.x 流程；不支持 RLDS 自动分组验证。保持原有 batch 2 / FSDP 2 / 学习率配置不变；数据选择变成训练划分，所以新实验的数据分布与原全量训练并不相同。

## 仪表盘操作

1. 完成数据审计和清洗，确认训练实际读取的数据集。
2. 进入“模型训练 → 训练 / 验证对比与过拟合观察”。
3. 旧的亮暗增强数据：填写**该数据集转换时使用的源 manifest**，选择“按源 manifest 绑定增强副本”。必须是与当前输出 episode 顺序、帧数、任务语言完全一致的清单。例如多任务 v2 使用其 `multitask_v2.json`，不要误用旧 combined 清单。
4. 点击“生成分组验证清单”。默认每个任务约 10% 的独立示教组进入验证；每个任务至少需要两组。源 bag 及其 `__aug_dark` / `__aug_bright` 副本始终在同一侧。不会重新复制或编码大数据集，只生成小型 JSON。
5. 成功后，界面自动填入“启用的划分清单”。到“归一化”重新计算，默认 32 workers。检查显示的统计路径包含 `holdout-…`，它与旧的全量统计隔离，训练和归一化使用相同清单。
6. 使用**新实验名**启动训练。默认每 1000 step 评估一次，每侧最多 16 batches（batch 2 时最多 32 帧）。样本首次加载后缓存在 CPU，减少重复视频解码；样本数太小导致某个任务覆盖不足时，会明确报错，请增加 batches。
7. 查看训练参考 / 验证双曲线、差距、最佳验证 step 和提示。可在“原始训练 loss”和“稳健去毛刺趋势”之间切换。

只有确实没有增强副本、重复示教时，才可选择“确认每条独立”。自动规则不会识别肉眼相似的固定轨迹或同场景重复示教；它们也可能造成偏乐观的验证。需要按布局/采集会话严格隔离时，可在源 manifest 的每条记录增加相同的 `group_id`，将相关示教归为一组。一个组当前必须对应同一任务；跨任务的同布局联合留出需要另建实验设计，不能把这里的 task 内留出当成跨场景测试。

## 已经在训练的实验怎么办

- 更新代码不会把新逻辑注入已经运行的 Python 进程，不要为了刷新界面强杀训练。
- 原实验不填写划分清单，行为保持不变；新趋势视图可显示仪表盘已有的原始日志。
- **如果原模型已经训练过全部数据，不能从这些数据里事后抽一部分，再称为未见过的验证集。** 要可靠观察过拟合，应从 base 新开分组实验，或者另外提供真正未见过的数据做独立评估。不会自动替换或删除你的现有模型。
- 开启分组验证后的续训必须使用相同清单、评估 seed、batch、固定样本数量和配置；协议变化会拒绝续训，防止曲线失去可比性。不要改变数据集 metadata。
- 划分清单不可原地覆盖。要改划分，生成新文件、重新计算统计，并使用新实验名。

## 如何判断是否还该继续训练

- 训练参考与验证 loss 都持续下降：通常还有改善空间。
- 训练参考下降，验证逐渐平台：继续训练的验证收益可能已经很小。
- 训练参考继续下降，验证持续上升、差距扩大：是过拟合的警示，需要看连续多次评估，而不是单个尖峰。
- 两者都上升或出现非有限值：也可能是训练不稳定，不能一律解释为过拟合。

当前提示规则：最近连续三次验证值都比历史最好高 2% 以上，同时这三次训练参考 loss 下降。它是启发式提示，**不是统计检验或自动早停**。图中的 `±2 batch SE` 是 batch 均值离散程度的近似提示，不是严格置信区间；同一 episode 内的帧有相关性，样本少时应增加评估 batches、换未见布局并测任务成功率。

最佳步数依据**原始 `val_loss`**，不会依据平滑曲线选择。它是“已评估步数中的最佳”，不代表所有 step 的全局最佳，也不保证该 step 正好保存了 checkpoint。现有保存间隔/保留周期不变，不自动复制大权重；请在仍然保留的 checkpoint 中选验证较好者测试。loss 低也不等于真机成功率高。

## 曲线去毛刺怎么处理

显示趋势使用 9 点局部 Hampel / MAD 离群检测，再做 EMA（alpha 0.2）。相较简单移动均值，可以减少孤立峰和短连续峰的视觉影响，并用独立纵轴显示趋势。

**不删除原始记录、不改示教数据、不改反向传播、不平滑验证值或泛化差距、不用平滑值触发“最佳步数”。** 持续恶化的趋势仍会显示。NaN/Inf 不会被涂成正常值；评估遇到非有限值会报错。端点证据不足时不会擅自判为毛刺，不能保证把所有真实随机波动“清除干净”。

保存位置：

```text
<checkpoint 根目录>/<config>/<experiment>/monitoring/
├── metrics.jsonl / metrics.csv       原始训练及定期评估指标
├── validation_protocol.json          固定评估协议
├── validation_split.json             运行时划分备份
├── validation.jsonl                  原始逐次评估结果
├── validation_summary.json           最佳步数和最新提示
├── generalization_latest.png         同模式双曲线和泛化差距
├── loss_trend_latest.png             原始训练 loss 与独立纵轴趋势对照
└── training_overview_latest.png      原有训练总览，保留原始峰值
```

## 命令行用法（可选）

先设置自己的 `HF_LEROBOT_HOME` 等环境变量。以现有 multitask v2 为例：

```bash
cd ~/zzk_data/openpi-bpx-pi05-finetune
export HF_LEROBOT_HOME=/scratch/bpx_datasets/lerobot
export SPLIT=/scratch/bpx_datasets/reports/validation_split_v1.json

.venv/bin/python -m openpi.training.validation_split \
  --dataset-root "$HF_LEROBOT_HOME/bpx/bpx_blocks_multitask_v2" \
  --source-manifest /scratch/bpx_datasets/bpx_pi05_multitask_v2/manifests/multitask_v2.json \
  --output "$SPLIT" --fraction 0.1 --seed 42

.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_bpx_full_low_mem_multitask_v2 \
  --validation-manifest "$SPLIT" --num-workers 32
```

以新实验名训练；不会改已有实验目录：

```bash
CUDA_VISIBLE_DEVICES=2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.96 \
.venv/bin/python scripts/train.py pi05_bpx_full_low_mem_multitask_v2 \
  --exp-name=bpx_multitask_v2_holdout_bs2_v001 \
  --checkpoint-base-dir=/scratch/bpx_datasets/checkpoints \
  --batch-size=2 --num-train-steps=200000 --fsdp-devices=2 \
  --save-interval=5000 --keep-period=25000 \
  --validation-manifest="$SPLIT" \
  --validation-interval=1000 --validation-batches=16
```

新实验不要加 `--overwrite`；同一分组实验续训时使用完全相同的配置，并增加 `--resume`。验证间隔须能被 `log_interval` 整除（现有默认 100）；可以增加 batches 或加大验证间隔来折中准确度和额外耗时。验证会增加前向计算与编译开销，具体双卡显存、耗时需在服务器短跑测量，不承诺零成本。

新模型推理仍使用相同 config 和它自己的 checkpoint，推理会加载该 checkpoint 内保存的训练部分统计，不需要改机器人端接口，也不需要在推理命令添加验证清单。
