# 训练监控、能力保留诊断与候选 Checkpoint

## 不改变什么

**训练目标、compute_loss、train_step 的梯度计算、优化器、学习率、训练 batch/FSDP、训练噪声随机序列均不因这些指标改变。** 不加入 NC-DFM、注意力损失或任何辅助损失。CF 公式仅在 model.eval 的前向评估中计算，不参与反向传播，不自动控制机器人、补采或提前停训。

定期评估会额外消耗时间、显存和少量磁盘；不承诺零开销。它复用训练模型参数，不另加载第二份完整模型。支持单主机 JAX Pi0/Pi05 + LeRobot；多主机、PyTorch 和 FAST 当前不支持这套速度场评估。

## 原有监控与新增持久化

根目录随原训练参数，推荐仍是 `/scratch/bpx_datasets/checkpoints/<config>/<experiment>/`：

```text
monitoring/
  train_loss_steps.jsonl       # 每个优化批次原始 loss，新增
  metrics.jsonl / metrics.csv  # 原有 log_interval 汇总指标
  validation.jsonl            # 固定训练参考 / 验证 loss，含标准误差
  validation_protocol.json    # 分组、采样、模式、归一化身份
  validation_summary.json
  training_overview_latest.png
  loss_trend_latest.png
  generalization_latest.png   # train-reference / val 对比
  selection/
    protocol.json / training_config.json
    suite.json / probes.npz   # 冻结场景、指令、目标与真实噪声数组
    evaluations.jsonl / evaluations.csv
    selection_latest.png
    recommended.json         # 推荐路径、step、状态、保留集合、理由和预警
    decision_<step>.json      # 每次决策快照
    probes/step_<step>.json   # 每个场景、每个 seed 的原始指标
    probes/step_<step>.npz    # 预测动作、目标动作、反事实动作、初始速度场差
selected_checkpoints/
  <step>/params/
  <step>/assets/
  <step>/selection.json
  <step>/evaluation/          # 协议、配置与冻结样本的硬链接
```

每批次 loss 在现有日志同步点批量写入，不为每一步新增 GPU 同步。默认 log_interval=100 时，突然断电最多可能丢失尚未同步的这一小段；正常退出或可捕获的 Python 中断会尝试保存待写指标并刷新图片，SIGKILL/断电无法保证执行收尾。

原始 loss 不去毛刺、不用于篡改选模分数。Hampel/EMA 仅用于已有显示曲线；验证和选模使用原始指标。公平比较的是同一 eval 模式下的 train_eval_loss 与 val_loss，而不是直接把含训练增强的优化 loss 当作验证 loss 的对照。

沿用原 checkpoint 目录命名：目录 `5000` 对应原循环标签 step=5000，实际 optimizer_step 通常为 5001。新增记录同时保留两者，不偷偷重命名旧流程。初始化基线标记为 step=-1、optimizer_step=0，不是已经保存的 Checkpoint。

## 指标的含义与证据边界

| 指标 | 做法 | 不能推出什么 |
|---|---|---|
| action_mse | 固定持出图像、状态、指令与噪声，采样动作对比示教目标；只计有效维度和非 padding 时域 | 离线误差低不等于真机抓取成功 |
| native_rmse_by_active_dimension | 将误差按动作归一化尺度换回各维原单位，保存在逐场景 JSON | 不将关节弧度和夹爪开度混合成一个“物理误差” |
| jerk_excess | 归一化动作轨迹按数据 FPS 求三阶差分 RMS，与同一示教轨迹比较 | 不是机器人闭环抖动、碰撞或稳定性证明 |
| s_cf | 用户给定公式：固定 O/状态，共享 epsilon，在初始速度场上计算平方 L2，再对 noise seeds 求平均 | 响应不同不等于语言理解正确，不是通用 VQA/VLM 测试 |
| s_cf_active | 同一公式仅计真实动作维度，排除模型填充维度，作为选模代理之一 | 与完整 32 维 s_cf 不同，不能混用阈值 |
| grounding_margin | 错指令动作对原示教的 MSE − 正确指令 MSE | 同一场景可有多条正确轨迹，正差也非完整语义证明 |
| s_paraphrase_active / instruction_specificity | 比较同义改写响应；specificity=CF/(CF+同义响应) | 没有人工复核的指令对，不能自动宣称语义能力保留 |

CF 使用的是 **平方 L2 的和，不是 RMS，也不是物理关节速度**。OpenPI 的 ODE 约定与公式相反：纯噪声端在 OpenPI t=1，目标在 t=0，因此这里把公式的噪声端 t=0 映射为 OpenPI t=1；时间反向导致的整体速度符号改变不影响差的平方范数。完整 `s_cf` 包含动作 horizon×模型维度；`s_cf_active` 仅为有明确标签的有效维度版本。分辨率、观察、状态、指令、动作空间、horizon、归一化、噪声数组和采样步数在所有候选间固定。

同源亮/暗增强副本归入同一组，先对种子和组内帧求平均，再估计组间标准误差，避免把同一条示教复制品当独立样本。图中 ±2 SE 是近似提示，不是严格置信区间或多重比较校正。布局/会话若仍相关，估计仍可能偏乐观。

动作平滑度默认从连续关节维度计算。可按名字排除 gripper/jaw/claw，但无名字的本体必须人工检查 continuous_dims；夹爪正常阶跃不应被当成连续手臂抖动。

## 准备与仪表盘使用

1. 首先按 `TRAIN_VALIDATION_CN.md` 创建分组验证清单，避免增强副本泄漏。用同一清单计算训练部分的归一化。
2. “模型训练 → 多指标 Checkpoint 选模”填写新的固定评估集输出目录，点击准备。默认最多 16 个场景、3 个固定 noise seeds。该准备任务在独立 CPU 进程执行；不创建模型权重。
3. 留空指令对 JSON 时，程序从持出组挑选固定起始帧，用其他数据任务的语言生成**未复核**反事实模板。它不证明图中两个目标同时存在，因此只能用于诊断，不能当成严格语义 benchmark。
4. 在服务器查看生成的 editable_case_spec.json，结合原视频确认每张场景中两条不同指令都有意义。填写真正不同措辞、但同义的 paraphrase_prompt。确认后设置 semantic_reviewed=true。不同指令分词后相同（如被截断）会拒绝准备。
5. 换一个新的输出目录，用人工复核 JSON 重建。原 suite.json/probes.npz 不要原地改；它们与划分、Config、模型形状、归一化和真实噪声数组绑定。
6. 界面中“启用的选模 Suite”填正式 suite.json，去噪步数默认 10。Checkpoint 总保留上限默认 3：启用选模时保留最新可续训完整模型 1 个 + 优选权重候选最多 2 个。Suite 留空不启用选模，此时总上限 3 表示保留最近 3 个完整模型。旧实验续训需保持原保留策略；此前未启用总上限的，清空该输入框。
7. 使用新实验名训练。开始优化前记录一次初始化基线；之后在每次原有 save_interval 与末尾保存时，等待真实保存完成，再评估实际推理使用的 EMA 参数（若启用 EMA），否则评估 current 参数。普通未启用选模的实验仍保留原验证模式。
8. 查看任务详情下方的多指标图、推荐路径与预警。推荐 step 与“最低验证 loss step”分开显示；不会自动启动下一轮训练。

如果旧模型已经看过全部训练数据，不能事后抽取其中一部分称为独立验证。启用本功能应从 base 新开真正分组训练，或使用原来就未参与训练的数据；无法补回历史上未记录的初始化敏感度。已有训练不能无协议地中途改成另一套评估后继续画可比曲线。

## 服务器命令

### 单帧相机 mask 的类型边界

若准备Suite报`Observation.image_masks`类型错误，实际值为`numpy.bool_`，是准备代码跳过了训练组batch/推理输入中的数组化步骤，并不是数据或归一化文件损坏。单帧有效标记应为shape `()`、dtype `bool`的NumPy数组；新版本在构造Observation前对输入树执行`np.asarray`，不加batch轴、不改已有数组精度或值，也不关闭类型检查。不修改BpxInputs、训练损失或统计。

该错误发生在输出目录创建前，更新代码后通常可以使用同一输出目录重新准备，无须重算归一化。若目录确实已存在，使用新的草稿目录，不覆盖或删除原文件。

服务器可选回归：

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
.venv/bin/python -u -m openpi.bpx_studio.cpu_affinity --socket 1 -- \
  .venv/bin/python -m pytest src/openpi/training/selection_probes_test.py -q
```

本机仅静态检查，未执行这些依赖测试或模型测试。

本机不安装环境、不运行训练或测试。服务器先准备 suite（路径以你真实的分组清单为准）：

可先在服务器执行纯 CPU 的指标与候选保留回归检查（不需要加载模型权重）：

```bash
cd ~/zzk_data/openpi-bpx-pi05-finetune
.venv/bin/python -m pytest src/openpi/training/checkpoint_selection_test.py -q
```

这只能验证指标与保留逻辑；JAX 前向、FSDP 显存与实际数据契约仍需要服务器短程训练验证。

```bash
cd ~/zzk_data/openpi-bpx-pi05-finetune
export HF_LEROBOT_HOME=/scratch/bpx_datasets/lerobot
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
.venv/bin/python scripts/prepare_checkpoint_probes.py \
  --config-name pi05_bpx_full_low_mem_multitask_v2 \
  --validation-manifest /scratch/bpx_datasets/reports/validation_split_v1.json \
  --output-dir /scratch/bpx_datasets/reports/selection_suite_draft
```

修改 draft/editable_case_spec.json 后，换正式目录重建：

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
.venv/bin/python scripts/prepare_checkpoint_probes.py \
  --config-name pi05_bpx_full_low_mem_multitask_v2 \
  --validation-manifest /scratch/bpx_datasets/reports/validation_split_v1.json \
  --case-spec /scratch/bpx_datasets/reports/selection_suite_draft/editable_case_spec.json \
  --output-dir /scratch/bpx_datasets/reports/selection_suite_reviewed
```

在你原来的完整训练命令中只增加：

```text
--validation-manifest=/scratch/bpx_datasets/reports/validation_split_v1.json
--selection-suite=/scratch/bpx_datasets/reports/selection_suite_reviewed/suite.json
--selection-keep=3
--selection-denoise-steps=10
```

仍沿用 CUDA 2,3、batch 2、FSDP 2、显存比例 0.96、save_interval=5000、keep_period=25000。新功能不会自动把这些换掉。续训使用同一 suite/划分/归一化/噪声/去噪步数，参数变化会拒绝续训。

本例仍是已经存在的一代 multitask_v2 Config，不是假设二代硬件适配已完成。二代必须先建立与其单位、关节顺序和动作空间匹配的训练 Config、数据和归一化；不能因为模型都填充到 32 维就混用。

## 如何筛选，而不是一味追求最低 loss

- 优先考虑实际可用、完整保存且指标有限的候选。把动作误差与验证 loss 在各自不确定性带内接近当前最好者作为优先区域；有冲突时保留 Pareto 替代，明确标记冲突。
- 在动作误差、验证 loss、相对示教的 jerk excess、指令 specificity（存在时）上构建 Pareto 前沿，不把不同量纲硬凑成任意加权总分。
- 在可比区域优先更早、动作稳定的候选。人工复核的 CF/同义指令对及正 grounding margin 提供额外支持，而不是只最大化 CF 数值。
- 有限候选覆盖推荐折中点、动作误差优点、指令 specificity 优点等，默认最多 3 个。
- 不使用 3000–6000 步硬窗口，不默认初始 CF 的 50%。趋势预警比较较早可靠参考带、连续多次评估以及验证-训练差距。默认 uncertainty_multiplier=2、patience=3 是可配置启发式，不是论文中已经验证的经验阈值。
- 未提供绝对适配标准或语义证据时，结果为 provisional_offline_candidate，即“暂定离线候选”。不能把一堆差模型中相对最好的一个说成抓取已经成形。
- 原始模型可能本来就不具备新动作空间的良好输出；初始化 CF 低也不自动等于预训练 VLM 退化。应结合校准样本、正负指令和独立 benchmark。

人工 JSON 中 gates 可选择填写：max_action_mse、max_jerk_excess、min_s_cf、min_s_cf_active、min_task_success、vlm_noninferiority_margin。缺失/为 null 表示未校准，不偷填 50% 等阈值。min_s_cf 对应完整公式值，min_s_cf_active 是有效维度版本，不能互换。低于 min_s_cf 的场景会给出原位反事实补采建议，但不会自动启动录制。

## 接入真机任务成功率与独立 VLM benchmark

动作策略没有现成的通用 VQA 文本输出接口，本功能不伪造 VLM 准确率。可将外部、人工复核的标准化测评结果通过 editable_case_spec.json 的 external_assessments 指向一个服务器 JSON；重建 suite 后训练会在每次评估时读取该文件并保留内容快照和 SHA-256。

格式示意（数字仅说明字段，不是示例实验结果）：

```json
{
  "schema_version": 1,
  "suite_sha256": "从 monitoring/selection/protocol.json 复制 suite_sha256",
  "run_dir": "/scratch/bpx_datasets/checkpoints/CONFIG/EXPERIMENT",
  "records": [{
    "checkpoint_step": 5000,
    "operator_reviewed": true,
    "task_success": {"successes": 0, "trials": 1, "protocol_id": "固定布局和试验规则版本", "evidence_uri": "本地视频/报告路径"},
    "vlm_retention": {"correct": 0, "total": 1, "baseline_correct": 0, "baseline_total": 1, "baseline_id": "未微调参考模型及版本", "protocol_id": "同一独立语义基准版本和固定样本清单", "evidence_uri": "本地测评报告"}
  }]
}
```

程序要求协议和证据路径非空，跨 Checkpoint 保持相同协议、固定 VLM 基线和样本数；具体样本是否匹配仍需人工保证。程序使用 Wilson 区间；任务成功率门槛作用于下界，VLM 非劣效门槛作用于“当前下界−基线的上界”。需要预先合理设置 min_task_success 与 vlm_noninferiority_margin，不能把“差异不显著”当成“能力没有退化”。试验必须独立、同一协议，结果只支持该 benchmark，不证明所有 VLM 能力保留。没有外部证据时，程序明确保留离线代理结论。

## 候选保存与下一轮微调

归档使用同文件系统的硬链接保留 params/assets，与不可变的 Orbax 保存结果共享数据块。普通 Checkpoint 后续被保留规则清理时，候选仍可读取。选模器只删除超过预算且带本程序 owner_run 标记的候选链接；普通完整 Checkpoint 始终由 Orbax 管理，不手工删除正在写入的模型。若文件系统不支持硬链接，报错并保留原模型，不暗中复制几十 GB 权重。

### 总量上限与续训（2026-10-31）

新增 `--checkpoint-limit=3`，统计**本次实验内**普通完整模型与优选权重归档的合计数量，不是额外再保留 3 个候选。启用 Suite 时，Orbax 只保留最新完整模型 1 个（含优化器，可 resume），候选最多 `min(selection_keep, 2)` 个；候选按已有多指标在线筛选，不是按最新或只按 val loss。即使“候选保留上限”仍填旧值 3，也自动限制为 2。未启用 Suite 时保留最近 3 个完整模型。

总上限优先于 `keep_period`，避免 25k 周期模型无限累积；`save_interval=5000` 不变。只有新模型成功提交后才由 Orbax 清理旧普通模型，保存/归档过程中允许临时文件与短暂超额，**不会为卡住瞬时数量而先删除唯一可续训模型**。每次保存与选模完成后最多 3 个；评估失败仍保留最新完整模型和之前至多 2 个候选。硬链接不额外复制权重块。日志、图表、Suite 和指标历史不算模型，不因该上限删除。

预算是在线有界保留：已被淘汰的权重不能日后仅凭保留的指标复活，不等于事后全历史最优搜索。保存失败时旧模型仍在；选模失败时不推荐未评分模型。其他实验、原始数据、base 权重均不受影响。旧实验恢复时不要悄悄更换保留策略或 Suite；新策略用于新实验。恢复旧的大目录不会启动时立即删历史模型，待下一次成功保存再由管理器执行策略。

仪表盘默认总上限 3；命令行仅 `pi05_bpx_full_low_mem_v1_base_estimated` 预置 3，其他旧 Config 默认 `None`，无显式参数时保持旧保留方式。训练损失、数据集、归一化、学习率、batch/FSDP 不变。

服务器回归检查（本机不运行 OpenPI 环境）：

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
  src/openpi/training/checkpoint_retention_test.py \
  src/openpi/training/bpx_config_test.py \
  src/openpi/training/checkpoint_selection_test.py -q
```

CPU1 专用机器需在已绑定 CPU1 的终端执行，或使用现有 `openpi.bpx_studio.cpu_affinity --socket 1 -- ...` 入口；不需要重新安装环境。回归包含微小数组的真实 Orbax 异步保存、总量统计、早期候选硬链接存活及旧实验保护，不加载 base 权重或训练机器人模型。

recommended.json 的 checkpoint_path 可直接用于 serve_policy.py 的 --policy.dir。下一轮微调使用新的实验名，在原完整训练命令中增加：

```text
--init-checkpoint=/scratch/bpx_datasets/checkpoints/CONFIG/OLD_EXPERIMENT/monitoring/selection/recommended.json
```

这是**只加载权重**的新训练，optimizer、step 和随机序列从新实验初始化；不是原实验 --resume。新数据的归一化仍按新数据计算，并保持关节含义、动作定义和模型架构兼容。不要对初始化权重所在的原实验使用 --overwrite。原实验正常续训仍使用原来的 --resume。

选模失败会单独保存 evaluation_error_<step>.json、打印错误并保留先前推荐，不把失败指标当作好分数。监控与选模代码已编写，但遵照用户要求没有在本机执行测试、训练或 GPU 验证；先在服务器短程确认，再用于长跑。
