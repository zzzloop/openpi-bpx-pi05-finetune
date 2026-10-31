# 一代80条基础数据：估计反馈恢复试验

这不是补回真实测量，而是尽量挽救这批数据的**独立估计反馈实验**。保留原始数据、原始state、命令和视频，所有推断都有标记。能否work必须通过服务器验证与后续实际评估，不能凭生成成功下结论。

用户当前数据仅在Windows：`C:\Users\zrobot\Desktop\一代基础数据集`。本机只编写代码，不运行拟合、转换或测试；以下命令由用户在服务器执行。

## 1. 固定三个目录，不覆盖旧数据

```text
原始数据：/scratch/bpx_datasets/lerobot/bpx/bpx_v1_base_raw
估计副本：/scratch/bpx_datasets/lerobot/bpx/bpx_v1_base_estimated_v001
评估报告：/scratch/bpx_datasets/reports/bpx_v1_base_recovery_v001
```

把本地这批数据的**目录内容**完整上传到原始目录。正确结构是：

```text
bpx_v1_base_raw/
  meta/info.json
  meta/episodes.jsonl
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.left_eye/episode_000000.mp4
  ...其余相机和episode...
```

不要多套一层`一代基础数据集/`；不要替换旧combined/multitask_v2数据。不必为这批已经有MP4的数据重新转视频。采集必须已停止，服务器使用冻结快照。

## 2. 更新代码与可选的服务器回归

```bash
cd ~/zzk_data/openpi-bpx-pi05-finetune
git remote -v
git status --short
git pull --ff-only origin agent/bpx-pi05-finetune
git log -1 --oneline
```

这里假定**服务器origin**是`zzzloop/openpi-bpx-pi05-finetune`，不是Physical-Intelligence上游。依赖用现有`.venv`，不执行uv sync，不升级LeRobot，不装新环境。

可先由用户在服务器做CPU回归：

```bash
.venv/bin/python -m pytest \
  src/openpi/bpx_studio/feedback_recovery_test.py \
  src/openpi/training/bpx_config_test.py -q
```

事务测试的媒体检查使用桩，只验证数值/元数据/源不变；它不代替下方真实数据审计、LeRobot SDK读取或GPU训练验证。作者遵照用户约束未在本机执行这些测试。

## 3. 一条命令做校准、留出验证并生成独立副本

确认原始路径正确、输出与报告目录尚不存在：

```bash
cd ~/zzk_data/openpi-bpx-pi05-finetune
bash scripts/recover_bpx_v1_feedback.sh \
  --apply \
  --accept-estimated-feedback \
  --workers 32 \
  --video-check full
```

脚本直接用项目`.venv/bin/python`，不联网、不安装依赖、不加载模型、不碰ROS。默认源/输出/报告就是第1节三个目录；可用`--source-root`、`--output-root`、`--report-dir`覆盖，三者必须互不嵌套。

如果只想先看估计误差，不生成副本，去掉两个开关，另选新的报告目录：

```bash
bash scripts/recover_bpx_v1_feedback.sh \
  --report-dir /scratch/bpx_datasets/reports/bpx_v1_base_recovery_analysis_v001
```

报告/输出都不覆盖。失败会保留明确命名的临时目录供检查，不自动删除任何原始文件；重试选新版本目录。本工具尚未实现断点续写，数据较小，禁止拿半成品训练。

## 4. 实际做了什么

- 编号0–5为校准来源；先排除孤立毛刺及明显越界的夹爪读数。只对可靠前后参考存在的1/2帧异常作标记的局部插值，不平滑整段真实轨迹。
- 对头部/双腕/左夹爪分别估计：过去命令的延迟、一阶响应、受限增益与偏置。使用实际采样时间，不假设采集精确30Hz。
- 使用**嵌套整episode留出**：外层留出的真实反馈不参与超参数选择，内层再选响应参数。报告保留每个关节、每条轨迹MAE/RMSE/P95，并与“上一时刻命令”基线比较。
- 前6条右爪没有闭合样本，因此从左爪迁移动态规律，仅用右侧已测张开位置校正零偏，再输入**右爪自己的历史命令**。不是复制同时刻左爪数值；右侧闭合/接触动态明确标记未独立验证。
- 编号6–79只替换已确认冻结的10维：`[4,5,6,7,12,13,14,15,16,17]`。若指定范围并非全段同一向量，立即拒绝，不覆盖有效反馈。
- 保留其他8维真实反馈、action及所有非state原字段；Parquet逐列读回核对。MP4不重编码，在同一文件系统硬链接，否则复制并校验哈希；元数据重新统计。
- 不使用同一帧/未来命令来制造精确目标标签；第一帧只能由初始目标近似初始化，启动不确定性单独标记。没有跨episode空档命令历史，不能把初始状态说成已恢复。

输出附加字段：

| 字段 | 含义 |
| --- | --- |
| `bpx.state_original` | 原始state，包含原先的冻结数值，永不冒充新测量 |
| `bpx.state_estimated_mask` | 哪些坐标由命令推断 |
| `bpx.state_repaired_mask` | 前6条哪些孤立异常做了局部修复 |
| `bpx.state_extrapolation_mask` | 哪些命令超出该关节校准范围 |
| `bpx.state_review_mask` | 启动近似、超范围或右侧迁移等需复核位置；不是统计置信区间 |

右爪、接触停位、负载变化、未见动作范围都不能凭左右相同换算公式保证准确。输出`observation.state`使用估计值只是为了让原训练映射可用于独立实验；来源在Parquet、`meta/bpx_feedback_recovery.json`、服务器报告以及推理metadata中始终保留。

## 5. 先看报告，不盲目训练

报告目录中：

```text
calibration_report.json        # 每维嵌套留出、基线、参数、限制
validation_dim_*.png           # 真实反馈 / 留出预测 / 命令对比
recovery_report.json           # 修改范围、超范围计数、复审结果
validation_split.json          # 本批试验的固定划分
```

无matplotlib时明确跳过PNG，不临时安装；JSON仍保留。没有右侧实测闭合曲线，不会生成假的“右爪验证准确率”。

看每个关节的留出误差、各折一致性、与基线差别、超范围比例，再决定是否值得试训。没有预先标定的通用MAE阈值，脚本不会把相对最优宣称为真实恢复成功。程序生成成功只保证派生流程与声明一致，不证明真机动作正确。

## 6. 验证划分已处理单例，不要再次点自动生成

新数据24种任务中，16种组合任务只有1条。工具保留它们在**训练侧**，不复制同一条冒充独立验证。

原始校准0–5也固定在训练侧，避免恢复模型见过的反馈被包装成独立政策验证来源。8种有重复示教的单动作任务各留出1条，预期72条训练、8条验证。具体episode编号在清单中固定，不要原地改动。

**验证只覆盖这些重复任务；没有对16个组合任务的独立泛化结论。** 估计状态上的验证loss和选模仅为代理，还可能有同场景相关性，不能证明真实闭环成功率或通用VLM语义保留。

## 7. 仪表盘具体填写

启动/重启服务器Studio并刷新网页：

```bash
cd ~/zzk_data/openpi-bpx-pi05-finetune
bash scripts/start_bpx_studio.sh
```

| 位置 | 填写 |
| --- | --- |
| 数据审计目录 | `/scratch/bpx_datasets/lerobot/bpx/bpx_v1_base_estimated_v001` |
| 数据审计报告 | `/scratch/bpx_datasets/reports/bpx_v1_base_recovery_v001/dashboard_audit.json` |
| 清洗输出目录（此时作为当前派生数据链位置） | `/scratch/bpx_datasets/lerobot/bpx/bpx_v1_base_estimated_v001` |
| 归一化Config name | `pi05_bpx_full_low_mem_v1_base_estimated` |
| 模型训练Config name | 同上 |
| 启用的划分清单 | `/scratch/bpx_datasets/reports/bpx_v1_base_recovery_v001/validation_split.json` |
| 归一化workers | `32` |
| Experiment name | `bpx_v1_base_estimated_bs2_v001`，必须是未使用的新名 |
| Checkpoint根 | `/scratch/bpx_datasets/checkpoints` |
| CUDA / 显存比例 / batch / FSDP | `2,3` / `0.96` / `2` / `2` |
| 保存 / 长期保留间隔 | `5000` / `25000` |
| 首次试验总steps | `20000`，不要照抄网页原默认200000；这是试验预算，不是最优步数保证 |

本配置已新增注册，指向估计副本，旧Config/统计/模型不改。按帧均匀采样，不让16个单例标签因任务等权变成约2/3采样概率；不增加任何辅助损失。

操作次序：

1. 先审计恢复副本，复核恢复报告。
2. 到模型训练页，手动填入工具生成的**启用的划分清单**，不再点“生成分组验证清单”。
3. 回归一化页，用新Config计算**训练部分**统计；显示的实际数据必须是estimated_v001，统计加载/写入一致，路径含holdout身份。
4. 当前版本会识别估计来源，显示警告并要求单独确认“仅作为估计反馈实验”，不会把恢复报告伪装成普通毛刺清洗PASS。
5. 回训练页准备固定评估Suite，先草稿目录 `/scratch/bpx_datasets/reports/bpx_v1_base_recovery_v001/suite_draft`；按图像复核反事实/同义指令后换 `suite_reviewed` 新目录重建。
6. “启用的选模Suite”填写正式`suite_reviewed/suite.json`，保留候选3、评估去噪10步，初始化checkpoint留空（从pi05_base开始），新实验模式。不能resume旧的全量训练。
7. 检查log/验证与选模结果，再决定是否延长；后续续训保持参数/清单/Suite一致，仅提高目标总steps并用resume。

推荐路径在`monitoring/selection/recommended.json`，候选保留在`selected_checkpoints/`。这是“估计数据上的候选”，不能自动称为“不过拟合或VLM能力保留的真机最佳模型”。

## 8. 重要边界

如果推理时Feetech反馈仍冻结，离线估计数据也不会修好在线反馈链。正式真机部署前仍须修复/验证实时反馈，不能因为训练loss下降就直接发布机器人命令。

当前交付的是恢复工具、独立Config和操作链；**未在本机运行拟合/恢复/测试，尚未生成用户的恢复数据集**。用户上传并在服务器运行后，需要依据实际输出判断可用程度。
