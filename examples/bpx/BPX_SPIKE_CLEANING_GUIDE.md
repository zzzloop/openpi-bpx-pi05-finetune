# BPX 示教数据毛刺审计与清洗规范

本文规定 BPX 实机示教数据的关节毛刺处理流程。适用于当前 BPX 机器人、30 Hz 采样、弧度制关节位置以及既定关节顺序。新批次必须遵循“原始数据只读、先审计、再清洗、最后验收”的顺序。

## 1. 适用工具

| 数据格式 | 只读审计 | 生成清洗副本 |
|---|---|---|
| ROS1 bag（当前采集格式） | `examples/bpx/audit_new_rosbags.py`、`examples/bpx/audit_bpx_rosbag_numeric.py` | `examples/bpx/clean_bpx_rosbags.py` |
| HDF5（旧59条格式） | `examples/bpx/clean_bpx_hdf5.py --dry-run` | `examples/bpx/clean_bpx_hdf5.py` |

亮度增强脚本 `augment_bpx_rosbag_brightness.py` 不负责毛刺清洗。顺序必须是：

```text
原始数据 → 结构审计 → 数值/毛刺审计 → 清洗副本 → 清洗验收
        → 可选亮度增强 → LeRobot转换 → norm stats → 训练
```

## 2. 不可违反的原则

1. 原始数据目录设为只读，不覆盖、不原地修改。
2. 清洗输出必须使用独立的同级目录，不能是原始目录本身，也不能放在原始目录内部。
3. 先保存只读审计报告并人工复核，再运行清洗脚本。
4. 自动清洗只修改实际反馈位置：ROS bag 的 `/data/joint_position` 或 HDF5 的 `observations/qpos`。
5. `action` 是遥操系统下发的训练目标，不自动平滑。发现疑似 action 毛刺时应单独审查、隔离或重新采集，不能直接把它当作反馈毛刺修改。
6. 图像、时间戳、其他话题、HDF5 属性和其他数据集必须保持不变。
7. 不以“清洗后候选数必须为零”为目标。持续真实动作、边界样本或有指令支持的突变可以被保留。
8. 修改阈值必须使用新的输出目录，并记录理由；不能对已有清洗目录用另一套阈值继续 `resume`。

## 3. 当前毛刺判定算法

算法仅针对时间上孤立的反馈异常：

1. 对每个关节独立计算半径3帧的滚动中位数，即观察7帧局部窗口。
2. 当前值偏离局部中位数超过阈值时，标记为候选点。
3. 向候选点前后寻找最近的非候选有效点，最大搜索距离为4帧。
4. 只有前后有效点彼此足够接近时才允许修复，以避免抹掉持续运动或真实阶跃。
5. 修复值为前、后有效点的算术平均：

```text
replacement = (previous_valid + next_valid) / 2
```

默认参数：

| 维度 | 候选偏差阈值 | 前后桥接阈值 |
|---|---:|---:|
| 普通关节 | 0.25 rad | 0.25 rad |
| 夹爪槽 10、11、19、20 | 0.02 | 0.05 |
| 窗口半径 | 3帧 | 最大参考距离4帧 |

这些参数只适用于当前30 Hz、当前单位和当前关节含义。如果采样频率、单位、关节顺序、夹爪编码或机器人本体发生变化，必须重新评估阈值。

## 4. ROS bag 标准流程（推荐）

以下命令均从 OpenPI 仓库根目录执行。示例变量使用相对路径，按实际目录修改：

```bash
export RAW_ROOT=data_work/raw_bags
export AUDIT_DIR=data_work/audit
export CLEAN_ROOT=data_work/cleaned_bags

mkdir -p "$AUDIT_DIR" "$CLEAN_ROOT"
```

`CLEAN_ROOT` 必须是 `RAW_ROOT` 的同级独立目录，禁止设置为 `RAW_ROOT/cleaned`。

### 4.1 准备 rosbags 运行路径

```bash
export ROSBAGS_PATH="$(uv run --with rosbags==0.11.0 python -c \
'import pathlib, rosbags; print(pathlib.Path(rosbags.__file__).resolve().parents[1])')"

test -d "$ROSBAGS_PATH"
```

本项目固定使用 `rosbags==0.11.0`。不要混用系统中其他版本。

### 4.2 结构审计

```bash
uv run --with rosbags==0.11.0 \
  examples/bpx/audit_new_rosbags.py \
  --root "$RAW_ROOT" \
  --rosbags-path "$ROSBAGS_PATH" \
  --sample-payloads \
  --output "$AUDIT_DIR/structural_audit.json"
```

至少人工确认：

- bag 数量及各任务目录数量正确；
- 预期图像和数值话题均存在；
- `--sample-payloads` 抽查的第一条 bag 图像是360×640，编码为 `rgb8` 或 `bgr8`；
- 数值话题消息类型和长度符合采集契约；
- 没有空 bag、异常短轨迹或明显错误任务目录。

这里的载荷抽样只覆盖第一条 bag，不代表全量图像验收。后续组合数据准备和 LeRobot 转换验证仍必须逐条检查图像契约。

### 4.3 严格数值与毛刺审计

```bash
uv run --with rosbags==0.11.0 \
  examples/bpx/audit_bpx_rosbag_numeric.py \
  --root "$RAW_ROOT" \
  --rosbags-path "$ROSBAGS_PATH" \
  --output "$AUDIT_DIR/numeric_audit.json"
```

该脚本严格检查以下四路数值话题：

```text
/data/action
/data/past_action
/data/joint_position
/data/joint_velocity
```

当前 bag 每帧应为25维；训练使用前23维，索引23、24的轮子维度不参与毛刺判定。

快速查看汇总：

```bash
uv run python -c \
'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); print({k:r[k] for k in ("bag_count","qpos_candidate_count","action_candidate_count","qpos_candidates_by_joint","action_candidates_by_joint")})' \
  "$AUDIT_DIR/numeric_audit.json"
```

审计成功不等于可以直接清洗，必须查看候选明细：

- `qpos_candidates`：实际反馈位置候选；
- `action_candidates`：遥操目标候选，只报告、不自动修改；
- `qpos_rejected_count` / `action_rejected_count`：因找不到可靠前后参考点或前后差异过大而拒绝修复的候选；
- `command_supports_original`：同帧 action 是否更接近原始 qpos。为 `true` 时，ROS bag 清洗器会保留原值；
- `replacement_value`：通过局部一致性判断后的建议均值。

遇到以下情况应停止自动清洗并人工调查：

- action 候选数量异常或集中在同一轨迹；
- 大量关节同帧同时跳变；
- 候选持续多帧且对应真实快速动作；
- 候选集中出现在轨迹起点、终点或通信中断附近；
- 关节维度、单位、频率与当前契约不一致。

### 4.4 单条 pilot 清洗

先选择一条包含候选的 bag 运行 pilot。`--only` 使用相对于 `RAW_ROOT` 的路径：

```bash
uv run --with rosbags==0.11.0 \
  examples/bpx/clean_bpx_rosbags.py \
  --source-root "$RAW_ROOT" \
  --output-root "$CLEAN_ROOT" \
  --audit-report "$AUDIT_DIR/numeric_audit.json" \
  --rosbags-path "$ROSBAGS_PATH" \
  --only "任务目录/样例.bag" \
  --verify \
  --min-free-gib 5
```

检查终端必须正常结束，并确认输出摘要中的 `repair_count` 与审计报告一致。

### 4.5 全量清洗

pilot 通过后运行全量命令：

```bash
uv run --with rosbags==0.11.0 \
  examples/bpx/clean_bpx_rosbags.py \
  --source-root "$RAW_ROOT" \
  --output-root "$CLEAN_ROOT" \
  --audit-report "$AUDIT_DIR/numeric_audit.json" \
  --rosbags-path "$ROSBAGS_PATH" \
  --verify \
  --min-free-gib 5
```

脚本的事务行为：

- 有修复的 bag 先写入 `.partial`，完成后再原子改名；
- 无修复的 bag 在同一文件系统优先建立硬链接，失败后才完整复制；
- 已存在的输出会被重新验证，不会静默覆盖；
- 发现单独残留的 `.partial` 会停止，要求先人工检查；
- `--verify` 会逐消息确认话题、类型、时间戳和未修改消息完全一致。

注意：硬链接节省空间，但原始文件与清洗目录中的文件指向同一份数据。脚本不会修改它们；后续也必须把两边都视为只读。如果要求物理独立副本，应把输出放到不同文件系统，或完成后另行复制并校验。

脚本可以在中断后用同一条命令继续：已经存在的完整输出会验证，缺少的文件会继续生成。不要删除或忽略 `.partial` 后直接重跑，必须先判断它是否是不完整文件。

`clean_bpx_rosbags.py` 当前还会写出本项目既有任务目录的 `task_language.json`。如果新批次增加或重命名任务目录，毛刺清洗本身仍可运行，但必须先在代码评审中更新任务语言映射，并在转换前检查每个目录都有正确 language；不得把缺失任务默认归入通用指令。

### 4.6 ROS bag 清洗验收

输出目录应包含：

```text
cleaned_bags/
├── 各任务目录/*.bag
└── _bpx_audit/
    ├── cleaning_summary.json
    └── task_language.json
```

检查源与输出 bag 数量：

```bash
find "$RAW_ROOT" -type f -name '*.bag' | wc -l
find "$CLEAN_ROOT" -type f -name '*.bag' | wc -l
find "$CLEAN_ROOT" -type f -name '*.partial' -print
```

检查清洗摘要：

```bash
uv run python -c \
'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); print({k:r[k] for k in ("processed_bags","approved_repair_count_in_full_audit","applied_repair_count")})' \
  "$CLEAN_ROOT/_bpx_audit/cleaning_summary.json"
```

对清洗副本再次运行数值审计：

```bash
uv run --with rosbags==0.11.0 \
  examples/bpx/audit_bpx_rosbag_numeric.py \
  --root "$CLEAN_ROOT" \
  --rosbags-path "$ROSBAGS_PATH" \
  --output "$AUDIT_DIR/cleaned_numeric_audit.json"
```

验收要求：

- 源与输出 bag 数量一致；
- 不存在 `.partial`；
- `processed_bags` 等于源 bag 数；
- `applied_repair_count` 等于按报告批准并实际应用的数量；
- action、图像、时间戳和其他消息未经修改；
- 剩余候选均能由审计报告解释，不能为了清零继续调低阈值。

## 5. HDF5 标准流程（旧格式）

HDF5 输入必须包含：

```text
action              (T, 23) float32
observations/qpos   (T, 23) float32
observations/images/*
```

### 5.1 设置独立目录

```bash
export HDF5_RAW_ROOT=data_work/raw_hdf5
export HDF5_CLEAN_ROOT=data_work/cleaned_hdf5
```

两者必须不同。正式清洗会复制完整HDF5，因此需要接近原数据总大小的额外空间，再加预留空间。

### 5.2 只读扫描

```bash
uv run examples/bpx/clean_bpx_hdf5.py \
  --input-dir "$HDF5_RAW_ROOT" \
  --output-dir "$HDF5_CLEAN_ROOT" \
  --dry-run
```

记录每个 episode 的 `repairs` 和 `rejected_candidates`。数量异常时不要进入正式清洗。

### 5.3 单条 pilot

```bash
uv run examples/bpx/clean_bpx_hdf5.py \
  --input-dir "$HDF5_RAW_ROOT" \
  --output-dir "$HDF5_CLEAN_ROOT" \
  --episodes 0 \
  --verify-unchanged \
  --min-free-gib 5
```

检查生成的：

```text
cleaning_audit.csv
cleaning_summary.json
.cleaning_records/episode_0.json
episode_0.hdf5
```

### 5.4 全量继续

```bash
uv run examples/bpx/clean_bpx_hdf5.py \
  --input-dir "$HDF5_RAW_ROOT" \
  --output-dir "$HDF5_CLEAN_ROOT" \
  --verify-unchanged \
  --min-free-gib 5 \
  --resume
```

`--resume` 会核对原文件大小、修改时间和清洗参数。原始文件发生变化或参数不同会拒绝继续，此时必须建立新的清洗输出目录。

### 5.5 HDF5 验收

```bash
uv run examples/bpx/validate_bpx_hdf5.py \
  --raw-dir "$HDF5_CLEAN_ROOT" \
  --action-horizon 30

uv run examples/bpx/clean_bpx_hdf5.py \
  --input-dir "$HDF5_CLEAN_ROOT" \
  --output-dir data_work/recheck_unused \
  --dry-run
```

第二条命令只读复查清洗结果。已修复的孤立毛刺应不再出现；保留的拒绝候选仍可能存在。

`--verify-unchanged` 会逐数据集核验：

- HDF5树结构、shape、dtype、压缩和属性不变；
- `observations/qpos` 精确等于预期修复结果；
- action、全部图像及其他数据集通过SHA-256分块摘要确认未变。

## 6. 可选亮度增强

只有清洗副本验收通过后才能进行亮度增强。脚本对每一条源 bag 生成一暗一亮两个版本，同一条 bag 的全部图像使用同一个倍率，不会逐帧随机；不同 bag 的倍率由固定种子独立生成，可复现。

```bash
uv run --with rosbags==0.11.0 \
  examples/bpx/augment_bpx_rosbag_brightness.py \
  --root "$CLEAN_ROOT" \
  --rosbags-path "$ROSBAGS_PATH" \
  --dark-gain-min 0.45 \
  --dark-gain-max 0.90 \
  --bright-gain-min 1.10 \
  --bright-gain-max 1.35 \
  --seed 20260901 \
  --verify \
  --min-free-gib 5
```

不要随意使用 `--overwrite-existing`。正常中断后，如果一暗一亮均已存在，脚本会验证并跳过；只存在其中一个时会拒绝继续，必须先人工调查。

## 7. 修改阈值的规范

只有在以下证据同时具备时才修改默认阈值：

1. 候选点已经定位到具体 bag、帧和关节；
2. 结合 action、前后反馈和图像确认它确实是传感器异常；
3. 当前阈值无法捕获，或错误捕获了大量正常动作；
4. 新阈值已在少量 pilot 上验证；
5. 使用新的输出目录并记录完整命令、Git提交和审计报告。

HDF5脚本支持显式参数：

```text
--window-radius
--joint-spike-threshold
--joint-bridge-threshold
--gripper-spike-threshold
--gripper-bridge-threshold
```

ROS bag审计脚本当前将已批准阈值固定在代码中。需要改变时应修改代码、增加测试并形成新版本，不能只手工编辑审计JSON中的 replacement 值。

## 8. 归档与可追溯性

每次清洗至少保存：

```text
原始数据只读位置及文件数量
结构审计报告
原始数值审计报告
清洗摘要和逐点修复记录
清洗后复审报告
实际执行命令
OpenPI Git commit
相关脚本SHA-256
人工复核结论
```

记录版本：

```bash
git rev-parse HEAD
sha256sum \
  examples/bpx/audit_bpx_rosbag_numeric.py \
  examples/bpx/clean_bpx_rosbags.py \
  examples/bpx/clean_bpx_hdf5.py
```

## 9. 最终门禁清单

- [ ] 原始数据未被修改，且仍可恢复。
- [ ] 输入格式、关节顺序、单位和30 Hz频率与当前BPX契约一致。
- [ ] 已完成结构审计和数值审计。
- [ ] 已人工检查所有 action 候选和异常集中的 qpos 候选。
- [ ] 已先完成单条 pilot。
- [ ] 清洗输出位于独立目录。
- [ ] 全量清洗启用了验证。
- [ ] 源/输出轨迹数量一致且无 `.partial`。
- [ ] 清洗摘要、逐点记录和复审报告已保存。
- [ ] action、图像、时间戳和未授权字段没有变化。
- [ ] 亮度增强在清洗验收后执行。
- [ ] LeRobot转换只读取最终验收通过的数据目录。

## 附录：Windows PowerShell命令模板

正文使用Ubuntu/Bash语法。在Windows PowerShell中，ROS bag核心流程可写为：

```powershell
$RawRoot = "data_work/raw_bags"
$AuditDir = "data_work/audit"
$CleanRoot = "data_work/cleaned_bags"

New-Item -ItemType Directory -Force $AuditDir, $CleanRoot | Out-Null
$RosbagsPath = (uv run --with rosbags==0.11.0 python -c "import pathlib,rosbags; print(pathlib.Path(rosbags.__file__).resolve().parents[1])").Trim()

uv run --with rosbags==0.11.0 `
  examples/bpx/audit_new_rosbags.py `
  --root $RawRoot `
  --rosbags-path $RosbagsPath `
  --sample-payloads `
  --output "$AuditDir/structural_audit.json"

uv run --with rosbags==0.11.0 `
  examples/bpx/audit_bpx_rosbag_numeric.py `
  --root $RawRoot `
  --rosbags-path $RosbagsPath `
  --output "$AuditDir/numeric_audit.json"

uv run --with rosbags==0.11.0 `
  examples/bpx/clean_bpx_rosbags.py `
  --source-root $RawRoot `
  --output-root $CleanRoot `
  --audit-report "$AuditDir/numeric_audit.json" `
  --rosbags-path $RosbagsPath `
  --verify `
  --min-free-gib 5
```

HDF5只读扫描与正式清洗：

```powershell
$Hdf5RawRoot = "data_work/raw_hdf5"
$Hdf5CleanRoot = "data_work/cleaned_hdf5"

uv run examples/bpx/clean_bpx_hdf5.py `
  --input-dir $Hdf5RawRoot `
  --output-dir $Hdf5CleanRoot `
  --dry-run

uv run examples/bpx/clean_bpx_hdf5.py `
  --input-dir $Hdf5RawRoot `
  --output-dir $Hdf5CleanRoot `
  --verify-unchanged `
  --min-free-gib 5
```

Windows记录脚本哈希：

```powershell
Get-FileHash -Algorithm SHA256 `
  examples/bpx/audit_bpx_rosbag_numeric.py, `
  examples/bpx/clean_bpx_rosbags.py, `
  examples/bpx/clean_bpx_hdf5.py
```
