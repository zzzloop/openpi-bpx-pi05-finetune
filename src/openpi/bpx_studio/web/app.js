const $ = (id) => document.getElementById(id);
const defaultsVersion = "4";
const stored = [
  "openpiRoot", "hfLerobotHome", "checkpointRoot", "hfHome", "hfDatasetsCache", "tmpdir", "uvCacheDir",
  "auditDataset", "auditReport", "auditWorkers", "auditVideoCheck", "spikeSource", "spikeWorkers",
  "thresholdsPath", "spikeScanReport", "spikeCleanReport", "cleanOutput", "normConfig", "normWorkers", "exportSource",
  "exportOutput", "exportWorkers", "exportCompression", "compressionLevel", "exportEpisodes", "trainConfig", "expName",
  "cudaDevices", "memoryFraction", "batchSize", "fsdpDevices", "trainSteps", "saveInterval", "keepPeriod",
  "trainMode", "serveConfig", "serveCheckpoint", "serveCudaDevices", "serveMemoryFraction",
  "validationManifest", "validationOutput", "validationSource", "validationGrouping", "validationFraction",
  "validationInterval", "validationBatches", "lossView",
  "probeOutput", "probeCaseSpec", "probeCases", "selectionSuite", "selectionKeep", "selectionDenoise", "initCheckpoint",
];

let selectedJob = null;
let checkpointRecords = [];
let currentConfigProfile = null;
let currentSpikeReport = null;
let currentSpikeReportPath = "";
let spikeReportOffset = 0;
const spikePageSize = 100;
let lastAutoLoadedSpikeJob = "";
let pendingValidation = null;
let pendingProbe = null;
let recommendedCheckpoint = "";
let selectionJobId = null;
let selectedMetrics = [];

function value(id) {
  return $(id).value.trim();
}

function num(id) {
  return Number(value(id));
}

function save() {
  stored.forEach((id) => {
    if ($(id)) localStorage.setItem(`bpx-${id}`, $(id).value);
  });
}

function dataEnvironment() {
  return {
    hf_lerobot_home: value("hfLerobotHome"),
    validation_manifest: value("validationManifest"),
    hf_home: value("hfHome"),
    hf_datasets_cache: value("hfDatasetsCache"),
    tmpdir: value("tmpdir"),
    uv_cache_dir: value("uvCacheDir"),
  };
}

async function api(path, options = {}) {
  const headers = { ...options.headers };
  if (options.body) headers["Content-Type"] = "application/json";
  let response;
  try {
    response = await fetch(path, { signal: AbortSignal.timeout(20000), ...options, headers });
  } catch (error) {
    if (error.name === "TimeoutError" || error.name === "AbortError") {
      throw new Error("仪表盘后台请求超时；不是数据目录格式错误。后台训练任务不会因此自动停止。");
    }
    throw error;
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
  return body;
}

function toast(text, bad = false) {
  const node = $("toast");
  node.textContent = text;
  node.style.background = bad ? "#b4233a" : "#172b4d";
  node.classList.add("show");
  setTimeout(() => node.classList.remove("show"), 3000);
}

async function start(path, payload) {
  save();
  try {
    const job = await api(path, { method: "POST", body: JSON.stringify(payload) });
    selectedJob = job.id;
    toast(`任务已启动：${job.id}`);
    await refreshJobs();
  } catch (error) {
    toast(error.message, true);
  }
}

function normalizedPath(path) {
  return path.replaceAll("\\", "/").replace(/\/+$/, "");
}

function pathMatches(left, right) {
  return normalizedPath(left) === normalizedPath(right);
}

function confirmedSpikeIdentity() {
  return localStorage.getItem("bpx-confirmed-spike-report") || "";
}

function cleanReviewMatchesDataset(datasetPath) {
  return Boolean(
    currentSpikeReport?.applied
    && currentSpikeReport.clean_audit_passed
    && pathMatches(currentSpikeReport.output || "", datasetPath)
    && currentSpikeReport.report_identity === confirmedSpikeIdentity()
  );
}

function renderNormalizationReviewState(datasetPath) {
  const node = $("normReviewState");
  if (currentConfigProfile?.feedback_recovery && pathMatches(datasetPath, currentConfigProfile.dataset_path)) {
    node.textContent = "估计反馈试验 · 需单独确认";
    node.className = "warn-text";
  } else if (!pathMatches(datasetPath, value("cleanOutput"))) {
    node.textContent = "非当前清洗输出";
    node.className = "warn-text";
  } else if (cleanReviewMatchesDataset(datasetPath)) {
    node.textContent = "已确认 · 报告未变化";
    node.className = "ok-text";
  } else {
    node.textContent = "需查看并确认清洗报告";
    node.className = "warn-text";
  }
}

function setBinding(node, level, text) {
  node.className = `binding ${level}`;
  node.textContent = text;
}

function renderConfigProfile(profile) {
  currentConfigProfile = profile;
  $("normDatasetSource").value = profile.dataset_path;
  $("normRepoId").value = profile.repo_id;
  $("normStatsOutput").value = profile.computed_stats_path;
  $("normStatsLoaded").value = profile.loaded_stats_path;
  $("trainDatasetSource").value = profile.dataset_path;
  $("trainStatsSource").value = profile.loaded_stats_path;
  $("normPurpose").textContent = `配置用途：${profile.purpose}`;
  $("trainPurpose").textContent = `配置用途：${profile.purpose}`;
  $("normDatasetState").textContent = profile.dataset_exists ? "目录存在 · 可读取" : "目录不存在";
  $("normDatasetState").className = profile.dataset_exists ? "ok-text" : "warn-text";
  $("normStatsBinding").textContent = profile.stats_paths_match ? "写入 = 训练加载" : "路径不一致 · 已阻止";
  $("normStatsBinding").className = profile.stats_paths_match ? "ok-text" : "bad-text";
  $("normStatsSize").textContent = profile.computed_stats_exists
    ? `${human(profile.stats_size_bytes)} · 已生成`
    : "尚未生成 · 预计仅数 KB";
  $("normStatsSize").className = profile.computed_stats_exists ? "ok-text" : "muted-text";
  renderNormalizationReviewState(profile.dataset_path);

  $("flowRaw").textContent = value("spikeSource") || "—";
  $("flowClean").textContent = value("cleanOutput") || "—";
  $("flowConfigDataset").textContent = profile.dataset_path;
  $("flowStats").textContent = profile.loaded_stats_path;
  $("flowCheckpoint").textContent = value("serveCheckpoint") || value("checkpointRoot") || "—";

  const datasetMatches = pathMatches(profile.dataset_path, value("cleanOutput"));
  if (!profile.stats_paths_match) {
    setBinding(
      $("normBinding"),
      "bad",
      "配置错误：计算脚本写入的统计路径与训练加载路径不同。此 Config 不应直接重新计算归一化统计。",
    );
    $("flowStatus").className = "flow-status bad";
    $("flowStatus").textContent = "归一化资产路径未贯通";
  } else if (!datasetMatches) {
    setBinding(
      $("normBinding"),
      "warn",
      `数据链未贯通：毛刺清洗输出是 ${value("cleanOutput")}，但该 Config 实际读取 ${profile.dataset_path}。`,
    );
    $("flowStatus").className = "flow-status warn";
    $("flowStatus").textContent = "清洗输出与 Config 数据源不一致";
  } else if (!profile.dataset_exists) {
    setBinding($("normBinding"), "warn", "路径已贯通，但目标 LeRobot 数据集当前不存在；请先完成清洗输出。");
    $("flowStatus").className = "flow-status warn";
    $("flowStatus").textContent = "路径贯通，等待数据集生成";
  } else {
    setBinding($("normBinding"), "ok", "数据链已贯通：归一化和训练读取毛刺清洗后的同一个 LeRobot 数据集。");
    $("flowStatus").className = "flow-status ok";
    $("flowStatus").textContent = "RAW → CLEAN → CONFIG → STATS → MODEL 已贯通";
  }
}

function renderConfigError(error) {
  currentConfigProfile = null;
  $("flowRaw").textContent = value("spikeSource") || "—";
  $("flowClean").textContent = value("cleanOutput") || "—";
  $("flowConfigDataset").textContent = "无法解析";
  $("flowStats").textContent = "无法解析";
  $("flowCheckpoint").textContent = value("serveCheckpoint") || value("checkpointRoot") || "—";
  ["normDatasetSource", "normRepoId", "normStatsOutput", "normStatsLoaded", "trainDatasetSource", "trainStatsSource"]
    .forEach((id) => { $(id).value = "无法解析"; });
  $("normPurpose").textContent = `配置解析失败：${error.message}`;
  $("trainPurpose").textContent = $("normPurpose").textContent;
  $("normDatasetState").textContent = "Config 解析失败";
  $("normDatasetState").className = "bad-text";
  $("normStatsBinding").textContent = "无法验证";
  $("normStatsBinding").className = "bad-text";
  $("normStatsSize").textContent = "无法读取";
  $("normStatsSize").className = "muted-text";
  $("normReviewState").textContent = "Config 解析失败";
  $("normReviewState").className = "bad-text";
  setBinding($("normBinding"), "bad", "Config 必须先在 OpenPI 训练配置中注册，才能计算归一化统计或训练。");
  $("flowStatus").className = "flow-status bad";
  $("flowStatus").textContent = `Config 查询失败：${error.message}`;
}

async function loadConfigProfile() {
  const configName = value("normConfig");
  if (!configName) return renderConfigError(new Error("Config name 为空"));
  const query = new URLSearchParams({
    config_name: configName,
    hf_lerobot_home: value("hfLerobotHome"),
    validation_manifest: value("validationManifest"),
  });
  try {
    renderConfigProfile(await api(`/api/config-profile?${query}`));
  } catch (error) {
    renderConfigError(error);
  }
}

async function confirmDataBinding(action) {
  if (!currentConfigProfile) { toast("配置尚未解析成功，请修复配置后重试", true); return false; }
  if (!currentConfigProfile.stats_paths_match) {
    toast("归一化写入与训练加载路径不一致，操作已阻止", true);
    return false;
  }
  const recovered = currentConfigProfile.feedback_recovery;
  if (recovered) {
    if (!recovered.structural_audit_passed) {
      toast("估计反馈数据缺少完整发布/结构审计记录，不能启动训练或归一化", true);
      return false;
    }
    if (!recovered.validation_manifest || !pathMatches(value("validationManifest"), recovered.validation_manifest)) {
      toast(`请先在模型训练页的“启用的划分清单”填写：${recovered.validation_manifest || "恢复工具未生成有效清单"}`, true);
      return false;
    }
    return confirm(
      `这是估计反馈数据，不是真实测量修复。\n\n` +
      `数据集：${currentConfigProfile.dataset_path}\n` +
      `估计坐标：${recovered.estimated_coordinates}；超校准范围坐标：${recovered.extrapolated_coordinates}\n` +
      `右夹爪只做左右对称迁移，未独立验证闭合/接触动态。\n` +
      `评估报告：${recovered.report_path}\n\n` +
      `请使用恢复工具生成的验证清单，不要重新按全部24任务划分。\n` +
      `确认已复核报告，仅作为独立实验${action}？`,
    );
  }
  if (!pathMatches(currentConfigProfile.dataset_path, value("cleanOutput"))) {
    return confirm(
      `当前 Config 读取的是：\n${currentConfigProfile.dataset_path}\n\n` +
      `毛刺清洗输出是：\n${value("cleanOutput")}\n\n两者不同。确认仍要${action}？`,
    );
  }
  try {
    const cleanReport = await fetchSpikeReport(value("spikeCleanReport"), 0, 1);
    currentSpikeReport = cleanReport;
    currentSpikeReportPath = value("spikeCleanReport");
    renderNormalizationReviewState(currentConfigProfile.dataset_path);
    if (!cleanReport.applied || !cleanReport.clean_audit_passed) {
      toast("清洗报告尚未通过清洗后复审，操作已阻止", true);
      return false;
    }
    if (!pathMatches(cleanReport.output || "", currentConfigProfile.dataset_path)) {
      toast("已查看的清洗报告不属于 Config 实际读取的数据集", true);
      return false;
    }
    if (cleanReport.report_identity !== confirmedSpikeIdentity()) {
      toast("请先查看清洗报告并点击“确认清洗结果”", true);
      return false;
    }
  } catch (error) {
    toast(`无法验证清洗报告：${error.message}`, true);
    return false;
  }
  return true;
}

document.querySelectorAll("nav button").forEach((button) => {
  button.onclick = () => {
    document.querySelectorAll("nav button,.panel").forEach((node) => node.classList.remove("active"));
    button.classList.add("active");
    document.querySelector(`[data-panel='${button.dataset.tab}']`).classList.add("active");
  };
});

const auditModeDescriptions = {
  none: "仍会全量审计 Parquet、索引、时间戳和数值；视频只检查文件路径集合，不打开 MP4。",
  headers: "推荐日常使用：逐个打开 MP4，验证视频流、分辨率和 FPS，但不解码全部帧。",
  full: "冻结训练集前使用：在 Header 检查基础上解码每一帧，并核对视频帧数与 episode 帧数。",
};

function renderAuditMode() {
  const mode = value("auditVideoCheck");
  $("auditModeHelp").textContent = auditModeDescriptions[mode];
  document.querySelectorAll("#auditModeGuide [data-mode]")
    .forEach((node) => node.classList.toggle("active", node.dataset.mode === mode));
}

$("auditVideoCheck").addEventListener("change", () => {
  renderAuditMode();
  save();
});

$("startAudit").onclick = () => start("/api/jobs/audit", {
  dataset_root: value("auditDataset"), report_path: value("auditReport"), workers: num("auditWorkers"),
  video_check: value("auditVideoCheck"),
});

function spike(clean) {
  return start(clean ? "/api/jobs/spikes/clean" : "/api/jobs/spikes/scan", {
    source_root: value("spikeSource"), output_root: value("cleanOutput"), thresholds_path: value("thresholdsPath"),
    report_path: value(clean ? "spikeCleanReport" : "spikeScanReport"), workers: num("spikeWorkers"),
  });
}

$("scanSpikes").onclick = () => spike(false);
$("cleanSpikes").onclick = () => spike(true);

async function fetchSpikeReport(path, offset = spikeReportOffset, limit = spikePageSize) {
  const query = new URLSearchParams({
    path,
    offset: String(offset),
    limit: String(limit),
    feature: value("spikeFeatureFilter"),
  });
  const dimension = value("spikeDimensionFilter");
  if (dimension !== "") query.set("dimension", dimension);
  return api(`/api/spikes/report?${query}`);
}

function numeric(valueToFormat, digits = 6) {
  const parsed = Number(valueToFormat);
  return Number.isFinite(parsed) ? parsed.toFixed(digits) : "—";
}

function renderSpikeReport(report) {
  currentSpikeReport = report;
  $("unresolvedReview").textContent = report.unresolved_jumps == null
    ? "旧版报告未统计剩余大跳变，请重新扫描。"
    : `自动修复后仍有 ${report.unresolved_jumps} 个大跳变（相邻帧 × 维度计数）。阶跃、连续异常和边界跳变保留原值，需要人工复核。结构审计 PASS 不代表所有跳变已消除。`;
  $("unresolvedRows").innerHTML = (report.unresolved_examples || []).map((hit) =>
    `<tr><td>${escapeHtml(String(hit.episode_index ?? "—"))}</td><td>${escapeHtml(hit.feature)} · ${hit.dimension}</td><td>${hit.frame_index}</td><td>${numeric(hit.previous)}</td><td>${numeric(hit.current)}</td><td>${numeric(hit.delta)}</td></tr>`).join("") || "<tr><td colspan='6'>暂无记录</td></tr>";
  $("spikeTotal").textContent = report.applied
    ? `${report.total_repairs} 已修复`
    : `${report.total_repairs} 个候选`;
  $("spikeFiles").textContent = `${report.affected_files} / ${report.episode_files}`;
  $("spikeFeatureCounts").textContent = `${report.by_feature["observation.state"] || 0} / ${report.by_feature.action || 0}`;
  $("spikeAuditState").textContent = report.applied
    ? (report.clean_audit_passed ? "PASS" : "未通过")
    : "未清洗";
  $("spikeAuditState").className = report.clean_audit_passed ? "ok-text" : (report.applied ? "bad-text" : "muted-text");
  $("spikeReviewHint").textContent = report.applied
    ? `清洗报告：${report.report_path}`
    : `扫描报告：${report.report_path}`;
  $("confirmSpikeReview").disabled = !(report.applied && report.clean_audit_passed);
  $("confirmSpikeReview").textContent = report.report_identity === confirmedSpikeIdentity()
    ? "✓ 已确认，允许归一化"
    : "确认清洗结果，允许归一化";

  const first = report.matched_repairs ? report.offset + 1 : 0;
  const last = Math.min(report.offset + report.hits.length, report.matched_repairs);
  $("spikePage").textContent = `${first}–${last} / ${report.matched_repairs}`;
  $("spikePrev").disabled = report.offset === 0;
  $("spikeNext").disabled = report.offset + report.limit >= report.matched_repairs;
  $("spikeRows").innerHTML = report.hits.length
    ? report.hits.map((hit, index) => `<tr data-file="${hit.file_index}" data-repair="${hit.repair_index}">
        <td>${report.offset + index + 1}</td><td>${hit.episode_index ?? "—"}</td>
        <td>${escapeHtml(hit.feature)} · ${hit.dimension} ${escapeHtml(hit.dimension_name)}</td>
        <td>${hit.frame_index}</td><td>${numeric(hit.previous)}</td><td class="changed">${numeric(hit.original)}</td>
        <td>${numeric(hit.following)}</td><td>${numeric(hit.repaired)}</td><td><button>查看 ± 帧</button></td>
      </tr>`).join("")
    : "<tr><td colspan='9'>当前筛选条件没有毛刺点</td></tr>";
  document.querySelectorAll("#spikeRows tr[data-file]").forEach((row) => {
    row.onclick = () => loadSpikeTrace(Number(row.dataset.file), Number(row.dataset.repair), row);
  });
  if (currentConfigProfile) renderNormalizationReviewState(currentConfigProfile.dataset_path);
}

async function loadSpikeReport(path, resetOffset = true) {
  if (!path) return toast("毛刺报告路径为空", true);
  if (resetOffset) spikeReportOffset = 0;
  try {
    currentSpikeReportPath = path;
    renderSpikeReport(await fetchSpikeReport(path));
  } catch (error) {
    toast(error.message, true);
  }
}

function tracePath(frames, values, bounds) {
  const { xmin, xmax, ymin, ymax } = bounds;
  return frames.map((frame, index) => {
    const x = 58 + 808 * (frame - xmin) / Math.max(1, xmax - xmin);
    const y = 220 - 174 * (values[index] - ymin) / Math.max(1e-12, ymax - ymin);
    return `${index ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
}

function renderSpikeTrace(trace) {
  $("traceLabel").textContent = `EPISODE ${trace.episode_index ?? "?"} · FRAME ${trace.center_frame}`;
  $("traceTitle").textContent = `${trace.feature} · ${trace.dimension} ${trace.dimension_name}`;
  const allValues = [...trace.source_values, ...trace.review_values].map(Number);
  let ymin = Math.min(...allValues); let ymax = Math.max(...allValues);
  if (ymin === ymax) { ymin -= 1; ymax += 1; }
  const padding = (ymax - ymin) * 0.1;
  ymin -= padding; ymax += padding;
  const xmin = Math.min(...trace.frames); const xmax = Math.max(...trace.frames);
  const bounds = { xmin, xmax, ymin, ymax };
  const markerX = 58 + 808 * (trace.center_frame - xmin) / Math.max(1, xmax - xmin);
  $("spikeTraceChart").innerHTML = `
    <line x1="58" y1="220" x2="866" y2="220" stroke="#cad6e5"/>
    <line x1="58" y1="46" x2="58" y2="220" stroke="#cad6e5"/>
    <line x1="${markerX}" y1="38" x2="${markerX}" y2="220" stroke="#f0a24a" stroke-dasharray="6 5"/>
    <path d="${tracePath(trace.frames, trace.source_values, bounds)}" fill="none" stroke="#d1435b" stroke-width="3"/>
    <path d="${tracePath(trace.frames, trace.review_values, bounds)}" fill="none" stroke="#1678d4" stroke-width="3"/>
    <text x="58" y="242" font-size="11" fill="#718096">frame ${xmin} → ${xmax}</text>
    <text x="64" y="42" font-size="11" fill="#718096">${numeric(ymax, 4)}</text>
    <text x="64" y="216" font-size="11" fill="#718096">${numeric(ymin, 4)}</text>
    <text x="${Math.min(markerX + 6, 810)}" y="35" font-size="11" fill="#a56108">命中帧 ${trace.center_frame}</text>`;
  $("traceRows").innerHTML = trace.frames.map((frame, index) => {
    const changed = Math.abs(trace.source_values[index] - trace.review_values[index]) > 1e-10;
    return `<tr class="${frame === trace.center_frame ? "selected" : ""}"><td>${frame}</td>
      <td>${numeric(trace.timestamps[index], 4)}</td><td>${numeric(trace.source_values[index])}</td>
      <td class="${changed ? "changed" : ""}">${numeric(trace.review_values[index])}</td><td>${changed ? numeric(trace.review_values[index] - trace.source_values[index]) : "—"}</td></tr>`;
  }).join("");
  $("traceVectorRows").innerHTML = trace.frame_vector.map((item) => {
    const changed = Math.abs(item.source - item.cleaned) > 1e-10;
    return `<tr><td>${item.dimension} · ${escapeHtml(item.name)}</td><td>${numeric(item.source)}</td><td class="${changed ? "changed" : ""}">${numeric(item.cleaned)}</td></tr>`;
  }).join("");
}

async function loadSpikeTrace(fileIndex, repairIndex, row) {
  try {
    const query = new URLSearchParams({
      report_path: currentSpikeReportPath,
      file_index: String(fileIndex),
      repair_index: String(repairIndex),
      radius: String(num("spikeTraceRadius")),
    });
    renderSpikeTrace(await api(`/api/spikes/trace?${query}`));
    document.querySelectorAll("#spikeRows tr").forEach((node) => node.classList.remove("selected"));
    row.classList.add("selected");
  } catch (error) {
    toast(error.message, true);
  }
}

$("viewScanReport").onclick = () => loadSpikeReport(value("spikeScanReport"));
$("viewCleanReport").onclick = () => loadSpikeReport(value("spikeCleanReport"));
$("applySpikeFilter").onclick = () => loadSpikeReport(currentSpikeReportPath || value("spikeScanReport"));
$("spikePrev").onclick = () => {
  spikeReportOffset = Math.max(0, spikeReportOffset - spikePageSize);
  loadSpikeReport(currentSpikeReportPath, false);
};
$("spikeNext").onclick = () => {
  spikeReportOffset += spikePageSize;
  loadSpikeReport(currentSpikeReportPath, false);
};
$("confirmSpikeReview").onclick = () => {
  if (!currentSpikeReport?.applied || !currentSpikeReport.clean_audit_passed) return;
  localStorage.setItem("bpx-confirmed-spike-report", currentSpikeReport.report_identity);
  renderSpikeReport(currentSpikeReport);
  toast("清洗结果已确认；报告发生变化时需要重新确认");
};

$("startNormalize").onclick = async () => {
  await loadConfigProfile();
  if (!(await confirmDataBinding("计算归一化统计"))) return;
  start("/api/jobs/normalize", {
    ...dataEnvironment(), openpi_root: value("openpiRoot"), config_name: value("normConfig"),
    workers: num("normWorkers"),
    validation_manifest: value("validationManifest"),
  });
};
$("startExport").onclick = () => {
  const rawEpisodes = value("exportEpisodes");
  if (rawEpisodes && !/^\d+(?:\s*[,，]\s*\d+)*$/.test(rawEpisodes)) {
    toast("episode 请填写非负整数，用逗号分隔，例如 0,2,5；留空表示全部");
    return;
  }
  start("/api/jobs/export/hdf5", {
    source_root: value("exportSource"), output_root: value("exportOutput"), workers: num("exportWorkers"),
    compression: value("exportCompression"), compression_level: num("compressionLevel"),
    episode_indices: rawEpisodes ? rawEpisodes.split(/[,，]/).map(Number) : null,
  });
};
$("startTrain").onclick = async () => {
  await loadConfigProfile();
  if (!(await confirmDataBinding("启动训练"))) return;
  start("/api/jobs/train", {
    ...dataEnvironment(), openpi_root: value("openpiRoot"), config_name: value("trainConfig"),
    exp_name: value("expName"), checkpoint_base_dir: value("checkpointRoot"), cuda_devices: value("cudaDevices"),
    memory_fraction: num("memoryFraction"), batch_size: num("batchSize"), num_train_steps: num("trainSteps"),
    fsdp_devices: num("fsdpDevices"), save_interval: num("saveInterval"), keep_period: num("keepPeriod"),
    mode: value("trainMode"),
    validation_manifest: value("validationManifest"), validation_interval: num("validationInterval"),
    validation_batches: num("validationBatches"),
    selection_suite: value("selectionSuite"), selection_keep: num("selectionKeep"),
    selection_denoise_steps: num("selectionDenoise"), init_checkpoint: value("initCheckpoint"),
  });
};
$("createProbeSuite").onclick = async () => {
  try {
    const output = value("probeOutput");
    const job = await api("/api/jobs/selection-suite", {method: "POST", body: JSON.stringify({
      ...dataEnvironment(), config_name: value("trainConfig"), output_dir: output,
      case_spec: value("probeCaseSpec"), max_cases: num("probeCases"),
    })});
    pendingProbe = {id: job.id, output}; selectedJob = job.id;
    toast("正在准备固定选模样本；未经人工复核的指令对仅供诊断");
    await refreshJobs();
  } catch (error) { toast(error.message, true); }
};
$("useSelectedInit").onclick = () => {
  if (!recommendedCheckpoint) return;
  $("initCheckpoint").value = recommendedCheckpoint;
  save();
  toast("已填入初始化权重；请生成新实验名并选择新实验模式，不会自动启动训练");
};
$("createValidationSplit").onclick = async () => {
  try {
    const output = value("validationOutput");
    const result = await api("/api/jobs/validation-split", {method: "POST", body: JSON.stringify({
      dataset_root: value("trainDatasetSource"), output_path: output,
      source_manifest: value("validationGrouping") === "manifest" ? value("validationSource") : "",
      independent_episodes: value("validationGrouping") === "independent", fraction: num("validationFraction"),
    })});
    pendingValidation = {id: result.id, output};
    selectedJob = result.id;
    toast("正在分组划分；完成后自动填入启用清单，请重新计算归一化");
    await refreshJobs();
  } catch (error) { toast(error.message, true); }
};
$("validationManifest").addEventListener("change", () => {
  save(); loadConfigProfile();
  $("validationHint").textContent = value("validationManifest")
    ? "已选择独立验证。先用该清单重新归一化，再启动新实验；旧的全量训练不能补称独立验证。"
    : "未启用独立验证；仅显示训练 loss。";
});
$("startServe").onclick = () => start("/api/jobs/serve", {
  openpi_root: value("openpiRoot"), config_name: value("serveConfig"), checkpoint_dir: value("serveCheckpoint"),
  cuda_devices: value("serveCudaDevices"), memory_fraction: num("serveMemoryFraction"),
});
$("stopServe").onclick = async () => {
  try {
    const jobs = await api("/api/jobs");
    const running = jobs.find((job) => job.kind === "policy_serve" && ["queued", "running", "cancelling"].includes(job.status));
    if (!running) throw new Error("当前没有运行中的模型推理服务");
    await api(`/api/jobs/${running.id}/cancel`, { method: "POST" });
    selectedJob = running.id;
    toast("模型推理服务已停止");
    await refreshJobs();
  } catch (error) {
    toast(error.message, true);
  }
};

function human(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let index = 0;
  while (bytes >= 1024 && index < units.length - 1) {
    bytes /= 1024;
    index += 1;
  }
  return `${bytes.toFixed(index ? 1 : 0)} ${units[index]}`;
}

let systemRefreshing = false;
async function refreshSystem() {
  if (systemRefreshing) return;
  systemRefreshing = true;
  $("refreshSystem").disabled = true;
  try {
    const snapshot = await api(`/api/system?path=${encodeURIComponent(value("checkpointRoot"))}`);
    $("connection").textContent = "服务正常";
    $("connection").className = "pill ok";
    $("gpuGrid").innerHTML = snapshot.gpus.length
      ? snapshot.gpus.map((gpu) => `<div class="gpu"><strong>GPU ${escapeHtml(gpu.index)} · ${escapeHtml(gpu.name)}</strong>${escapeHtml(gpu.memory_used_mib)}/${escapeHtml(gpu.memory_total_mib)} MiB · ${escapeHtml(gpu.utilization_percent)}% · ${escapeHtml(gpu.temperature_c)}°C</div>`).join("")
      : escapeHtml(snapshot.gpu_error || "未检测到 NVIDIA GPU");
    $("systemUpdated").textContent = `更新于 ${new Date().toLocaleTimeString()}${snapshot.gpu_error && snapshot.gpus.length ? " · " + snapshot.gpu_error : ""}`;
    if (!snapshot.disk.error) {
      $("diskUsage").textContent = `${human(snapshot.disk.free)} 可用 / ${human(snapshot.disk.total)}`;
      $("diskMeter").value = 100 * snapshot.disk.used / snapshot.disk.total;
    }
    else { $("diskUsage").textContent = snapshot.disk.error; $("diskMeter").value = 0; }
  } catch (error) {
    $("gpuGrid").textContent = `状态读取失败：${error.message}。将自动重试，也可点击刷新。`;
    $("systemUpdated").textContent = "本次更新失败";
    $("connection").textContent = "连接失败";
    $("connection").className = "pill bad";
  } finally { systemRefreshing = false; $("refreshSystem").disabled = false; }
}

function escapeHtml(text) {
  const node = document.createElement("div");
  node.textContent = text;
  return node.innerHTML;
}

function fmt(valueToFormat) {
  return Number.isFinite(Number(valueToFormat)) ? Number(valueToFormat).toExponential(3) : "—";
}

function chart(id, points, key, color) {
  const svg = $(id);
  const values = points.filter((point) => point.metrics?.[key] != null)
    .map((point) => [Number(point.step ?? point.current), Number(point.metrics[key])]).filter((point) => point.every(Number.isFinite));
  if (values.length < 2) {
    svg.innerHTML = "";
    return;
  }
  const xs = values.map((point) => point[0]);
  const ys = values.map((point) => point[1]);
  const xmin = Math.min(...xs); const xmax = Math.max(...xs); const ymin = Math.min(...ys); const ymax = Math.max(...ys);
  const path = values.map(([x, y], index) => `${index ? "L" : "M"}${20 + 560 * (x - xmin) / Math.max(1, xmax - xmin)},${150 - 125 * (y - ymin) / Math.max(1e-12, ymax - ymin)}`).join(" ");
  svg.innerHTML = `<path d="${path}" fill="none" stroke="${color}" stroke-width="3"/><text x="20" y="165" font-size="11" fill="#718096">${xmin} → ${xmax} steps</text>`;
}

function renderLossCharts(points) {
  const trends = points.map((p) => ({...p, metrics: {...p.metrics, loss: p.display?.loss_trend ?? p.metrics?.loss}}));
  chart("lossChart", value("lossView") === "raw" ? points : trends, "loss", "#2457e6");
  $("lossPeakCount").textContent = `${points.filter((p) => p.display?.loss_outlier).length} 个趋势离群点；原始值未删除`;
  const evaluations = points.filter((p) => Number.isFinite(p.metrics?.val_loss));
  const last = evaluations.at(-1)?.metrics;
  $("metricTrainEval").textContent = last ? fmt(last.train_eval_loss) : "—";
  $("metricValidation").textContent = last ? fmt(last.val_loss) : "—";
  $("metricGap").textContent = last ? fmt(last.generalization_gap) : "—";
  $("metricBestStep").textContent = last ? String(last.best_validation_step) : "—";
  $("overfitHint").textContent = !last ? "尚无独立验证记录；旧日志无法补算历史验证 loss。" : last.overfit_warning
    ? "提示：训练参考 loss 下降，而连续验证值高于历史最佳。疑似过拟合，请结合误差条和任务成功率判断；不会自动停训。"
    : "观察同模式原始 loss 的趋势，不要求两条线重合；最佳评估 step 不一定有保留的 checkpoint。";
  const svg = $("validationChart");
  if (!evaluations.length) { svg.innerHTML = ''; return; }
  const xs = evaluations.map((p) => Number(p.step ?? p.current));
  const ys = evaluations.flatMap((p) => [p.metrics.train_eval_loss,
    p.metrics.val_loss-2*(p.metrics.val_stderr || 0), p.metrics.val_loss+2*(p.metrics.val_stderr || 0)]);
  const lo = Math.min(...ys); const hi = Math.max(...ys);
  const x = (i) => 40 + 530 * (xs[i] - xs[0]) / Math.max(1, xs.at(-1)-xs[0]);
  const y = (v) => 190 - 145 * (v-lo) / Math.max(1e-12, hi-lo);
  svg.innerHTML = [['train_eval_loss','#2457e6','训练参考'],['val_loss','#f17835','验证集']].map(([key,color,label],k) => {
    const path = evaluations.map((p,i) => `${i ? 'L':'M'}${x(i)},${y(p.metrics[key])}`).join(' ');
    return `<path d="${path}" fill="none" stroke="${color}" stroke-width="2.5"/>`
      + evaluations.map((p,i) => `<circle cx="${x(i)}" cy="${y(p.metrics[key])}" r="3" fill="${color}"><title>step ${xs[i]}: ${p.metrics[key]}</title></circle>`).join('')
      + `<text x="${40+k*155}" y="22" fill="${color}" font-size="13">${label}（eval）</text>`;
  }).join('') + evaluations.map((p,i) => {
    const se = 2*(p.metrics.val_stderr || 0);
    return `<line x1="${x(i)}" x2="${x(i)}" y1="${y(p.metrics.val_loss-se)}" y2="${y(p.metrics.val_loss+se)}" stroke="#f17835" opacity=".45"/>`;
  }).join('') + `<text x="40" y="220" fill="#718096" font-size="12">${xs[0]} → ${xs.at(-1)} steps · loss ${lo.toExponential(2)} ~ ${hi.toExponential(2)} · 验证 ±2 batch SE</text>`;
}
$("lossView").addEventListener("change", () => { save(); renderLossCharts(selectedMetrics); });

async function show(job) {
  selectedJob = job.id;
  document.querySelectorAll(".job").forEach((node) => node.classList.toggle("selected", node.dataset.id === selectedJob));
  $("selectedKind").textContent = job.kind.toUpperCase();
  $("selectedTitle").textContent = job.id;
  if (selectionJobId !== job.id) {
    selectionJobId = job.id; recommendedCheckpoint = "";
    $("selectionPath").textContent = "等待当前任务的选模结果";
    $("selectionWarnings").textContent = "";
    $("selectionPlot").hidden = true;
    $("useSelectedInit").disabled = true;
  }
  const progress = job.progress || {};
  $("progressText").textContent = `${progress.current ?? 0} / ${progress.total ?? "?"}`;
  $("progressPercent").textContent = `${Number(progress.percent || 0).toFixed(1)}%`;
  $("progressBar").value = progress.percent || 0;
  $("metricLoss").textContent = fmt(job.metrics?.loss);
  $("metricGrad").textContent = fmt(job.metrics?.grad_norm);
  $("metricLr").textContent = fmt(job.metrics?.learning_rate);
  $("metricStatus").textContent = job.status;
  $("commandText").textContent = `${Object.entries(job.environment || {}).map(([key, val]) => `${key}=${val}`).join("\n")}\n${job.command}`;
  $("cancelJob").hidden = !["queued", "running", "cancelling"].includes(job.status);
  try {
    $("jobLog").textContent = (await api(`/api/jobs/${job.id}/log`)).text || "日志尚未产生";
    const metrics = await api(`/api/jobs/${job.id}/metrics`);
    selectedMetrics = metrics;
    renderLossCharts(metrics);
    chart("normChart", metrics, "grad_norm", "#06a4ba");
    $("selectionResults").hidden = job.kind !== "train";
    if (job.kind === "train") {
      const report = await api(`/api/jobs/${job.id}/selection`);
      const labels = {not_evaluated: "尚未启用或尚未完成评估", insufficient_evidence: "证据不足",
        awaiting_resumed_evaluation: "已恢复较早训练状态，等待重新评估",
        provisional_offline_candidate: "暂定离线候选", offline_balanced_candidate: "离线多指标候选",
        candidate_with_external_benchmark_support: "有外部基准支持的候选",
        no_candidate_meets_user_gates: "尚无候选达到设定标准"};
      $("selectionStatus").textContent = labels[report.status] || report.status;
      $("selectionReason").textContent = `推荐 step：${report.recommended_step ?? "—"}；保留：${(report.retained_steps || []).join(", ") || "—"}`;
      recommendedCheckpoint = report.checkpoint_path || "";
      $("selectionPath").textContent = recommendedCheckpoint || "尚无推荐";
      const warningLabels = {
        overfit_suspected: "疑似过拟合：验证恶化、训练参考 loss 仍下降。",
        instruction_response_decline: "反事实语言响应持续衰减：建议复核并补采固定场景的对照指令。",
        validation_plateau_possible: "验证收益可能进入平台期，变化小于当前测量波动。",
        absolute_adaptation_not_calibrated: "未校准绝对任务适配标准：相对最优不代表已能完成抓取。",
        semantic_retention_unconfirmed: "语义保留证据不足：需要人工复核指令对、同义改写及正确性证据。",
        adaptation_worse_than_initial_reference: "动作预测比初始化参考恶化，请检查适配和数据。",
        action_validation_tradeoff: "动作误差与验证 loss 的较优区域存在冲突。",
        low_cf_scene_response: "部分固定场景的指令响应低于所设阈值，建议人工复核/补采。",
        external_vlm_benchmark_decline: "外部语义基准测评显示退化迹象；结论限于该基准。",
      };
      $("selectionWarnings").textContent = (report.warnings || []).map((message) =>
        warningLabels[message.split(":")[0]] || message).join("\n");
      $("useSelectedInit").disabled = !recommendedCheckpoint;
      if (report.status !== "not_evaluated") {
        $("selectionPlot").hidden = false;
        $("selectionPlot").src = `/api/jobs/${job.id}/selection-plot?version=${report.updated_at || report.recommended_step || 0}`;
      } else $("selectionPlot").hidden = true;
    }
  } catch (_error) { /* Job may finish between list and detail fetch. */ }
}

async function refreshJobs() {
  try {
    const jobs = await api("/api/jobs");
    if (pendingProbe && jobs.some((job) => job.id === pendingProbe.id && job.status === "succeeded")) {
      $("selectionSuite").value = `${pendingProbe.output.replace(/\/$/, "")}/suite.json`;
      pendingProbe = null; save(); toast("固定评估集已准备；请确认指令对是否已人工复核");
    }
    if (pendingValidation) {
      const completed = jobs.find((job) => job.id === pendingValidation.id && job.status === 'succeeded');
      if (completed) {
        $("validationManifest").value = pendingValidation.output;
        pendingValidation = null;
        $("validationManifest").dispatchEvent(new Event('change'));
      }
    }
    $("jobList").innerHTML = jobs.length
      ? jobs.map((job) => `<div class="job" data-id="${job.id}"><strong>${escapeHtml(job.kind)}</strong><br><small>${job.id} · ${job.status} · ${Number(job.progress?.percent || 0).toFixed(1)}%</small></div>`).join("")
      : "暂无任务";
    document.querySelectorAll(".job").forEach((node) => {
      node.onclick = () => show(jobs.find((job) => job.id === node.dataset.id));
    });
    if (selectedJob) {
      const job = jobs.find((candidate) => candidate.id === selectedJob);
      if (job) {
        await show(job);
        if (job.status === "succeeded" && job.id !== lastAutoLoadedSpikeJob) {
          if (job.kind === "spike_scan") {
            lastAutoLoadedSpikeJob = job.id;
            await loadSpikeReport(value("spikeScanReport"));
          } else if (job.kind === "spike_clean") {
            lastAutoLoadedSpikeJob = job.id;
            await loadSpikeReport(value("spikeCleanReport"));
          }
        }
      }
    }
  } catch (_error) { /* Connection status is handled by refreshSystem. */ }
}

$("cancelJob").onclick = async () => {
  if (selectedJob && confirm("确定终止当前任务？")) {
    await api(`/api/jobs/${selectedJob}/cancel`, { method: "POST" });
    refreshJobs();
  }
};
$("refreshJobs").onclick = refreshJobs;
$("refreshSystem").onclick = refreshSystem;

function applyCheckpointConfig() {
  const selected = checkpointRecords.find((item) => item.path === value("serveCheckpoint"));
  if (selected) {
    $("serveConfig").value = selected.config_name;
    $("flowCheckpoint").textContent = selected.path;
  }
  save();
}

async function loadCheckpoints(preferred) {
  try {
  checkpointRecords = await api(`/api/checkpoints?checkpoint_root=${encodeURIComponent(value("checkpointRoot"))}`);
  const select = $("serveCheckpoint");
  select.innerHTML = checkpointRecords.length
    ? checkpointRecords.map((item) => `<option value="${escapeHtml(item.path)}">${escapeHtml(item.config_name)} / ${escapeHtml(item.experiment_name)} / step ${item.step}</option>`).join("")
    : "<option value=''>没有找到完整 checkpoint</option>";
  const preferredRecord = checkpointRecords.find((item) => item.path === preferred);
  const sameConfig = checkpointRecords.find((item) => item.config_name === value("serveConfig"));
  if (preferredRecord) select.value = preferredRecord.path;
  else if (sameConfig) select.value = sameConfig.path;
  applyCheckpointConfig();
  $("startServe").disabled = !select.value;
  } catch (error) {
    checkpointRecords = [];
    $("serveCheckpoint").innerHTML = "<option value=''>扫描失败，请点击刷新重试</option>";
    $("startServe").disabled = true;
    toast(error.message, true);
  }
}

$("refreshCheckpoints").onclick = () => loadCheckpoints(value("serveCheckpoint"));
$("newExperiment").onclick = async () => {
  try {
    const query = new URLSearchParams({config_name: value("trainConfig"), checkpoint_root: value("checkpointRoot")});
    $("expName").value = (await api(`/api/next-experiment?${query}`)).experiment_name;
    save();
  } catch (error) { toast(error.message, true); }
};

async function synchronizeConfig(event) {
  const configName = event.target.value.trim();
  ["normConfig", "trainConfig"].forEach((id) => {
    if ($(id) !== event.target) $(id).value = configName;
  });
  try {
    $("expName").value = (await api(`/api/next-experiment?config_name=${encodeURIComponent(configName)}&checkpoint_root=${encodeURIComponent(value("checkpointRoot"))}`)).experiment_name;
  } catch (error) {
    toast(error.message, true);
  }
  await loadConfigProfile();
  save();
}

["normConfig", "trainConfig"].forEach((id) => $(id).addEventListener("change", synchronizeConfig));
$("serveCheckpoint").addEventListener("change", applyCheckpointConfig);
$("hfLerobotHome").addEventListener("change", loadConfigProfile);
$("cleanOutput").addEventListener("change", () => {
  $("exportSource").value = value("cleanOutput");
  save();
  loadConfigProfile();
});
$("spikeSource").addEventListener("change", loadConfigProfile);
$("checkpointRoot").addEventListener("change", async () => {
  await Promise.all([loadCheckpoints(value("serveCheckpoint")), refreshSystem()]);
  if (currentConfigProfile) renderConfigProfile(currentConfigProfile);
});

async function init() {
  renderAuditMode();
  if (localStorage.getItem("bpx-defaults-version") !== defaultsVersion) {
    ["auditDataset", "auditReport", "spikeSource", "spikeScanReport", "spikeCleanReport", "cleanOutput", "exportSource", "exportOutput", "normConfig", "trainConfig", "serveConfig", "expName"]
      .forEach((id) => localStorage.removeItem(`bpx-${id}`));
    localStorage.setItem("bpx-defaults-version", defaultsVersion);
  }
  stored.forEach((id) => {
    if ($(id) && localStorage.getItem(`bpx-${id}`) !== null) $(id).value = localStorage.getItem(`bpx-${id}`);
  });
  try {
    const config = await api("/api/config");
    const defaults = {
      openpiRoot: config.default_openpi_root,
      hfLerobotHome: config.hf_lerobot_home,
      checkpointRoot: config.checkpoint_root,
      hfHome: config.hf_home,
      hfDatasetsCache: config.hf_datasets_cache,
      tmpdir: config.tmpdir,
      uvCacheDir: config.uv_cache_dir,
      validationOutput: config.validation_split,
      auditDataset: config.raw_dataset,
      auditReport: config.audit_report,
      spikeSource: config.raw_dataset,
      spikeScanReport: config.spike_scan_report,
      spikeCleanReport: config.spike_clean_report,
      cleanOutput: config.clean_dataset,
      exportSource: config.clean_dataset,
      exportOutput: config.hdf5_export,
      normConfig: config.default_config_name,
      trainConfig: config.default_config_name,
      serveConfig: config.default_config_name,
      expName: config.default_experiment_name,
      probeOutput: config.selection_suite_output,
    };
    Object.entries(defaults).forEach(([id, val]) => {
      if (["cleanOutput", "exportSource"].includes(id) && pathMatches($(id).value, `${config.hf_lerobot_home}/bpx/bpx_v2_humanoid`)) $(id).value = val;
      if ($(id) && !$(id).value) $(id).value = val;
    });
    $("bpxConfigNames").innerHTML = config.known_config_names.map((name) => `<option value="${escapeHtml(name)}"></option>`).join("");
    await Promise.allSettled([
      loadCheckpoints(localStorage.getItem("bpx-serveCheckpoint")),
      loadConfigProfile(),
      refreshSystem(),
      refreshJobs(),
    ]);
  } catch (error) {
    toast(error.message, true);
  }
}

refreshSystem();
init();
setInterval(refreshJobs, 2000);
setInterval(refreshSystem, 15000);
setInterval(loadConfigProfile, 30000);
