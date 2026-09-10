const $ = (id) => document.getElementById(id);
const token = new URLSearchParams(location.search).get("token") || localStorage.getItem("bpx-token") || "";
if (token) localStorage.setItem("bpx-token", token);

const defaultsVersion = "3";
const stored = [
  "openpiRoot", "hfLerobotHome", "checkpointRoot", "hfHome", "hfDatasetsCache", "tmpdir", "uvCacheDir",
  "auditDataset", "auditReport", "auditWorkers", "auditVideoCheck", "spikeSource", "spikeWorkers",
  "thresholdsPath", "spikeReport", "cleanOutput", "normConfig", "normWorkers", "exportSource",
  "exportOutput", "exportWorkers", "exportCompression", "compressionLevel", "trainConfig", "expName",
  "cudaDevices", "memoryFraction", "batchSize", "fsdpDevices", "trainSteps", "saveInterval", "keepPeriod",
  "trainMode", "serveConfig", "serveCheckpoint", "serveCudaDevices", "serveMemoryFraction",
];

let selectedJob = null;
let checkpointRecords = [];
let currentConfigProfile = null;

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
    hf_home: value("hfHome"),
    hf_datasets_cache: value("hfDatasetsCache"),
    tmpdir: value("tmpdir"),
    uv_cache_dir: value("uvCacheDir"),
  };
}

async function api(path, options = {}) {
  const headers = { "X-BPX-Token": token, ...options.headers };
  if (options.body) headers["Content-Type"] = "application/json";
  const response = await fetch(path, { ...options, headers });
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
  setBinding($("normBinding"), "bad", "Config 必须先在 OpenPI 训练配置中注册，才能计算归一化统计或训练。");
  $("flowStatus").className = "flow-status bad";
  $("flowStatus").textContent = "Config 无法解析";
}

async function loadConfigProfile() {
  const configName = value("normConfig");
  if (!configName) return renderConfigError(new Error("Config name 为空"));
  const query = new URLSearchParams({
    config_name: configName,
    hf_lerobot_home: value("hfLerobotHome"),
  });
  try {
    renderConfigProfile(await api(`/api/config-profile?${query}`));
  } catch (error) {
    renderConfigError(error);
  }
}

function confirmDataBinding(action) {
  if (!currentConfigProfile) return confirm(`当前 Config 尚未成功解析，仍要${action}吗？`);
  if (!currentConfigProfile.stats_paths_match) {
    toast("归一化写入与训练加载路径不一致，操作已阻止", true);
    return false;
  }
  if (!pathMatches(currentConfigProfile.dataset_path, value("cleanOutput"))) {
    return confirm(
      `当前 Config 读取的是：\n${currentConfigProfile.dataset_path}\n\n` +
      `毛刺清洗输出是：\n${value("cleanOutput")}\n\n两者不同。确认仍要${action}？`,
    );
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

$("startAudit").onclick = () => start("/api/jobs/audit", {
  dataset_root: value("auditDataset"), report_path: value("auditReport"), workers: num("auditWorkers"),
  video_check: value("auditVideoCheck"),
});

function spike(clean) {
  return start(clean ? "/api/jobs/spikes/clean" : "/api/jobs/spikes/scan", {
    source_root: value("spikeSource"), output_root: value("cleanOutput"), thresholds_path: value("thresholdsPath"),
    report_path: value("spikeReport"), workers: num("spikeWorkers"),
  });
}

$("scanSpikes").onclick = () => spike(false);
$("cleanSpikes").onclick = () => spike(true);
$("startNormalize").onclick = async () => {
  await loadConfigProfile();
  if (!confirmDataBinding("计算归一化统计")) return;
  start("/api/jobs/normalize", {
    ...dataEnvironment(), openpi_root: value("openpiRoot"), config_name: value("normConfig"),
    workers: num("normWorkers"),
  });
};
$("startExport").onclick = () => start("/api/jobs/export/hdf5", {
  source_root: value("exportSource"), output_root: value("exportOutput"), workers: num("exportWorkers"),
  compression: value("exportCompression"), compression_level: num("compressionLevel"),
});
$("startTrain").onclick = async () => {
  await loadConfigProfile();
  if (!confirmDataBinding("启动训练")) return;
  start("/api/jobs/train", {
    ...dataEnvironment(), openpi_root: value("openpiRoot"), config_name: value("trainConfig"),
    exp_name: value("expName"), checkpoint_base_dir: value("checkpointRoot"), cuda_devices: value("cudaDevices"),
    memory_fraction: num("memoryFraction"), batch_size: num("batchSize"), num_train_steps: num("trainSteps"),
    fsdp_devices: num("fsdpDevices"), save_interval: num("saveInterval"), keep_period: num("keepPeriod"),
    mode: value("trainMode"),
  });
};
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

async function refreshSystem() {
  try {
    const snapshot = await api(`/api/system?path=${encodeURIComponent(value("checkpointRoot"))}`);
    $("connection").textContent = "服务正常";
    $("connection").className = "pill ok";
    $("gpuGrid").innerHTML = snapshot.gpus.length
      ? snapshot.gpus.map((gpu) => `<div class="gpu"><strong>GPU ${gpu.index} · ${gpu.name}</strong>${gpu.memory_used_mib}/${gpu.memory_total_mib} MiB · ${gpu.utilization_percent}% · ${gpu.temperature_c}°C</div>`).join("")
      : "未读取到 NVIDIA GPU";
    if (!snapshot.disk.error) {
      $("diskUsage").textContent = `${human(snapshot.disk.free)} 可用 / ${human(snapshot.disk.total)}`;
      $("diskMeter").value = 100 * snapshot.disk.used / snapshot.disk.total;
    }
  } catch (_error) {
    $("connection").textContent = "连接失败";
    $("connection").className = "pill bad";
  }
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
  const values = points.map((point) => [Number(point.step), Number(point.metrics?.[key])]).filter((point) => point.every(Number.isFinite));
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

async function show(job) {
  selectedJob = job.id;
  document.querySelectorAll(".job").forEach((node) => node.classList.toggle("selected", node.dataset.id === selectedJob));
  $("selectedKind").textContent = job.kind.toUpperCase();
  $("selectedTitle").textContent = job.id;
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
    chart("lossChart", metrics, "loss", "#2457e6");
    chart("normChart", metrics, "grad_norm", "#06a4ba");
  } catch (_error) { /* Job may finish between list and detail fetch. */ }
}

async function refreshJobs() {
  try {
    const jobs = await api("/api/jobs");
    $("jobList").innerHTML = jobs.length
      ? jobs.map((job) => `<div class="job" data-id="${job.id}"><strong>${escapeHtml(job.kind)}</strong><br><small>${job.id} · ${job.status} · ${Number(job.progress?.percent || 0).toFixed(1)}%</small></div>`).join("")
      : "暂无任务";
    document.querySelectorAll(".job").forEach((node) => {
      node.onclick = () => show(jobs.find((job) => job.id === node.dataset.id));
    });
    if (selectedJob) {
      const job = jobs.find((candidate) => candidate.id === selectedJob);
      if (job) await show(job);
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
  checkpointRecords = await api("/api/checkpoints");
  const select = $("serveCheckpoint");
  select.innerHTML = checkpointRecords.length
    ? checkpointRecords.map((item) => `<option value="${escapeHtml(item.path)}">${escapeHtml(item.config_name)} / ${escapeHtml(item.experiment_name)} / step ${item.step}</option>`).join("")
    : "<option value=''>没有找到完整 checkpoint</option>";
  const preferredRecord = checkpointRecords.find((item) => item.path === preferred);
  const sameConfig = checkpointRecords.find((item) => item.config_name === value("serveConfig"));
  if (preferredRecord) select.value = preferredRecord.path;
  else if (sameConfig) select.value = sameConfig.path;
  applyCheckpointConfig();
}

async function synchronizeConfig(event) {
  const configName = event.target.value.trim();
  ["normConfig", "trainConfig"].forEach((id) => {
    if ($(id) !== event.target) $(id).value = configName;
  });
  try {
    $("expName").value = (await api(`/api/next-experiment?config_name=${encodeURIComponent(configName)}`)).experiment_name;
  } catch (error) {
    toast(error.message, true);
  }
  await loadConfigProfile();
  save();
}

["normConfig", "trainConfig"].forEach((id) => $(id).addEventListener("change", synchronizeConfig));
$("serveCheckpoint").addEventListener("change", applyCheckpointConfig);
$("hfLerobotHome").addEventListener("change", loadConfigProfile);
$("cleanOutput").addEventListener("change", loadConfigProfile);
$("spikeSource").addEventListener("change", loadConfigProfile);
$("checkpointRoot").addEventListener("change", async () => {
  await Promise.all([loadCheckpoints(value("serveCheckpoint")), refreshSystem()]);
  if (currentConfigProfile) renderConfigProfile(currentConfigProfile);
});

async function init() {
  if (localStorage.getItem("bpx-defaults-version") !== defaultsVersion) {
    ["auditDataset", "auditReport", "spikeSource", "spikeReport", "cleanOutput", "exportSource", "exportOutput", "normConfig", "trainConfig", "serveConfig", "expName"]
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
      auditDataset: config.raw_dataset,
      auditReport: config.audit_report,
      spikeSource: config.raw_dataset,
      spikeReport: config.spike_report,
      cleanOutput: config.clean_dataset,
      exportSource: config.clean_dataset,
      exportOutput: config.hdf5_export,
      normConfig: config.default_config_name,
      trainConfig: config.default_config_name,
      serveConfig: config.default_config_name,
      expName: config.default_experiment_name,
    };
    Object.entries(defaults).forEach(([id, val]) => {
      if ($(id) && !$(id).value) $(id).value = val;
    });
    $("bpxConfigNames").innerHTML = config.known_config_names.map((name) => `<option value="${escapeHtml(name)}"></option>`).join("");
    await Promise.all([
      loadCheckpoints(localStorage.getItem("bpx-serveCheckpoint")),
      loadConfigProfile(),
      refreshSystem(),
      refreshJobs(),
    ]);
  } catch (error) {
    toast(error.message, true);
  }
}

init();
setInterval(refreshJobs, 2000);
setInterval(refreshSystem, 15000);
setInterval(loadConfigProfile, 30000);
