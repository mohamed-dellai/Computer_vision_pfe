const state = {
  datasets: [],
  selectedDatasetId: null,
  versions: [],
  activeJobId: null,
  currentLogJobId: null,
  logOffset: 0,
  jobPollTimer: null,
  trainingApiUnavailable: false,
};

const t = window.TrainingShared;
const AUGMENTATION_STORAGE_KEY = "training.augmentation.v1";
const AUGMENTATION_DEFAULTS = {
  hsv_h: 0.015,
  hsv_s: 0.7,
  hsv_v: 0.4,
  degrees: 0,
  translate: 0.1,
  scale: 0.5,
  shear: 0,
  perspective: 0,
  flipud: 0,
  fliplr: 0.5,
  mosaic: 1,
  mixup: 0,
  copy_paste: 0,
  erasing: 0.4,
  close_mosaic: 10,
};
const AUGMENTATION_INPUTS = {
  hsv_h: "aug-hsv-h",
  hsv_s: "aug-hsv-s",
  hsv_v: "aug-hsv-v",
  degrees: "aug-degrees",
  translate: "aug-translate",
  scale: "aug-scale",
  shear: "aug-shear",
  perspective: "aug-perspective",
  flipud: "aug-flipud",
  fliplr: "aug-fliplr",
  mosaic: "aug-mosaic",
  mixup: "aug-mixup",
  copy_paste: "aug-copy-paste",
  erasing: "aug-erasing",
  close_mosaic: "aug-close-mosaic",
};

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  initialize();
});

function bindEvents() {
  document.getElementById("train-dataset").addEventListener("change", onDatasetChange);
  document.getElementById("start-training-btn").addEventListener("click", startTraining);
  document.getElementById("stop-training-btn").addEventListener("click", stopTraining);
  document.getElementById("aug-enabled").addEventListener("change", onAugmentationChange);
  document.getElementById("aug-defaults-btn").addEventListener("click", () => {
    setAugmentationValues(AUGMENTATION_DEFAULTS);
    document.getElementById("aug-enabled").checked = true;
    onAugmentationChange();
  });
  document.getElementById("aug-zero-btn").addEventListener("click", () => {
    const values = Object.fromEntries(Object.keys(AUGMENTATION_DEFAULTS).map((key) => [key, 0]));
    setAugmentationValues(values);
    document.getElementById("aug-enabled").checked = true;
    onAugmentationChange();
  });
  Object.values(AUGMENTATION_INPUTS).forEach((id) => {
    document.getElementById(id).addEventListener("input", saveAugmentationSettings);
  });
}

async function initialize() {
  try {
    loadAugmentationSettings();
    state.datasets = await t.fetchDatasets();
    state.selectedDatasetId = t.resolveDatasetId(state.datasets, t.getStoredDatasetId());
    t.setStoredDatasetId(state.selectedDatasetId);
    renderDatasetSelect();
    await fetchVersions();
    await fetchModelSources();
    await pollJobs();
    state.jobPollTimer = setInterval(pollJobs, 5000);
  } catch (e) {
    t.notify(e.message);
  }
}

function loadAugmentationSettings() {
  try {
    const raw = localStorage.getItem(AUGMENTATION_STORAGE_KEY);
    if (!raw) {
      setAugmentationValues(AUGMENTATION_DEFAULTS);
      onAugmentationChange();
      return;
    }
    const saved = JSON.parse(raw);
    document.getElementById("aug-enabled").checked = !!saved.enabled;
    setAugmentationValues({ ...AUGMENTATION_DEFAULTS, ...(saved.values || {}) });
    onAugmentationChange();
  } catch (e) {
    setAugmentationValues(AUGMENTATION_DEFAULTS);
    onAugmentationChange();
  }
}

function saveAugmentationSettings() {
  const settings = {
    enabled: document.getElementById("aug-enabled").checked,
    values: readAugmentationValues(),
  };
  localStorage.setItem(AUGMENTATION_STORAGE_KEY, JSON.stringify(settings));
}

function onAugmentationChange() {
  const enabled = document.getElementById("aug-enabled").checked;
  document.getElementById("augmentation-controls").classList.toggle("disabled", !enabled);
  Object.values(AUGMENTATION_INPUTS).forEach((id) => {
    document.getElementById(id).disabled = !enabled;
  });
  saveAugmentationSettings();
}

function setAugmentationValues(values) {
  Object.entries(AUGMENTATION_INPUTS).forEach(([key, id]) => {
    document.getElementById(id).value = values[key];
  });
}

function readAugmentationValues() {
  const values = {};
  Object.entries(AUGMENTATION_INPUTS).forEach(([key, id]) => {
    const input = document.getElementById(id);
    const fallback = AUGMENTATION_DEFAULTS[key];
    const parsed = key === "close_mosaic" ? parseInt(input.value, 10) : parseFloat(input.value);
    values[key] = Number.isFinite(parsed) ? parsed : fallback;
  });
  return values;
}

function buildAugmentationPayload() {
  const enabled = document.getElementById("aug-enabled").checked;
  if (!enabled) return null;
  return {
    enabled: true,
    ...readAugmentationValues(),
  };
}

function renderDatasetSelect() {
  const select = document.getElementById("train-dataset");
  t.populateDatasetSelect(select, state.datasets, state.selectedDatasetId, "No datasets available");
}

async function onDatasetChange() {
  state.selectedDatasetId = document.getElementById("train-dataset").value || null;
  t.setStoredDatasetId(state.selectedDatasetId);
  await fetchVersions();
}

async function fetchVersions() {
  const select = document.getElementById("train-version");
  select.innerHTML = "";
  state.versions = [];
  if (!state.selectedDatasetId) return;

  try {
    const data = await t.api(`/training/datasets/${state.selectedDatasetId}/annotation-versions`);
    state.versions = data.annotation_versions || [];
    state.versions.forEach((v) => {
      const opt = document.createElement("option");
      opt.value = v.id;
      opt.textContent = v.name;
      select.appendChild(opt);
    });
  } catch (e) {
    t.notify(e.message);
  }
}

async function fetchModelSources() {
  const preset = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"];
  const select = document.getElementById("train-model");
  select.innerHTML = "";
  try {
    const data = await t.api("/models");
    const uploaded = (data.models || []).filter((m) => m.endsWith(".pt"));
    [...preset, ...uploaded].forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      select.appendChild(opt);
    });
  } catch (e) {
    console.error(e);
  }
}

async function startTraining() {
  if (!state.selectedDatasetId) return t.notify("select a dataset first");
  const annotation_version_id = document.getElementById("train-version").value;
  if (!annotation_version_id) return t.notify("select annotation version");

  const payload = {
    dataset_id: state.selectedDatasetId,
    annotation_version_id,
    model_source: document.getElementById("train-model").value,
    epochs: parseInt(document.getElementById("train-epochs").value, 10) || 50,
    batch_size: parseInt(document.getElementById("train-batch").value, 10) || 16,
    imgsz: parseInt(document.getElementById("train-imgsz").value, 10) || 640,
    split_train: parseFloat(document.getElementById("split-train").value) || 0.8,
    split_val: parseFloat(document.getElementById("split-val").value) || 0.1,
    split_test: parseFloat(document.getElementById("split-test").value) || 0.1,
    seed: 42,
  };
  const augmentation = buildAugmentationPayload();
  if (augmentation) {
    payload.augmentation = augmentation;
  }
  saveAugmentationSettings();

  try {
    const job = await t.api("/training/jobs/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.activeJobId = job.id;
    state.currentLogJobId = job.id;
    state.logOffset = 0;
    document.getElementById("training-logs").textContent = "";
    document.getElementById("training-status-detail").textContent = "";
    t.notify("training started");
  } catch (e) {
    t.notify(e.message);
  }
}

async function stopTraining() {
  if (!state.activeJobId) return t.notify("no active training job");
  try {
    await t.api(`/training/jobs/${state.activeJobId}/stop`, { method: "POST" });
    t.notify("stop requested");
  } catch (e) {
    t.notify(e.message);
  }
}

async function pollJobs() {
  if (state.trainingApiUnavailable) return;

  try {
    const data = await t.api("/training/jobs");
    state.activeJobId = data.active_job_id || null;
    let active = null;
    if (state.activeJobId) {
      active = data.jobs.find((j) => j.id === state.activeJobId) || null;
    } else if ((data.jobs || []).length > 0) {
      active = data.jobs[0];
    }

    renderJobStatus(active);
    if (active) {
      if (state.currentLogJobId !== active.id) {
        state.currentLogJobId = active.id;
        state.logOffset = 0;
        document.getElementById("training-logs").textContent = "";
      }
      await pollLogs(active.id);
      if (active.status !== "running" && active.status !== "starting") {
        state.activeJobId = null;
      }
    }
  } catch (e) {
    const msg = e && e.message ? e.message : "";
    if (msg.includes("404")) {
      state.trainingApiUnavailable = true;
      if (state.jobPollTimer) {
        clearInterval(state.jobPollTimer);
        state.jobPollTimer = null;
      }
      document.getElementById("training-status").textContent = "Training API unavailable (404). Restart backend to enable training features.";
      document.getElementById("training-status-detail").textContent = "";
      return;
    }
    console.error(e);
  }
}

function renderJobStatus(job) {
  const el = document.getElementById("training-status");
  const detailEl = document.getElementById("training-status-detail");
  if (!job) {
    el.textContent = "No training jobs yet.";
    detailEl.textContent = "";
    return;
  }

  el.textContent = `Job ${job.id.slice(0, 8)} | ${job.status} | Epoch ${job.epoch_current}/${job.epoch_total} | ${(
    job.progress_pct || 0
  ).toFixed(1)}%`;

  if (job.message) {
    detailEl.textContent = `Message: ${job.message}`;
  } else {
    detailEl.textContent = "";
  }
}

async function pollLogs(jobId) {
  const data = await t.api(`/training/jobs/${jobId}/logs?offset=${state.logOffset}`);
  const logsEl = document.getElementById("training-logs");
  if (data.chunk) {
    logsEl.textContent += data.chunk;
    logsEl.scrollTop = logsEl.scrollHeight;
  }
  state.logOffset = data.next_offset || state.logOffset;
}
