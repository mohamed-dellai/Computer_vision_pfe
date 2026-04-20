const state = {
  datasets: [],
  selectedDatasetId: null,
  versions: [],
  selectedVersionId: null,
  classes: [],
  images: [],
  selectedImageId: null,
  annotations: [],
  selectedBoxIndex: -1,
  imageObj: null,
  canvasScale: 1,
  galleryQuery: "",
  filteredImages: [],
  galleryRenderedCount: 0,
  galleryBatchSize: 120,
  isGalleryRendering: false,
  searchTimer: null,
  autosaveTimer: null,
  pendingAutosave: false,
  lastSavedSignature: null,
  autodistillAvailable: false,
  autodistillRunning: false,
  autodistillSelectedClassIds: [],
};

const t = window.TrainingShared;
const canvas = document.getElementById("annotator-canvas");
const ctx = canvas.getContext("2d");

let drawMode = null;
let dragStart = null;
let dragOffset = null;
let resizeState = null;

const HIT_PADDING = 6;
const HANDLE_SIZE = 8;

const datasetSelectEl = document.getElementById("dataset-select");
const versionSelectEl = document.getElementById("version-select");
const sourceVersionEl = document.getElementById("source-version");
const imageGalleryEl = document.getElementById("image-gallery");
const imageSearchEl = document.getElementById("image-search");
const imageCountEl = document.getElementById("image-count-label");
const imageNameEl = document.getElementById("annotator-image-name");
const imagePosEl = document.getElementById("annotator-image-position");
const prevBtnEl = document.getElementById("annotator-prev-image");
const nextBtnEl = document.getElementById("annotator-next-image");
const datasetInfoEl = document.getElementById("dataset-selected-info");
const renameVersionBtnEl = document.getElementById("rename-version-btn");
const deleteVersionBtnEl = document.getElementById("delete-version-btn");
const autodistillPromptEl = document.getElementById("autodistill-prompt");
const autodistillProviderEl = document.getElementById("autodistill-provider");
const autodistillScopeEl = document.getElementById("autodistill-scope");
const autodistillBoxThresholdEl = document.getElementById("autodistill-box-threshold");
const autodistillTextThresholdEl = document.getElementById("autodistill-text-threshold");
const autodistillReplaceEl = document.getElementById("autodistill-replace-existing");
const autodistillRunBtnEl = document.getElementById("autodistill-run-btn");
const autodistillStatusEl = document.getElementById("autodistill-status");
const autodistillClassListEl = document.getElementById("autodistill-class-list");
const autodistillSelectAllClassesEl = document.getElementById("autodistill-select-all-classes");
const autodistillClearClassesEl = document.getElementById("autodistill-clear-classes");
const autodistillUseSelectedPromptEl = document.getElementById("autodistill-use-selected-prompt");
const annotatorClassEl = document.getElementById("annotator-class");
const annotatorClassQuickpickEl = document.getElementById("annotator-class-quickpick");
const annotationCountEl = document.getElementById("annotation-count");
const autosaveStatusEl = document.getElementById("autosave-status");

if (!canvas.width) canvas.width = 960;
if (!canvas.height) canvas.height = 540;

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  initialize();
});

function bindEvents() {
  datasetSelectEl.addEventListener("change", onDatasetChange);
  versionSelectEl.addEventListener("change", onVersionChange);
  document.getElementById("create-version-btn").addEventListener("click", createVersion);
  renameVersionBtnEl.addEventListener("click", renameSelectedVersion);
  deleteVersionBtnEl.addEventListener("click", deleteSelectedVersion);

  prevBtnEl.addEventListener("click", () => navigateAnnotatorImage(-1));
  nextBtnEl.addEventListener("click", () => navigateAnnotatorImage(1));

  imageSearchEl.addEventListener("input", onImageSearchInput);
  imageGalleryEl.addEventListener("scroll", onGalleryScroll);
  imageGalleryEl.addEventListener("click", onGalleryClick);

  document.getElementById("delete-box-btn").addEventListener("click", deleteSelectedBox);
  autodistillRunBtnEl.addEventListener("click", onRunAutodistill);
  annotatorClassEl.addEventListener("change", onAnnotatorClassChange);
  autodistillSelectAllClassesEl.addEventListener("click", selectAllAutodistillClasses);
  autodistillClearClassesEl.addEventListener("click", clearAutodistillClasses);
  autodistillClassListEl.addEventListener("change", onAutodistillClassToggle);
  autodistillUseSelectedPromptEl.addEventListener("change", onAutodistillPromptModeChange);
  document.addEventListener("keydown", onDocumentKeyDown);

  canvas.addEventListener("mousedown", onCanvasDown);
  canvas.addEventListener("mousemove", onCanvasMove);
  canvas.addEventListener("mouseup", onCanvasUp);
  canvas.addEventListener("mouseleave", onCanvasLeave);
}

function onDocumentKeyDown(ev) {
  const tag = (ev.target?.tagName || "").toLowerCase();
  const isTyping = tag === "input" || tag === "textarea" || tag === "select";
  if (isTyping) return;

  if (ev.key === "Delete") {
    deleteSelectedBox();
    ev.preventDefault();
    return;
  }
  if (ev.key === "ArrowLeft") {
    navigateAnnotatorImage(-1);
    ev.preventDefault();
    return;
  }
  if (ev.key === "ArrowRight") {
    navigateAnnotatorImage(1);
    ev.preventDefault();
  }
}

function onAnnotatorClassChange() {
  renderClassQuickPick();
  if (state.selectedBoxIndex < 0) return;
  const selected = state.annotations[state.selectedBoxIndex];
  if (!selected) return;
  const cls = parseInt(annotatorClassEl.value, 10);
  if (Number.isNaN(cls) || selected.class_id === cls) return;
  state.annotations[state.selectedBoxIndex] = { ...selected, class_id: cls };
  drawCanvas();
  queueAutosave();
}

async function initialize() {
  try {
    await refreshAutodistillStatus();
    state.datasets = await t.fetchDatasets();
    state.selectedDatasetId = t.resolveDatasetId(state.datasets, t.getStoredDatasetId());
    t.setStoredDatasetId(state.selectedDatasetId);

    renderDatasetSelect();

    if (!state.selectedDatasetId) {
      resetContext();
      updateFlowInfo();
      drawCanvas();
      return;
    }

    await loadDatasetContext();
  } catch (e) {
    t.notify(e.message);
  }
}

function resetContext() {
  state.versions = [];
  state.selectedVersionId = null;
  state.classes = [];
  state.images = [];
  state.filteredImages = [];
  state.selectedImageId = null;
  state.annotations = [];
  state.selectedBoxIndex = -1;
  state.imageObj = null;
  state.pendingAutosave = false;
  state.lastSavedSignature = null;
  state.autodistillSelectedClassIds = [];
  renderVersionSelect();
  renderSourceVersionSelect();
  renderImageGallery();
  updateAnnotatorImageNav();
  renderClassSelect();
  renderAutodistillClassList();
  updateAnnotationMeta();
  setAutosaveStatus("Saved");
}

function renderDatasetSelect() {
  t.populateDatasetSelect(datasetSelectEl, state.datasets, state.selectedDatasetId, "No datasets available");
}

async function onDatasetChange() {
  await flushAutosave();
  state.selectedDatasetId = datasetSelectEl.value || null;
  t.setStoredDatasetId(state.selectedDatasetId);

  if (!state.selectedDatasetId) {
    resetContext();
    updateFlowInfo();
    drawCanvas();
    return;
  }

  await loadDatasetContext();
}

async function loadDatasetContext() {
  state.selectedVersionId = null;
  state.selectedImageId = null;
  state.annotations = [];
  state.selectedBoxIndex = -1;
  state.imageObj = null;

  try {
    await Promise.all([fetchClasses(), fetchVersions(), fetchImages()]);

    if (state.selectedVersionId && state.images.length > 0 && !state.selectedImageId) {
      state.selectedImageId = state.images[0].id;
    }

    updateFlowInfo();
    updateAnnotatorImageNav();
    renderImageGallery();
    await refreshAnnotatorData();
  } catch (e) {
    t.notify(e.message);
  }
}

function setAutodistillStatus(message) {
  autodistillStatusEl.textContent = message;
}

async function refreshAutodistillStatus() {
  try {
    const data = await t.api("/training/autodistill/status");
    state.autodistillAvailable = Boolean(data.available);

    if (!state.autodistillAvailable) {
      const message = data.error || "AutoDistill unavailable";
      setAutodistillStatus(message);
      setAutodistillRunEnabled(false);
      return;
    }

    const providers = Array.isArray(data.providers) ? data.providers : null;
    if (providers && autodistillProviderEl) {
      Array.from(autodistillProviderEl.options).forEach((opt) => {
        opt.disabled = !providers.includes(opt.value);
      });
    }
    if (data.default_provider && autodistillProviderEl) {
      const hasDefault = Array.from(autodistillProviderEl.options).some((o) => o.value === data.default_provider);
      if (hasDefault) autodistillProviderEl.value = data.default_provider;
    }

    const provider = data.provider ? `Provider: ${data.provider}` : "AutoDistill ready";
    setAutodistillStatus(provider);
    setAutodistillRunEnabled(true);
  } catch (e) {
    state.autodistillAvailable = false;
    setAutodistillStatus(e.message || "Failed to connect to autodistill service");
    setAutodistillRunEnabled(false);
  }
}

function setAutodistillStatus(msg) {
  if (autodistillStatusEl) autodistillStatusEl.textContent = msg;
}

function setAutodistillRunEnabled(enabled) {
  // We keep the button enabled so the user can click it and see the error message
  if (autodistillRunBtnEl) {
    if (state.autodistillRunning) {
      autodistillRunBtnEl.disabled = true;
      autodistillRunBtnEl.textContent = "Running...";
    } else {
      autodistillRunBtnEl.disabled = false;
      autodistillRunBtnEl.textContent = "Run AutoDistill";
    }
  }
}

function parseThresholdInput(el, name) {
  const v = parseFloat(el.value);
  if (Number.isNaN(v) || v < 0 || v > 1) {
    throw new Error(`${name} must be between 0 and 1`);
  }
  return v;
}

function collectAutodistillImageIds(scope) {
  if (scope === "all") {
    return [];
  }
  if (scope === "filtered") {
    return state.filteredImages.map((img) => img.id);
  }
  if (!state.selectedImageId) {
    return [];
  }
  return [state.selectedImageId];
}

async function onRunAutodistill(e) {
  if (e) e.preventDefault();

  if (state.autodistillRunning) return;
  if (!state.autodistillAvailable) {
    const msg = autodistillStatusEl ? autodistillStatusEl.textContent : "autodistill service is not available";
    t.notify(msg);
    return;
  }
  if (!state.selectedDatasetId) return t.notify("select a dataset first");
  if (!state.selectedVersionId) return t.notify("select an annotation version first");
  if (state.images.length === 0) return t.notify("this dataset has no images");

  const scope = autodistillScopeEl.value || "current";
  const imageIds = collectAutodistillImageIds(scope);
  if (scope !== "all" && imageIds.length === 0) {
    return t.notify("no images available for selected scope");
  }

  let boxThreshold;
  let textThreshold;
  try {
    boxThreshold = parseThresholdInput(autodistillBoxThresholdEl, "box threshold");
    textThreshold = parseThresholdInput(autodistillTextThresholdEl, "text threshold");
  } catch (e) {
    return t.notify(e.message);
  }

  const useSelectedPrompt = Boolean(autodistillUseSelectedPromptEl.checked);
  const selectedClasses = getSelectedAutodistillClasses();
  if (selectedClasses.length === 0) {
    return t.notify("select at least one class for AutoDistill");
  }
  const prompt = useSelectedPrompt
    ? selectedClasses.map((cls) => cls.name).join(",") // Fix: Remove space after comma for better backend parsing
    : autodistillPromptEl.value.trim();
  if (!useSelectedPrompt && !prompt) {
    return t.notify("prompt is required");
  }
  if (!useSelectedPrompt && selectedClasses.length !== 1) {
    return t.notify("for custom prompt, select exactly one target class");
  }
  const provider = (autodistillProviderEl.value || "dino").trim().toLowerCase();
  const replaceExisting = Boolean(autodistillReplaceEl.checked);
  const payload = {
    provider,
    prompt,
    selected_class_ids: selectedClasses.map((cls) => Number(cls.id)),
    target_class_id: useSelectedPrompt ? null : Number(selectedClasses[0].id),
    image_ids: scope === "all" ? null : imageIds,
    box_threshold: boxThreshold,
    text_threshold: textThreshold,
    replace_existing: replaceExisting,
  };

  await flushAutosave();

  state.autodistillRunning = true;
  setAutodistillRunEnabled(false);
  setAutodistillStatus("Running AutoDistill...");

  try {
    const data = await t.api(
      `/training/datasets/${state.selectedDatasetId}/annotation-versions/${state.selectedVersionId}/autodistill`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    );

    const errorCount = Array.isArray(data.errors) ? data.errors.length : 0;
    const usedProvider = data.provider ? ` using ${data.provider}` : "";
    const summary = `Done${usedProvider}. ${data.predicted_boxes} boxes predicted on ${data.labeled_images}/${data.processed_images} images${errorCount ? `, ${errorCount} errors` : ""}.`;
    setAutodistillStatus(summary);

    if (state.selectedImageId) {
      await loadAnnotations();
      drawCanvas();
    }

    t.notify(summary);
  } catch (e) {
    setAutodistillStatus(e.message);
    t.notify(e.message);
  } finally {
    state.autodistillRunning = false;
    setAutodistillRunEnabled(state.autodistillAvailable);
  }
}

function updateFlowInfo() {
  if (!state.selectedDatasetId) {
    datasetInfoEl.textContent = "Select a dataset and version to start annotating.";
    return;
  }

  const ds = state.datasets.find((x) => x.id === state.selectedDatasetId);
  const datasetName = ds ? ds.name : "Dataset";

  if (!state.selectedVersionId) {
    datasetInfoEl.textContent = `${datasetName} selected. Choose an annotation version to view images.`;
    return;
  }

  const version = state.versions.find((v) => v.id === state.selectedVersionId);
  datasetInfoEl.textContent = `${datasetName} | Version: ${version ? version.name : "Unknown"} | ${state.images.length} images`;
}

function setAutosaveStatus(text, level = "normal") {
  autosaveStatusEl.textContent = text;
  autosaveStatusEl.classList.remove("is-warning", "is-danger");
  if (level === "warning") autosaveStatusEl.classList.add("is-warning");
  if (level === "danger") autosaveStatusEl.classList.add("is-danger");
}

function updateAnnotationMeta() {
  const count = state.annotations.length;
  annotationCountEl.textContent = `${count} ${count === 1 ? "box" : "boxes"}`;
}

async function fetchVersions() {
  if (!state.selectedDatasetId) {
    state.versions = [];
    state.selectedVersionId = null;
    renderVersionSelect();
    renderSourceVersionSelect();
    return;
  }

  const data = await t.api(`/training/datasets/${state.selectedDatasetId}/annotation-versions`);
  state.versions = data.annotation_versions || [];

  if (state.selectedVersionId && !state.versions.find((v) => v.id === state.selectedVersionId)) {
    state.selectedVersionId = null;
  }
  if (!state.selectedVersionId && state.versions.length > 0) {
    state.selectedVersionId = state.versions[0].id;
  }

  renderVersionSelect();
  renderSourceVersionSelect();
}

function renderVersionSelect() {
  versionSelectEl.innerHTML = "";
  if (state.versions.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No versions available";
    versionSelectEl.appendChild(opt);
    versionSelectEl.disabled = true;
    renameVersionBtnEl.disabled = true;
    deleteVersionBtnEl.disabled = true;
    return;
  }

  state.versions.forEach((v) => {
    const opt = document.createElement("option");
    opt.value = v.id;
    opt.textContent = v.name;
    versionSelectEl.appendChild(opt);
  });

  versionSelectEl.disabled = false;
  versionSelectEl.value = state.selectedVersionId;
  renameVersionBtnEl.disabled = false;
  deleteVersionBtnEl.disabled = false;
}

function renderSourceVersionSelect() {
  sourceVersionEl.innerHTML = '<option value="">Empty version</option>';
  state.versions.forEach((v) => {
    const opt = document.createElement("option");
    opt.value = v.id;
    opt.textContent = v.name;
    sourceVersionEl.appendChild(opt);
  });
}

async function onVersionChange() {
  await flushAutosave();
  state.selectedVersionId = versionSelectEl.value || null;

  if (state.selectedVersionId && state.images.length > 0 && !state.selectedImageId) {
    state.selectedImageId = state.images[0].id;
  }

  updateFlowInfo();
  updateAnnotatorImageNav();
  renderImageGallery();
  await refreshAnnotatorData();
}

async function createVersion() {
  if (!state.selectedDatasetId) return t.notify("select a dataset first");

  const name = document.getElementById("version-name").value.trim();
  const sourceVersionId = sourceVersionEl.value || null;
  if (!name) return t.notify("version name is required");

  try {
    const created = await t.api(`/training/datasets/${state.selectedDatasetId}/annotation-versions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, source_version_id: sourceVersionId }),
    });

    document.getElementById("version-name").value = "";
    await fetchVersions();
    state.selectedVersionId = created.id;
    renderVersionSelect();

    if (state.images.length > 0 && !state.selectedImageId) {
      state.selectedImageId = state.images[0].id;
    }

    updateFlowInfo();
    renderImageGallery();
    await refreshAnnotatorData();
  } catch (e) {
    t.notify(e.message);
  }
}

async function renameSelectedVersion() {
  if (!state.selectedVersionId) return;
  const current = state.versions.find((v) => v.id === state.selectedVersionId);
  if (!current) return;

  const name = prompt("New version name:", current.name);
  if (name === null) return;

  try {
    await t.api(`/training/datasets/${state.selectedDatasetId}/annotation-versions/${current.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    await fetchVersions();
    updateFlowInfo();
  } catch (e) {
    t.notify(e.message);
  }
}

async function deleteSelectedVersion() {
  if (!state.selectedVersionId) return;
  if (!confirm("Delete this annotation version?")) return;

  const deletingId = state.selectedVersionId;

  try {
    await t.api(`/training/datasets/${state.selectedDatasetId}/annotation-versions/${deletingId}`, { method: "DELETE" });
    state.selectedVersionId = null;
    await fetchVersions();

    updateFlowInfo();
    renderImageGallery();
    await refreshAnnotatorData();
  } catch (e) {
    t.notify(e.message);
  }
}

async function fetchClasses() {
  if (!state.selectedDatasetId) {
    state.classes = [];
    renderClassSelect();
    renderAutodistillClassList();
    return;
  }

  const data = await t.api(`/training/datasets/${state.selectedDatasetId}/classes`);
  state.classes = data.classes || [];
  reconcileAutodistillClassSelection();
  renderClassSelect();
  renderAutodistillClassList();
}

function renderClassSelect() {
  const previous = annotatorClassEl.value;
  annotatorClassEl.innerHTML = "";
  if (state.classes.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No classes available";
    annotatorClassEl.appendChild(opt);
    annotatorClassEl.disabled = true;
    renderClassQuickPick();
    return;
  }

  annotatorClassEl.disabled = false;
  state.classes.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c.id;
    opt.textContent = `${c.id}: ${c.name}`;
    annotatorClassEl.appendChild(opt);
  });
  const hasPrevious = state.classes.some((c) => String(c.id) === previous);
  annotatorClassEl.value = hasPrevious ? previous : String(state.classes[0].id);
  renderClassQuickPick();
}

function renderClassQuickPick() {
  annotatorClassQuickpickEl.innerHTML = "";
  if (state.classes.length === 0) return;

  const selectedClassId = annotatorClassEl.value;
  state.classes.forEach((cls) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = `class-chip ${selectedClassId === String(cls.id) ? "active" : ""}`;
    chip.textContent = cls.name;
    chip.title = `Class ${cls.id}`;
    chip.addEventListener("click", () => {
      annotatorClassEl.value = String(cls.id);
      renderClassQuickPick();
    });
    annotatorClassQuickpickEl.appendChild(chip);
  });
}

function reconcileAutodistillClassSelection() {
  const classIds = state.classes.map((cls) => Number(cls.id));
  const valid = new Set(classIds);
  state.autodistillSelectedClassIds = state.autodistillSelectedClassIds.filter((id) => valid.has(id));
  if (state.autodistillSelectedClassIds.length === 0 && classIds.length > 0) {
    state.autodistillSelectedClassIds = [...classIds];
  }
}

function renderAutodistillClassList() {
  autodistillClassListEl.innerHTML = "";
  if (state.classes.length === 0) {
    const empty = document.createElement("div");
    empty.className = "muted";
    empty.textContent = "Create classes first to run AutoDistill.";
    autodistillClassListEl.appendChild(empty);
    syncAutodistillPromptFromClasses();
    return;
  }

  const selected = new Set(state.autodistillSelectedClassIds);
  state.classes.forEach((cls) => {
    const row = document.createElement("label");
    row.className = "autodistill-class-item";
    row.innerHTML = `
      <input type="checkbox" data-class-id="${cls.id}" ${selected.has(Number(cls.id)) ? "checked" : ""} />
      <span class="class-color-dot" style="background:${cls.color};"></span>
      <span>${cls.id}: ${cls.name}</span>
    `;
    autodistillClassListEl.appendChild(row);
  });
  syncAutodistillPromptFromClasses();
}

function getSelectedAutodistillClasses() {
  const selected = new Set(state.autodistillSelectedClassIds);
  return state.classes.filter((cls) => selected.has(Number(cls.id)));
}

function onAutodistillClassToggle(ev) {
  if (!ev.target.matches("input[type=\"checkbox\"][data-class-id]")) return;
  const classId = Number(ev.target.dataset.classId);
  const selected = new Set(state.autodistillSelectedClassIds);
  if (ev.target.checked) selected.add(classId);
  else selected.delete(classId);
  state.autodistillSelectedClassIds = [...selected];
  syncAutodistillPromptFromClasses();
}

function selectAllAutodistillClasses() {
  state.autodistillSelectedClassIds = state.classes.map((cls) => Number(cls.id));
  renderAutodistillClassList();
}

function clearAutodistillClasses() {
  state.autodistillSelectedClassIds = [];
  renderAutodistillClassList();
}

function onAutodistillPromptModeChange() {
  syncAutodistillPromptFromClasses();
}

function syncAutodistillPromptFromClasses() {
  const useSelectedPrompt = Boolean(autodistillUseSelectedPromptEl.checked);
  autodistillPromptEl.readOnly = useSelectedPrompt;
  if (useSelectedPrompt) {
    const classNames = getSelectedAutodistillClasses().map((cls) => cls.name);
    autodistillPromptEl.value = classNames.join(","); // Remove space for proper backend mapping
  }
}

async function fetchImages() {
  if (!state.selectedDatasetId) {
    state.images = [];
    state.filteredImages = [];
    state.selectedImageId = null;
    renderImageGallery();
    updateAnnotatorImageNav();
    return;
  }

  const data = await t.api(`/training/datasets/${state.selectedDatasetId}/images`);
  state.images = data.images || [];

  if (state.selectedImageId && !state.images.find((img) => img.id === state.selectedImageId)) {
    state.selectedImageId = null;
  }
  if (!state.selectedImageId && state.images.length > 0) {
    state.selectedImageId = state.images[0].id;
  }

  applyImageFilter();
  updateAnnotatorImageNav();
}

function onImageSearchInput() {
  if (state.searchTimer) clearTimeout(state.searchTimer);
  state.searchTimer = setTimeout(() => {
    state.galleryQuery = imageSearchEl.value.trim().toLowerCase();
    applyImageFilter();
  }, 120);
}

function applyImageFilter() {
  if (!state.galleryQuery) {
    state.filteredImages = [...state.images];
  } else {
    state.filteredImages = state.images.filter((img) =>
      img.original_name.toLowerCase().includes(state.galleryQuery)
    );
  }

  renderImageGallery();
}

function renderImageGallery() {
  imageGalleryEl.innerHTML = "";
  state.galleryRenderedCount = 0;

  if (!state.selectedDatasetId) {
    renderGalleryEmpty("Select a dataset first.");
    imageCountEl.textContent = "";
    return;
  }

  if (!state.selectedVersionId) {
    renderGalleryEmpty("Choose an annotation version to display images.");
    imageCountEl.textContent = `0 / ${state.images.length}`;
    return;
  }

  if (state.filteredImages.length === 0) {
    renderGalleryEmpty("No images match your search.");
    imageCountEl.textContent = `0 / ${state.images.length}`;
    return;
  }

  appendGalleryBatch();
}

function renderGalleryEmpty(message) {
  const empty = document.createElement("div");
  empty.className = "gallery-empty";
  empty.textContent = message;
  imageGalleryEl.appendChild(empty);
}

function appendGalleryBatch() {
  if (state.isGalleryRendering) return;
  if (state.galleryRenderedCount >= state.filteredImages.length) return;

  state.isGalleryRendering = true;

  const start = state.galleryRenderedCount;
  const end = Math.min(start + state.galleryBatchSize, state.filteredImages.length);
  const fragment = document.createDocumentFragment();

  for (let i = start; i < end; i += 1) {
    const img = state.filteredImages[i];
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `image-card ${state.selectedImageId === img.id ? "active" : ""}`;
    btn.dataset.imageId = img.id;
    btn.innerHTML = `
      <img class="image-thumb" src="${img.url}" alt="${img.original_name}" loading="lazy" decoding="async" />
      <span class="image-card-name" title="${img.original_name}">${img.original_name}</span>
    `;
    fragment.appendChild(btn);
  }

  imageGalleryEl.appendChild(fragment);
  state.galleryRenderedCount = end;
  imageCountEl.textContent = `${state.galleryRenderedCount} / ${state.filteredImages.length} shown (${state.images.length} total)`;

  state.isGalleryRendering = false;

  while (
    state.galleryRenderedCount < state.filteredImages.length &&
    imageGalleryEl.scrollHeight <= imageGalleryEl.clientHeight
  ) {
    appendGalleryBatch();
  }
}

function onGalleryScroll() {
  if (!state.selectedVersionId) return;
  if (imageGalleryEl.scrollTop + imageGalleryEl.clientHeight >= imageGalleryEl.scrollHeight - 300) {
    appendGalleryBatch();
  }
}

function onGalleryClick(ev) {
  const card = ev.target.closest(".image-card");
  if (!card) return;

  const imageId = card.dataset.imageId;
  if (!imageId || imageId === state.selectedImageId) return;

  selectImage(imageId);
}

async function selectImage(imageId) {
  await flushAutosave();
  state.selectedImageId = imageId;
  updateGallerySelection();
  updateAnnotatorImageNav();
  await refreshAnnotatorData();
}

function updateGallerySelection() {
  const prev = imageGalleryEl.querySelector(".image-card.active");
  if (prev) prev.classList.remove("active");

  const current = imageGalleryEl.querySelector(`[data-image-id="${state.selectedImageId}"]`);
  if (current) current.classList.add("active");
}

function navigateAnnotatorImage(delta) {
  if (!state.selectedVersionId || !state.images.length) return;

  const idx = state.images.findIndex((x) => x.id === state.selectedImageId);
  const currentIdx = idx >= 0 ? idx : 0;
  const nextIdx = currentIdx + delta;

  if (nextIdx < 0 || nextIdx >= state.images.length) return;

  selectImage(state.images[nextIdx].id);
}

function updateAnnotatorImageNav() {
  if (!state.selectedVersionId) {
    imageNameEl.textContent = "Choose annotation version";
    imagePosEl.textContent = "0 / 0";
    prevBtnEl.disabled = true;
    nextBtnEl.disabled = true;
    return;
  }

  const total = state.images.length;
  const idx = state.images.findIndex((x) => x.id === state.selectedImageId);
  const current = idx >= 0 ? idx + 1 : 0;

  const image = idx >= 0 ? state.images[idx] : null;
  imageNameEl.textContent = image ? image.original_name : "No image selected";

  imagePosEl.textContent = `${current} / ${total}`;
  prevBtnEl.disabled = total === 0 || current <= 1;
  nextBtnEl.disabled = total === 0 || current >= total;
}

async function refreshAnnotatorData() {
  if (!state.selectedDatasetId || !state.selectedVersionId || !state.selectedImageId) {
    state.annotations = [];
    state.selectedBoxIndex = -1;
    state.imageObj = null;
    updateAnnotationMeta();
    setAutosaveStatus("Saved");
    drawCanvas();
    return;
  }

  await loadImage();
  await loadAnnotations();
  drawCanvas();
}

async function loadImage() {
  const image = state.images.find((x) => x.id === state.selectedImageId);
  if (!image) {
    state.imageObj = null;
    return;
  }

  const img = new Image();
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = reject;
    img.src = image.url;
  });

  state.imageObj = img;
  const maxWidth = 900;
  state.canvasScale = img.width > maxWidth ? maxWidth / img.width : 1;
  canvas.width = Math.round(img.width * state.canvasScale);
  canvas.height = Math.round(img.height * state.canvasScale);
}

async function loadAnnotations() {
  const data = await t.api(
    `/training/datasets/${state.selectedDatasetId}/annotation-versions/${state.selectedVersionId}/annotations/${state.selectedImageId}`
  );
  state.annotations = data.annotations || [];
  state.selectedBoxIndex = -1;
  state.pendingAutosave = false;
  state.lastSavedSignature = getAnnotationsSignature();
  updateAnnotationMeta();
  setAutosaveStatus("Saved");
}

function clamp01(v) {
  return Math.min(1, Math.max(0, v));
}

function toCanvasBox(a) {
  const iw = state.imageObj.width;
  const ih = state.imageObj.height;
  return {
    x: (a.x - a.w / 2) * iw * state.canvasScale,
    y: (a.y - a.h / 2) * ih * state.canvasScale,
    w: a.w * iw * state.canvasScale,
    h: a.h * ih * state.canvasScale,
  };
}

function toYoloBox(box) {
  const iw = state.imageObj.width;
  const ih = state.imageObj.height;
  const classValue = annotatorClassEl.value;
  const classId = classValue === "" ? null : parseInt(classValue, 10);
  return {
    class_id: classId,
    x: clamp01((box.x + box.w / 2) / (iw * state.canvasScale)),
    y: clamp01((box.y + box.h / 2) / (ih * state.canvasScale)),
    w: clamp01(box.w / (iw * state.canvasScale)),
    h: clamp01(box.h / (ih * state.canvasScale)),
  };
}

function getCanvasPoint(ev) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = rect.width > 0 ? canvas.width / rect.width : 1;
  const scaleY = rect.height > 0 ? canvas.height / rect.height : 1;
  return {
    x: (ev.clientX - rect.left) * scaleX,
    y: (ev.clientY - rect.top) * scaleY,
  };
}

function normalizeBox(box) {
  const x = box.w >= 0 ? box.x : box.x + box.w;
  const y = box.h >= 0 ? box.y : box.y + box.h;
  return { x, y, w: Math.abs(box.w), h: Math.abs(box.h) };
}

function clampBox(box) {
  const b = normalizeBox(box);
  b.x = Math.max(0, Math.min(b.x, canvas.width - 1));
  b.y = Math.max(0, Math.min(b.y, canvas.height - 1));
  b.w = Math.max(1, Math.min(b.w, canvas.width - b.x));
  b.h = Math.max(1, Math.min(b.h, canvas.height - b.y));
  return b;
}

function getHandleRects(box) {
  const hs = HANDLE_SIZE / 2;
  const points = {
    nw: { x: box.x, y: box.y },
    ne: { x: box.x + box.w, y: box.y },
    sw: { x: box.x, y: box.y + box.h },
    se: { x: box.x + box.w, y: box.y + box.h },
  };
  return Object.fromEntries(
    Object.entries(points).map(([key, p]) => [
      key,
      { x: p.x - hs, y: p.y - hs, w: HANDLE_SIZE, h: HANDLE_SIZE },
    ])
  );
}

function findResizeHandle(x, y, box) {
  const handles = getHandleRects(box);
  for (const [name, h] of Object.entries(handles)) {
    if (x >= h.x && x <= h.x + h.w && y >= h.y && y <= h.y + h.h) {
      return name;
    }
  }
  return null;
}

function drawCanvas(tempBox = null) {
  ctx.clearRect(0, 0, canvas.width || 0, canvas.height || 0);
  if (!state.imageObj) {
    ctx.fillStyle = "#1f2937";
    ctx.fillRect(0, 0, canvas.width || 960, canvas.height || 540);
    return;
  }

  ctx.drawImage(state.imageObj, 0, 0, canvas.width, canvas.height);

  state.annotations.forEach((a, idx) => {
    const b = toCanvasBox(a);
    const cls = state.classes.find((c) => c.id === a.class_id);
    const color = cls ? cls.color : "#00ff00";

    ctx.strokeStyle = idx === state.selectedBoxIndex ? "#ffffff" : color;
    ctx.lineWidth = idx === state.selectedBoxIndex ? 3 : 2;
    ctx.strokeRect(b.x, b.y, b.w, b.h);

    ctx.fillStyle = color;
    ctx.font = "12px Inter";
    ctx.fillText(cls ? cls.name : String(a.class_id), b.x + 2, Math.max(12, b.y - 4));

    if (idx === state.selectedBoxIndex) drawResizeHandles(b);
  });

  if (tempBox) {
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2;
    ctx.strokeRect(tempBox.x, tempBox.y, tempBox.w, tempBox.h);
  }
}

function drawResizeHandles(box) {
  const handles = getHandleRects(box);
  ctx.fillStyle = "#ffffff";
  ctx.strokeStyle = "#111827";
  ctx.lineWidth = 1;
  Object.values(handles).forEach((h) => {
    ctx.fillRect(h.x, h.y, h.w, h.h);
    ctx.strokeRect(h.x, h.y, h.w, h.h);
  });
}

function findHitBox(x, y) {
  for (let i = state.annotations.length - 1; i >= 0; i -= 1) {
    const b = toCanvasBox(state.annotations[i]);
    if (
      x >= b.x - HIT_PADDING &&
      x <= b.x + b.w + HIT_PADDING &&
      y >= b.y - HIT_PADDING &&
      y <= b.y + b.h + HIT_PADDING
    ) {
      return i;
    }
  }
  return -1;
}

function onCanvasDown(ev) {
  if (!state.imageObj || !state.selectedVersionId) return;
  const { x, y } = getCanvasPoint(ev);

  if (state.selectedBoxIndex >= 0) {
    const selectedBox = toCanvasBox(state.annotations[state.selectedBoxIndex]);
    const handle = findResizeHandle(x, y, selectedBox);
    if (handle) {
      drawMode = "resize";
      resizeState = { handle, box: selectedBox };
      return;
    }
  }

  const hit = findHitBox(x, y);

  if (hit >= 0) {
    state.selectedBoxIndex = hit;
    const b = toCanvasBox(state.annotations[hit]);
    const clsId = state.annotations[hit].class_id;
    if (state.classes.some((cls) => Number(cls.id) === Number(clsId))) {
      annotatorClassEl.value = String(clsId);
      renderClassQuickPick();
    }

    const handle = findResizeHandle(x, y, b);
    if (handle) {
      drawMode = "resize";
      resizeState = { handle, box: b };
      drawCanvas();
      return;
    }

    drawMode = "move";
    dragOffset = { x: x - b.x, y: y - b.y, w: b.w, h: b.h };
  } else {
    state.selectedBoxIndex = -1;
    drawMode = "draw";
    dragStart = { x, y };
  }

  drawCanvas();
}

function onCanvasMove(ev) {
  const { x, y } = getCanvasPoint(ev);

  if (!drawMode) {
    updateCursor(x, y);
    return;
  }

  if (drawMode === "draw") {
    drawCanvas(normalizeBox({ x: dragStart.x, y: dragStart.y, w: x - dragStart.x, h: y - dragStart.y }));
    return;
  }

  if (drawMode === "move" && state.selectedBoxIndex >= 0) {
    const box = clampBox({ x: x - dragOffset.x, y: y - dragOffset.y, w: dragOffset.w, h: dragOffset.h });
    box.x = Math.max(0, Math.min(box.x, canvas.width - box.w));
    box.y = Math.max(0, Math.min(box.y, canvas.height - box.h));

    const class_id = state.annotations[state.selectedBoxIndex].class_id;
    state.annotations[state.selectedBoxIndex] = { ...toYoloBox(box), class_id };
    drawCanvas();
    return;
  }

  if (drawMode === "resize" && state.selectedBoxIndex >= 0 && resizeState) {
    const start = resizeState.box;
    let box;

    if (resizeState.handle === "nw") {
      box = normalizeBox({ x, y, w: start.x + start.w - x, h: start.y + start.h - y });
    } else if (resizeState.handle === "ne") {
      box = normalizeBox({ x: start.x, y, w: x - start.x, h: start.y + start.h - y });
    } else if (resizeState.handle === "sw") {
      box = normalizeBox({ x, y: start.y, w: start.x + start.w - x, h: y - start.y });
    } else {
      box = normalizeBox({ x: start.x, y: start.y, w: x - start.x, h: y - start.y });
    }

    box = clampBox(box);
    const class_id = state.annotations[state.selectedBoxIndex].class_id;
    state.annotations[state.selectedBoxIndex] = { ...toYoloBox(box), class_id };
    drawCanvas();
  }
}

function onCanvasUp(ev) {
  if (!drawMode) return;
  const { x, y } = getCanvasPoint(ev);

  if (drawMode === "draw") {
    const box = normalizeBox({ x: dragStart.x, y: dragStart.y, w: x - dragStart.x, h: y - dragStart.y });
    if (box.w > 4 && box.h > 4) {
      const yolo = toYoloBox(box);
      if (yolo.class_id === null || Number.isNaN(yolo.class_id)) {
        t.notify("create classes first, then select a class");
      } else {
        state.annotations.push(yolo);
        state.selectedBoxIndex = state.annotations.length - 1;
      }
    }
  }

  if (drawMode === "move" || drawMode === "resize" || drawMode === "draw") {
    queueAutosave();
  }

  drawMode = null;
  dragStart = null;
  dragOffset = null;
  resizeState = null;
  canvas.style.cursor = "crosshair";
  updateAnnotationMeta();
  drawCanvas();
}

function onCanvasLeave() {
  if (!drawMode) {
    canvas.style.cursor = "crosshair";
    return;
  }

  drawMode = null;
  dragStart = null;
  dragOffset = null;
  resizeState = null;
  canvas.style.cursor = "crosshair";
  drawCanvas();
}

function updateCursor(x, y) {
  if (state.selectedBoxIndex >= 0) {
    const b = toCanvasBox(state.annotations[state.selectedBoxIndex]);
    const handle = findResizeHandle(x, y, b);

    if (handle === "nw" || handle === "se") {
      canvas.style.cursor = "nwse-resize";
      return;
    }
    if (handle === "ne" || handle === "sw") {
      canvas.style.cursor = "nesw-resize";
      return;
    }
  }

  const hit = findHitBox(x, y);
  canvas.style.cursor = hit >= 0 ? "move" : "crosshair";
}

function deleteSelectedBox() {
  if (state.selectedBoxIndex < 0) return;
  state.annotations.splice(state.selectedBoxIndex, 1);
  state.selectedBoxIndex = -1;
  updateAnnotationMeta();
  drawCanvas();
  queueAutosave();
}

function getAnnotationsSignature() {
  return JSON.stringify(state.annotations);
}

function queueAutosave() {
  if (!state.selectedDatasetId || !state.selectedVersionId || !state.selectedImageId) return;
  state.pendingAutosave = true;
  setAutosaveStatus("Unsaved changes", "warning");
  if (state.autosaveTimer) {
    clearTimeout(state.autosaveTimer);
  }
  state.autosaveTimer = setTimeout(() => {
    saveAnnotations(true);
  }, 450);
}

async function flushAutosave() {
  if (state.autosaveTimer) {
    clearTimeout(state.autosaveTimer);
    state.autosaveTimer = null;
  }
  if (state.pendingAutosave) {
    await saveAnnotations(true);
  }
}

async function saveAnnotations(silent = false) {
  if (!state.selectedDatasetId || !state.selectedVersionId || !state.selectedImageId) {
    return;
  }

  const signature = getAnnotationsSignature();
  if (signature === state.lastSavedSignature && !state.pendingAutosave) {
    return;
  }

  try {
    setAutosaveStatus("Saving...", "warning");
    await t.api(
      `/training/datasets/${state.selectedDatasetId}/annotation-versions/${state.selectedVersionId}/annotations/${state.selectedImageId}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(state.annotations),
      }
    );
    state.pendingAutosave = false;
    state.lastSavedSignature = signature;
    setAutosaveStatus("Saved");
    if (!silent) t.notify("annotations saved");
  } catch (e) {
    state.pendingAutosave = true;
    setAutosaveStatus("Save failed", "danger");
    t.notify(e.message);
  }
}
