const state = {
  datasets: [],
  selectedDatasetId: null,
  images: [],
  imageNameFilter: "",
  activeTransformation: "",
  cropBox: {
    left: 0.05,
    top: 0.05,
    right: 0.95,
    bottom: 0.95,
  },
  cropDragHandle: null,
  renderedImageCount: 0,
  imageRenderBatchSize: 100,
};

const t = window.TrainingShared;

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  initVideoImport();
  initialize();
});

function bindEvents() {
  document.getElementById("create-dataset-btn").addEventListener("click", createDataset);
  document.getElementById("upload-images-btn").addEventListener("click", uploadImages);
  document.getElementById("image-name-filter").addEventListener("input", (event) => {
    state.imageNameFilter = event.target.value.trim().toLowerCase();
    state.renderedImageCount = 0;
    renderImages();
  });
  const transformationSelect = document.getElementById("transformation-select");
  if (transformationSelect) {
    state.activeTransformation = transformationSelect.value;
    transformationSelect.addEventListener("change", (event) => {
      state.activeTransformation = event.target.value;
      if (state.activeTransformation === "crop") {
        openCropEditor();
      }
    });
  }
  const applyCropBtn = document.getElementById("apply-crop-btn");
  if (applyCropBtn) {
    applyCropBtn.addEventListener("click", applyCropToFilteredImages);
  }
  const cropEditorClose = document.getElementById("crop-editor-close");
  // Use shared crop editor handlers where available
  if (window.SharedCropEditor) {
    // render on load
    const cropImage = document.getElementById("crop-editor-img");
    if (cropImage) cropImage.addEventListener("load", () => { window.SharedCropEditor.render(); renderCropEditor(); });
    // update UI when shared crop changes
    document.addEventListener('sharedcrop:changed', () => renderCropEditor());
    document.addEventListener('sharedcrop:opened', () => renderCropEditor());
  }
  window.addEventListener("resize", updateCropFrame);
  document.getElementById("image-viewer-close").addEventListener("click", closeImageViewer);
  document.getElementById("image-viewer").addEventListener("click", (event) => {
    if (event.target.id === "image-viewer") closeImageViewer();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeImageViewer();
      closeCropEditor();
    }
  });
}

function initVideoImport() {
  if (!window.TrainingUploadVideo || typeof window.TrainingUploadVideo.init !== "function") {
    return;
  }

  window.TrainingUploadVideo.init({
    getSelectedDatasetId: () => state.selectedDatasetId,
    api: t.api,
    notify: t.notify,
    onSuccess: async () => {
      state.datasets = await t.fetchDatasets();
      renderDatasets();
      await fetchImages();
    },
  });
}

async function initialize() {
  try {
    state.datasets = await t.fetchDatasets();
    state.selectedDatasetId = t.resolveDatasetId(state.datasets, t.getStoredDatasetId());
    t.setStoredDatasetId(state.selectedDatasetId);
    renderDatasets();
    await fetchImages();
  } catch (e) {
    t.notify(e.message);
  }
}

function renderDatasets() {
  const el = document.getElementById("datasets-list");
  el.innerHTML = "";
  state.datasets.forEach((d) => {
    const row = document.createElement("div");
    row.className = `list-item ${state.selectedDatasetId === d.id ? "active" : ""}`;
    row.innerHTML = `
      <div>
        <strong>${d.name}</strong><br/>
        <small>${t.datasetLabel(d)}</small>
      </div>
      <div class="item-actions">
        <button class="btn btn-secondary">Open</button>
        <button class="btn btn-secondary">Delete</button>
      </div>
    `;
    const [openBtn, deleteBtn] = row.querySelectorAll("button");
    openBtn.onclick = async () => {
      state.selectedDatasetId = d.id;
      t.setStoredDatasetId(state.selectedDatasetId);
      renderDatasets();
      await fetchImages();
    };
    deleteBtn.onclick = () => deleteDataset(d.id);
    el.appendChild(row);
  });
}

async function createDataset() {
  const name = document.getElementById("dataset-name").value.trim();
  const description = document.getElementById("dataset-description").value.trim();
  if (!name) return t.notify("dataset name is required");

  try {
    const d = await t.api("/training/datasets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, description }),
    });
    document.getElementById("dataset-name").value = "";
    document.getElementById("dataset-description").value = "";
    state.datasets = await t.fetchDatasets();
    state.selectedDatasetId = d.id;
    t.setStoredDatasetId(state.selectedDatasetId);
    renderDatasets();
    await fetchImages();
  } catch (e) {
    t.notify(e.message);
  }
}

async function deleteDataset(datasetId) {
  if (!confirm("Delete this dataset and all related files?")) return;
  try {
    await t.api(`/training/datasets/${datasetId}`, { method: "DELETE" });
    state.datasets = await t.fetchDatasets();
    state.selectedDatasetId = t.resolveDatasetId(state.datasets, state.selectedDatasetId === datasetId ? null : state.selectedDatasetId);
    t.setStoredDatasetId(state.selectedDatasetId);
    renderDatasets();
    await fetchImages();
  } catch (e) {
    t.notify(e.message);
  }
}

async function fetchImages() {
  const info = document.getElementById("dataset-selected-info");
  const list = document.getElementById("images-list");
  list.innerHTML = "";

  if (!state.selectedDatasetId) {
    info.textContent = "Select a dataset to manage images.";
    state.images = [];
    renderImages();
    return;
  }

  const ds = state.datasets.find((d) => d.id === state.selectedDatasetId);
  info.textContent = `${ds ? ds.name : "Dataset"} - upload and delete images here.`;

  try {
    const data = await t.api(`/training/datasets/${state.selectedDatasetId}/images`);
    state.images = data.images || [];
    state.renderedImageCount = 0;
    renderImages();
  } catch (e) {
    t.notify(e.message);
  }
}

function renderImages() {
  const list = document.getElementById("images-list");
  const count = document.getElementById("image-filter-count");
  list.innerHTML = "";
  const filteredImages = getFilteredImages();
  if (!state.renderedImageCount) {
    state.renderedImageCount = Math.min(state.imageRenderBatchSize, filteredImages.length);
  } else {
    state.renderedImageCount = Math.min(state.renderedImageCount, filteredImages.length);
  }
  const visibleImages = filteredImages.slice(0, state.renderedImageCount);

  count.textContent = state.images.length
    ? `${visibleImages.length} / ${filteredImages.length} shown (${state.images.length} total)`
    : "";
  renderCropEditor();

  if (!filteredImages.length) {
    const empty = document.createElement("div");
    empty.className = "muted image-empty";
    empty.textContent = state.images.length ? "No images match this filter." : "No images uploaded yet.";
    list.appendChild(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  visibleImages.forEach((img) => {
    const row = document.createElement("div");
    row.className = "list-item image-item";

    const previewButton = document.createElement("button");
    previewButton.className = "image-preview-btn";
    previewButton.type = "button";
    previewButton.title = `View ${img.original_name || "image"} fullscreen`;
    previewButton.onclick = () => openImageViewer(img);

    const thumb = document.createElement("img");
    thumb.className = "image-thumb";
    thumb.src = img.url;
    thumb.alt = img.original_name || "Dataset image";
    thumb.loading = "lazy";
    previewButton.appendChild(thumb);

    const meta = document.createElement("div");
    meta.className = "image-meta";
    const name = document.createElement("span");
    name.className = "image-name";
    name.textContent = img.original_name || "";
    meta.appendChild(name);
    const size = document.createElement("small");
    size.className = "image-size";
    size.textContent = imageSizeLabel(img);
    meta.appendChild(size);
    thumb.onload = () => updateImageDimensions(img, thumb, size);

    const actions = document.createElement("div");
    actions.className = "item-actions";
    const deleteButton = document.createElement("button");
    deleteButton.className = "btn btn-secondary";
    deleteButton.type = "button";
    deleteButton.textContent = "Delete";
    deleteButton.onclick = () => deleteImage(img.id);
    actions.appendChild(deleteButton);

    row.appendChild(previewButton);
    row.appendChild(meta);
    row.appendChild(actions);
    fragment.appendChild(row);
  });
  list.appendChild(fragment);

  if (state.renderedImageCount < filteredImages.length) {
    const loadMore = document.createElement("button");
    loadMore.className = "load-more-images btn btn-secondary";
    loadMore.type = "button";
    loadMore.textContent = `Load ${Math.min(state.imageRenderBatchSize, filteredImages.length - state.renderedImageCount)} more`;
    loadMore.onclick = () => {
      state.renderedImageCount = Math.min(
        state.renderedImageCount + state.imageRenderBatchSize,
        filteredImages.length
      );
      renderImages();
    };
    list.appendChild(loadMore);
  }
}

function getFilteredImages() {
  if (!state.imageNameFilter) return state.images;
  return state.images.filter((img) => (img.original_name || "").toLowerCase().includes(state.imageNameFilter));
}

function imageSizeLabel(img) {
  if (img.width && img.height) {
    return `${img.width} x ${img.height} px`;
  }
  return "Size unavailable";
}

function updateImageDimensions(img, imageEl, sizeEl) {
  if (!imageEl.naturalWidth || !imageEl.naturalHeight) return;
  if (!img.width || !img.height) {
    img.width = imageEl.naturalWidth;
    img.height = imageEl.naturalHeight;
  }
  if (sizeEl) {
    sizeEl.textContent = imageSizeLabel(img);
  }
  renderCropEditor();
}

function cropOutputSize(width, height) {
  if (!width || !height) return null;
  const cropWidth = Math.max(1, Math.round(width * (state.cropBox.right - state.cropBox.left)));
  const cropHeight = Math.max(1, Math.round(height * (state.cropBox.bottom - state.cropBox.top)));
  return {
    label: `${cropWidth} x ${cropHeight}`,
  };
}

function openCropEditor() {
  const filteredImages = getFilteredImages();
  const select = document.getElementById("transformation-select");
  if (!filteredImages.length) {
    state.activeTransformation = "";
    if (select) select.value = "";
    t.notify("no filtered images to crop");
    return;
  }

  const editor = document.getElementById("crop-editor");
  const image = document.getElementById("crop-editor-img");
  if (!editor || !image) return;

  const previewImage = filteredImages[0];
  image.src = previewImage.url;
  image.alt = previewImage.original_name || "Crop preview";
  editor.hidden = false;
  document.body.classList.add("crop-editor-open");
  renderCropEditor();
  requestAnimationFrame(updateCropFrame);
}

function closeCropEditor() {
  const editor = document.getElementById("crop-editor");
  if (!editor || editor.hidden) return;
  editor.hidden = true;
  state.activeTransformation = "";
  state.cropDragHandle = null;
  const select = document.getElementById("transformation-select");
  if (select) select.value = "";
  document.body.classList.remove("crop-editor-open");
}

function renderCropEditor() {
  const editor = document.getElementById("crop-editor");
  if (!editor || editor.hidden) return;

  const filteredImages = getFilteredImages();
  const scope = document.getElementById("crop-editor-scope");
  const preview = document.getElementById("crop-editor-img");
  const originalSize = document.getElementById("crop-original-size");
  const outputSize = document.getElementById("crop-output-size");
  const createdCount = document.getElementById("crop-created-count");
  const applyBtn = document.getElementById("apply-crop-btn");
  if (!scope || !preview || !originalSize || !outputSize || !createdCount || !applyBtn) return;

  const imageCount = filteredImages.length;
  scope.textContent = imageCount
    ? `Current filter targets ${imageCount} image${imageCount === 1 ? "" : "s"}.`
    : "No images match the current filter.";
  createdCount.textContent = imageCount ? `${imageCount} image${imageCount === 1 ? "" : "s"}` : "-";
  applyBtn.disabled = !state.selectedDatasetId || imageCount === 0;
  applyBtn.textContent = "Create Crops";

  const img = filteredImages[0];
  if (!img) {
    originalSize.textContent = "-";
    outputSize.textContent = "-";
    return;
  }

  const setSizes = (width, height) => {
    originalSize.textContent = width && height ? `${width} x ${height}` : "-";
    const out = cropOutputSize(width, height);
    outputSize.textContent = out ? out.label : "-";
  };

  if (img.width && img.height) {
    setSizes(Number(img.width), Number(img.height));
    updateCropFrame();
    return;
  }

  if (preview.complete && preview.naturalWidth && preview.naturalHeight) {
    setSizes(preview.naturalWidth, preview.naturalHeight);
  } else {
    originalSize.textContent = "Loading...";
    outputSize.textContent = "Loading...";
    preview.onload = () => {
      setSizes(preview.naturalWidth, preview.naturalHeight);
      updateCropFrame();
    };
  }
  updateCropFrame();
    // sync normalized crop box into state so existing UI shows sizes
    try {
      const nb = window.SharedCropEditor.getNormalizedCropBox();
      if (nb && nb.length === 4) {
        state.cropBox.left = nb[0]; state.cropBox.top = nb[1]; state.cropBox.right = nb[2]; state.cropBox.bottom = nb[3];
      }
    } catch (e) {}
    // update derived UI values
    const natural = (window.SharedCropEditor && typeof window.SharedCropEditor.getImageNaturalSize === 'function') ? window.SharedCropEditor.getImageNaturalSize() : null;
    const iw = (natural && natural.width) || img.width || (preview.naturalWidth || null);
    const ih = (natural && natural.height) || img.height || (preview.naturalHeight || null);
    const out = cropOutputSize(iw, ih);
    document.getElementById('crop-output-size').textContent = out ? out.label : '-';
    document.getElementById('crop-created-count').textContent = filteredImages.length ? `${filteredImages.length} image${filteredImages.length===1? '':'s'}` : '-';
}

function updateCropFrame() {
  const editor = document.getElementById("crop-editor");
  const stage = document.getElementById("crop-stage");
  const image = document.getElementById("crop-editor-img");
  const frame = document.getElementById("crop-frame");
  const box = document.getElementById("crop-box");
  if (!editor || editor.hidden || !stage || !image || !frame || !box) return;
  if (!image.complete || !image.naturalWidth || !image.naturalHeight) return;
  // Prefer SharedCropEditor state when available
  const shared = window.SharedCropEditor;
  if (shared && typeof shared.getNormalizedCropBox === 'function') {
    const nb = shared.getNormalizedCropBox();
    if (nb && nb.length === 4) {
      state.cropBox.left = nb[0];
      state.cropBox.top = nb[1];
      state.cropBox.right = nb[2];
      state.cropBox.bottom = nb[3];
    }
  }

  const stageRect = stage.getBoundingClientRect();
  const imageRect = image.getBoundingClientRect();
  frame.style.left = `${imageRect.left - stageRect.left}px`;
  frame.style.top = `${imageRect.top - stageRect.top}px`;
  frame.style.width = `${imageRect.width}px`;
  frame.style.height = `${imageRect.height}px`;
  box.style.left = `${state.cropBox.left * 100}%`;
  box.style.top = `${state.cropBox.top * 100}%`;
  box.style.width = `${(state.cropBox.right - state.cropBox.left) * 100}%`;
  box.style.height = `${(state.cropBox.bottom - state.cropBox.top) * 100}%`;
}

function openImageViewer(img) {
  const viewer = document.getElementById("image-viewer");
  const image = document.getElementById("image-viewer-img");
  const caption = document.getElementById("image-viewer-caption");
  image.src = img.url;
  image.alt = img.original_name || "Dataset image";
  caption.textContent = img.original_name || "";
  viewer.hidden = false;
  document.body.classList.add("image-viewer-open");
}

function closeImageViewer() {
  const viewer = document.getElementById("image-viewer");
  if (viewer.hidden) return;
  viewer.hidden = true;
  document.getElementById("image-viewer-img").src = "";
  document.body.classList.remove("image-viewer-open");
}

async function uploadImages() {
  if (!state.selectedDatasetId) return t.notify("select a dataset first");
  const input = document.getElementById("image-upload");
  if (!input.files || input.files.length === 0) return t.notify("select images first");
  const form = new FormData();
  for (const f of input.files) form.append("files", f);

  try {
    await t.api(`/training/datasets/${state.selectedDatasetId}/images`, { method: "POST", body: form });
    input.value = "";
    state.datasets = await t.fetchDatasets();
    renderDatasets();
    await fetchImages();
  } catch (e) {
    t.notify(e.message);
  }
}

async function applyCropToFilteredImages() {
  if (!state.selectedDatasetId) return t.notify("select a dataset first");
  const images = getFilteredImages();
  if (!images.length) return t.notify("no filtered images to crop");

  const button = document.getElementById("apply-crop-btn");
  const status = document.getElementById("crop-editor-status");
  const imageIds = images.map((img) => img.id);
  button.disabled = true;
  status.textContent = `Creating static crops for ${images.length} image${images.length === 1 ? "" : "s"}...`;

  try {
    const shared = window.SharedCropEditor;
    let cropBox = state.cropBox;
    if (shared && typeof shared.getNormalizedCropBox === 'function') {
      const nb = shared.getNormalizedCropBox();
      if (nb && nb.length === 4) cropBox = { left: nb[0], top: nb[1], right: nb[2], bottom: nb[3] };
    }

    const result = await t.api(`/training/datasets/${state.selectedDatasetId}/images/transform/crop`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: "static",
        image_ids: imageIds,
        crop_left: cropBox.left,
        crop_top: cropBox.top,
        crop_right: cropBox.right,
        crop_bottom: cropBox.bottom,
        create_new_dataset: document.getElementById("crop-create-dataset") ? document.getElementById("crop-create-dataset").checked : false,
        new_dataset_name: document.getElementById("crop-new-dataset-name") ? document.getElementById("crop-new-dataset-name").value.trim() : "",
      }),
    });
    status.textContent = `Created ${result.created_count} cropped image${result.created_count === 1 ? "" : "s"}.`;
    state.datasets = await t.fetchDatasets();
    if (result.target_dataset_id && result.target_dataset_id !== state.selectedDatasetId) {
      state.selectedDatasetId = result.target_dataset_id;
      t.setStoredDatasetId(state.selectedDatasetId);
    }
    renderDatasets();
    await fetchImages();
  } catch (e) {
    status.textContent = "";
    t.notify(e.message);
  } finally {
    renderCropEditor();
  }
}

async function deleteImage(imageId) {
  if (!confirm("Delete this image and all its labels?")) return;
  try {
    await t.api(`/training/datasets/${state.selectedDatasetId}/images/${imageId}`, { method: "DELETE" });
    state.datasets = await t.fetchDatasets();
    renderDatasets();
    await fetchImages();
  } catch (e) {
    t.notify(e.message);
  }
}
