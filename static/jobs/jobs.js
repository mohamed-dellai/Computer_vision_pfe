// ─────────────────────────────────────────────────────────────────────────────
//  Jobs Wizard – state
// ─────────────────────────────────────────────────────────────────────────────
let currentStep = 1;
const TOTAL_STEPS = 6;

const jobConfig = {
  camera_id: null,
  model: null,
  tracker_file: 'bytetrack.yaml',
  conf: 0.25,
  iou: 0.7,
  rtsp_width: null,
  rtsp_height: null,
  rtsp_fps: 15,
  rtsp_buffer_size: 1,
  rtsp_reconnect_delay: 3.0,
  rtsp_read_timeout: 5.0,
  shop_id: null,
  crop_coords: null,   // [x1,y1,x2,y2] in RTSP frame
  line_coords: null,   // [x1,y1,x2,y2] in cropped frame (or RTSP frame if no crop)
};

// Line drawing state
let snapshotImage = null;
let isDrawing = false;
let startPoint = null;
let currentPoint = null;

// Camera name lookup
let camerasCache = {};

// ─────────────────────────────────────────────────────────────────────────────
//  Boot
// ─────────────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initWizard();
  fetchCameras();
  fetchModels();
  fetchTrackers();
  fetchJobs();
  fetchPresets();
  setInterval(fetchJobs, 3000);
  setupLineModal();

  const savedTracker = sessionStorage.getItem('selectedTracker');
  if (savedTracker) {
    document.getElementById('tracker-select').value = savedTracker;
    sessionStorage.removeItem('selectedTracker');
    showNotification(`Using tracker: ${savedTracker}`, 'info');
  }

  // Pre-select camera from URL param
  const urlParams = new URLSearchParams(window.location.search);
  const preSelectedCam = urlParams.get('camera_id');
  if (preSelectedCam) {
    jobConfig.camera_id = preSelectedCam;
  }
});

// ─────────────────────────────────────────────────────────────────────────────
//  Wizard – Navigation
// ─────────────────────────────────────────────────────────────────────────────
function initWizard() {
  // Wire next / back buttons via delegation
  document.querySelector('.wiz-panels').addEventListener('click', (e) => {
    if (e.target.closest('.wiz-next-btn')) wizNext();
    if (e.target.closest('.wiz-back-btn')) wizBack();
  });

  // Stepper circles: click to go back to a completed step
  document.querySelectorAll('.wiz-step').forEach(el => {
    el.addEventListener('click', () => {
      const n = parseInt(el.dataset.step, 10);
      if (!el.classList.contains('locked') && n < currentStep) {
        goToStep(n);
      }
    });
  });

  // Crop controls
  document.getElementById('crop-area-check').addEventListener('change', onCropCheckChange);
  document.getElementById('crop-edit-btn').addEventListener('click', openCropEditor);

  // Line controls
  document.getElementById('line-counting-check').addEventListener('change', onLineCheckChange);
  document.getElementById('line-edit-btn').addEventListener('click', () => openLineModal());

  // Save preset
  document.getElementById('save-preset-btn').addEventListener('click', promptSavePreset);
  // Start job
  document.getElementById('start-job-btn').addEventListener('click', startJob);

  // Preset toolbar
  document.getElementById('load-preset-btn').addEventListener('click', loadPreset);
  document.getElementById('delete-preset-btn').addEventListener('click', deletePreset);

  goToStep(1);
}

function wizNext() {
  if (!validateStep(currentStep)) return;
  collectStep(currentStep);
  if (currentStep < TOTAL_STEPS) goToStep(currentStep + 1);
}

function wizBack() {
  if (currentStep > 1) goToStep(currentStep - 1);
}

function goToStep(n) {
  // Hide current, show new
  document.getElementById(`wiz-panel-${currentStep}`).hidden = true;
  currentStep = n;
  document.getElementById(`wiz-panel-${n}`).hidden = false;

  updateStepIndicator();
  updateSummaryBar();

  if (n === 6) buildReviewSummary();
  if (n === 5) updateLineStepDesc();
}

function updateStepIndicator() {
  document.querySelectorAll('.wiz-step').forEach(el => {
    const s = parseInt(el.dataset.step, 10);
    el.classList.remove('active', 'completed', 'locked');
    if (s < currentStep) el.classList.add('completed');
    else if (s === currentStep) el.classList.add('active');
    else el.classList.add('locked');
  });

  // Connector state
  document.querySelectorAll('.wiz-connector').forEach((c, i) => {
    c.classList.toggle('filled', i + 1 < currentStep);
  });
}

function validateStep(n) {
  if (n === 1) {
    const val = document.getElementById('camera-select').value;
    if (!val) { showNotification('Please select a camera.', 'error'); return false; }
  }
  if (n === 2) {
    const val = document.getElementById('model-select').value;
    if (!val) { showNotification('Please select a model.', 'error'); return false; }
  }
  return true;
}

function collectStep(n) {
  if (n === 1) jobConfig.camera_id = document.getElementById('camera-select').value;
  if (n === 2) jobConfig.model = document.getElementById('model-select').value;
  if (n === 3) {
    jobConfig.tracker_file = document.getElementById('tracker-select').value;
    jobConfig.conf = parseFloat(document.getElementById('conf').value) || 0.25;
    jobConfig.iou = parseFloat(document.getElementById('iou').value) || 0.7;
  }
  if (n === 4) {
    const w = document.getElementById('rtsp_width').value.trim();
    const h = document.getElementById('rtsp_height').value.trim();
    jobConfig.rtsp_width = w ? parseInt(w) : null;
    jobConfig.rtsp_height = h ? parseInt(h) : null;
    jobConfig.rtsp_fps = parseInt(document.getElementById('rtsp_fps').value) || 15;
    jobConfig.rtsp_buffer_size = parseInt(document.getElementById('rtsp_buffer_size').value) || 1;
    jobConfig.rtsp_reconnect_delay = parseFloat(document.getElementById('rtsp_reconnect_delay').value) || 3.0;
    jobConfig.rtsp_read_timeout = parseFloat(document.getElementById('rtsp_read_timeout').value) || 5.0;
    jobConfig.shop_id = document.getElementById('shop-id').value.trim() || null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Summary Bar (pills showing prior selections)
// ─────────────────────────────────────────────────────────────────────────────
function updateSummaryBar() {
  const bar = document.getElementById('wiz-summary');
  const pills = [];

  if (currentStep > 1 && jobConfig.camera_id) {
    const cam = camerasCache[jobConfig.camera_id];
    pills.push({ icon: 'CA', text: cam ? cam.name : jobConfig.camera_id.slice(0, 8) });
  }
  if (currentStep > 2 && jobConfig.model) {
    pills.push({ icon: 'MD', text: jobConfig.model });
  }
  if (currentStep > 3 && jobConfig.tracker_file) {
    const base = jobConfig.tracker_file.replace('.yaml', '').replace(/.*\//, '');
    pills.push({ icon: 'TK', text: base });
  }
  if (currentStep > 4) {
    const res = (jobConfig.rtsp_width && jobConfig.rtsp_height)
      ? `${jobConfig.rtsp_width}×${jobConfig.rtsp_height}`
      : 'default res';
    pills.push({ icon: 'RS', text: res });
  }
  if (currentStep > 5) {
    pills.push(jobConfig.crop_coords
      ? { icon: 'CR', text: 'Crop set', highlight: true }
      : { icon: 'CR', text: 'No crop' });
  }

  bar.innerHTML = pills.map(p => `
    <span class="summary-pill${p.highlight ? ' summary-pill-active' : ''}">
      <span class="summary-pill-icon">${p.icon}</span>
      <span class="summary-pill-text">${p.text}</span>
    </span>
  `).join('');

  bar.hidden = pills.length === 0;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Review Summary (step 6)
// ─────────────────────────────────────────────────────────────────────────────
function buildReviewSummary() {
  const cam = camerasCache[jobConfig.camera_id];
  const camLabel = cam ? `${cam.name}` : (jobConfig.camera_id || '—');
  const res = (jobConfig.rtsp_width && jobConfig.rtsp_height)
    ? `${jobConfig.rtsp_width} × ${jobConfig.rtsp_height} px`
    : 'Camera default';
  const trackerBase = (jobConfig.tracker_file || 'bytetrack.yaml')
    .replace(/.*\//, '').replace('.yaml', '');
  const cropText = jobConfig.crop_coords
    ? `(${jobConfig.crop_coords[0]}, ${jobConfig.crop_coords[1]}) → (${jobConfig.crop_coords[2]}, ${jobConfig.crop_coords[3]})`
    : 'None';
  const lineText = jobConfig.line_coords
    ? `(${jobConfig.line_coords[0]}, ${jobConfig.line_coords[1]}) → (${jobConfig.line_coords[2]}, ${jobConfig.line_coords[3]})`
    : 'None';

  const rows = [
    ['Camera', camLabel],
    ['Model', jobConfig.model || '—'],
    ['Tracker', trackerBase],
    ['Conf / IOU', `${jobConfig.conf} / ${jobConfig.iou}`],
    ['Resolution', res],
    ['Crop area', cropText],
    ['Counting line', lineText],
    ['FPS', `${jobConfig.rtsp_fps}`],
  ];

  document.getElementById('review-grid').innerHTML = rows.map(([k, v]) => `
    <span class="review-key">${k}</span>
    <span class="review-val">${v}</span>
  `).join('');
}

function updateLineStepDesc() {
  const desc = document.getElementById('line-step-desc');
  if (jobConfig.crop_coords) {
    desc.textContent = 'Draw a line on the cropped area. Coordinates will be measured inside the crop region.';
    const badge = document.getElementById('line-modal-badge');
    badge.textContent = 'Coords in cropped space';
    badge.hidden = false;
  } else {
    desc.textContent = 'Draw a line on the camera view. Objects crossing it will be counted.';
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Crop Step
// ─────────────────────────────────────────────────────────────────────────────
async function onCropCheckChange(e) {
  if (e.target.checked) {
    // Ensure camera is set
    if (!jobConfig.camera_id) {
      jobConfig.camera_id = document.getElementById('camera-select').value;
    }
    if (!jobConfig.camera_id) {
      showNotification('Camera not selected – go back to step 1.', 'error');
      e.target.checked = false;
      return;
    }
    await openCropEditor();
    if (!jobConfig.crop_coords) e.target.checked = false;
  } else {
    jobConfig.crop_coords = null;
    // Also clear line coords since they were in cropped space
    jobConfig.line_coords = null;
    document.getElementById('crop-coords-display').hidden = true;
    document.getElementById('line-coords-display').hidden = true;
    document.getElementById('line-counting-check').checked = false;
  }
}

async function openCropEditor() {
  collectStep(4); // ensure resolution is fresh
  const cameraId = jobConfig.camera_id || document.getElementById('camera-select').value;
  if (!cameraId) { showNotification('Select a camera first.', 'error'); return; }

  let url = `/cameras/${cameraId}/snapshot`;
  const params = new URLSearchParams();
  if (jobConfig.rtsp_width) params.append('width', jobConfig.rtsp_width);
  if (jobConfig.rtsp_height) params.append('height', jobConfig.rtsp_height);
  if (params.toString()) url += `?${params.toString()}`;

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error('Could not fetch snapshot');
    const blob = await res.blob();
    const imgUrl = URL.createObjectURL(blob);
    const coords = await SharedCropEditor.openWithImage(imgUrl);
    URL.revokeObjectURL(imgUrl);

    if (!coords) return; // user cancelled

    jobConfig.crop_coords = coords;
    // Invalidate any previously drawn line since coordinate space changed
    jobConfig.line_coords = null;
    document.getElementById('line-coords-display').hidden = true;
    document.getElementById('line-counting-check').checked = false;

    document.getElementById('crop-values').textContent =
      `(${coords[0]}, ${coords[1]}) → (${coords[2]}, ${coords[3]})`;
    document.getElementById('crop-coords-display').hidden = false;
    document.getElementById('crop-area-check').checked = true;
  } catch (err) {
    showNotification('Failed to get snapshot: ' + err.message, 'error');
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Line Step
// ─────────────────────────────────────────────────────────────────────────────
async function onLineCheckChange(e) {
  if (e.target.checked) {
    const cameraId = jobConfig.camera_id || document.getElementById('camera-select').value;
    if (!cameraId) {
      showNotification('Camera not selected – go back to step 1.', 'error');
      e.target.checked = false;
      return;
    }
    try {
      await openLineModal();
    } catch (err) {
      showNotification('Failed to get snapshot: ' + err.message, 'error');
      e.target.checked = false;
    }
    if (!jobConfig.line_coords) e.target.checked = false;
  } else {
    jobConfig.line_coords = null;
    document.getElementById('line-coords-display').hidden = true;
  }
}

function setupLineModal() {
  const canvas = document.getElementById('line-canvas');

  canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    startPoint = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    currentPoint = { ...startPoint };
    isDrawing = true;
    redrawLineCanvas();
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    currentPoint = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    redrawLineCanvas();
  });

  canvas.addEventListener('mouseup', () => { isDrawing = false; });

  document.getElementById('confirm-line').addEventListener('click', () => {
    if (!startPoint || !currentPoint) {
      showNotification('Draw a line first.', 'error');
      return;
    }
    const canvas = document.getElementById('line-canvas');
    // Scale display coords → actual image pixel coords
    const scaleX = snapshotImage.width / canvas.width;
    const scaleY = snapshotImage.height / canvas.height;

    jobConfig.line_coords = [
      Math.round(startPoint.x * scaleX),
      Math.round(startPoint.y * scaleY),
      Math.round(currentPoint.x * scaleX),
      Math.round(currentPoint.y * scaleY),
    ];

    document.getElementById('coords-values').textContent =
      `(${jobConfig.line_coords[0]}, ${jobConfig.line_coords[1]}) → (${jobConfig.line_coords[2]}, ${jobConfig.line_coords[3]})`;
    document.getElementById('line-coords-display').hidden = false;
    document.getElementById('line-counting-check').checked = true;
    document.getElementById('line-modal').style.display = 'none';
  });

  document.getElementById('cancel-line').addEventListener('click', () => {
    document.getElementById('line-modal').style.display = 'none';
    if (!jobConfig.line_coords) {
      document.getElementById('line-counting-check').checked = false;
    }
  });
}

async function openLineModal() {
  const modal = document.getElementById('line-modal');
  const canvas = document.getElementById('line-canvas');

  // Reset drawing state
  startPoint = null;
  currentPoint = null;
  isDrawing = false;

  const cameraId = jobConfig.camera_id || document.getElementById('camera-select').value;
  collectStep(4); // ensure resolution values are current

  let url = `/cameras/${cameraId}/snapshot`;
  const params = new URLSearchParams();
  if (jobConfig.rtsp_width) params.append('width', jobConfig.rtsp_width);
  if (jobConfig.rtsp_height) params.append('height', jobConfig.rtsp_height);
  if (params.toString()) url += `?${params.toString()}`;

  const res = await fetch(url);
  if (!res.ok) throw new Error('Could not fetch snapshot');
  const blob = await res.blob();

  const fullImg = new Image();
  fullImg.src = URL.createObjectURL(blob);
  await new Promise(r => { fullImg.onload = r; });

  // ── Coordinate-space fix ──────────────────────────────────────────────────
  // If a crop is active, we crop the snapshot client-side before showing it on
  // the canvas.  Line coordinates drawn by the user are therefore naturally in
  // the cropped-frame coordinate space – exactly what the processor expects.
  if (jobConfig.crop_coords) {
    const [cx1, cy1, cx2, cy2] = jobConfig.crop_coords;
    const cropW = cx2 - cx1;
    const cropH = cy2 - cy1;

    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = cropW;
    tmpCanvas.height = cropH;
    tmpCanvas.getContext('2d').drawImage(fullImg, cx1, cy1, cropW, cropH, 0, 0, cropW, cropH);

    const croppedImg = new Image();
    croppedImg.src = tmpCanvas.toDataURL('image/jpeg', 0.92);
    await new Promise(r => { croppedImg.onload = r; });
    snapshotImage = croppedImg;
    URL.revokeObjectURL(fullImg.src);

    // Update badge
    const badge = document.getElementById('line-modal-badge');
    badge.textContent = `Drawing on cropped area (${cropW}×${cropH})`;
    badge.hidden = false;
  } else {
    snapshotImage = fullImg;
    document.getElementById('line-modal-badge').hidden = true;
  }

  // Fit canvas to modal
  const maxW = 700;
  const maxH = 500;
  let w = snapshotImage.width;
  let h = snapshotImage.height;
  if (w > maxW) { h = h * (maxW / w); w = maxW; }
  if (h > maxH) { w = w * (maxH / h); h = maxH; }
  canvas.width = Math.round(w);
  canvas.height = Math.round(h);

  modal.style.display = 'flex';
  redrawLineCanvas();
}

function redrawLineCanvas() {
  const canvas = document.getElementById('line-canvas');
  if (!snapshotImage) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(snapshotImage, 0, 0, canvas.width, canvas.height);

  if (startPoint && currentPoint) {
    ctx.beginPath();
    ctx.moveTo(startPoint.x, startPoint.y);
    ctx.lineTo(currentPoint.x, currentPoint.y);
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.fillStyle = '#ff4444';
    [startPoint, currentPoint].forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
      ctx.fill();
    });
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Presets
// ─────────────────────────────────────────────────────────────────────────────
let presetsCache = [];

async function fetchPresets() {
  try {
    const r = await fetch('/jobs/presets');
    presetsCache = await r.json();
    renderPresetsDropdown();
  } catch (e) {
    console.warn('Could not fetch presets:', e);
  }
}

function renderPresetsDropdown() {
  const sel = document.getElementById('preset-select');
  sel.innerHTML = '<option value="">Select a saved preset…</option>';
  presetsCache.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.id;
    const res = (p.rtsp_width && p.rtsp_height) ? ` ${p.rtsp_width}×${p.rtsp_height}` : '';
    const cam = p.camera_id && camerasCache[p.camera_id] ? ` · ${camerasCache[p.camera_id].name}` : '';
    opt.textContent = `${p.name}${cam}${res}`;
    sel.appendChild(opt);
  });
  const toolbar = document.getElementById('preset-toolbar');
  toolbar.hidden = presetsCache.length === 0;
}

function loadPreset() {
  const id = document.getElementById('preset-select').value;
  if (!id) return;
  const preset = presetsCache.find(p => p.id === id);
  if (!preset) return;

  jobConfig.crop_coords = preset.crop_coords || null;
  jobConfig.line_coords = preset.line_coords || null;

  // Optionally apply resolution if set in preset
  if (preset.rtsp_width) {
    jobConfig.rtsp_width = preset.rtsp_width;
    document.getElementById('rtsp_width').value = preset.rtsp_width;
  }
  if (preset.rtsp_height) {
    jobConfig.rtsp_height = preset.rtsp_height;
    document.getElementById('rtsp_height').value = preset.rtsp_height;
  }

  // Update crop display
  if (jobConfig.crop_coords) {
    const c = jobConfig.crop_coords;
    document.getElementById('crop-values').textContent = `(${c[0]}, ${c[1]}) → (${c[2]}, ${c[3]})`;
    document.getElementById('crop-coords-display').hidden = false;
    document.getElementById('crop-area-check').checked = true;
  } else {
    document.getElementById('crop-coords-display').hidden = true;
    document.getElementById('crop-area-check').checked = false;
  }

  // Update line display
  if (jobConfig.line_coords) {
    const l = jobConfig.line_coords;
    document.getElementById('coords-values').textContent = `(${l[0]}, ${l[1]}) → (${l[2]}, ${l[3]})`;
    document.getElementById('line-coords-display').hidden = false;
    document.getElementById('line-counting-check').checked = true;
  } else {
    document.getElementById('line-coords-display').hidden = true;
    document.getElementById('line-counting-check').checked = false;
  }

  // Warn if resolution mismatch
  const warning = document.getElementById('preset-warning');
  const warningText = document.getElementById('preset-warning-text');
  const resMismatch =
    (preset.rtsp_width && jobConfig.rtsp_width && preset.rtsp_width !== jobConfig.rtsp_width) ||
    (preset.rtsp_height && jobConfig.rtsp_height && preset.rtsp_height !== jobConfig.rtsp_height);

  if (resMismatch) {
    warningText.textContent =
      `Preset was recorded at ${preset.rtsp_width}×${preset.rtsp_height}. ` +
      `Current resolution differs — coordinates may be offset.`;
    warning.hidden = false;
  } else {
    warning.hidden = true;
  }

  showNotification(`Preset "${preset.name}" loaded.`, 'success');
}

async function deletePreset() {
  const id = document.getElementById('preset-select').value;
  if (!id) return;
  const preset = presetsCache.find(p => p.id === id);
  if (!preset) return;
  if (!confirm(`Delete preset "${preset.name}"?`)) return;
  try {
    await fetch(`/jobs/presets/${id}`, { method: 'DELETE' });
    await fetchPresets();
    showNotification('Preset deleted.', 'info');
  } catch (e) {
    showNotification('Failed to delete preset.', 'error');
  }
}

async function promptSavePreset() {
  if (!jobConfig.crop_coords && !jobConfig.line_coords) {
    showNotification('Nothing to save – configure crop or line first.', 'error');
    return;
  }
  const name = prompt('Preset name:');
  if (!name || !name.trim()) return;

  collectStep(currentStep);
  try {
    await fetch('/jobs/presets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: name.trim(),
        camera_id: jobConfig.camera_id,
        rtsp_width: jobConfig.rtsp_width,
        rtsp_height: jobConfig.rtsp_height,
        crop_coords: jobConfig.crop_coords,
        line_coords: jobConfig.line_coords,
      }),
    });
    await fetchPresets();
    showNotification(`Preset "${name.trim()}" saved.`, 'success');
  } catch (e) {
    showNotification('Failed to save preset.', 'error');
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Start / Stop Jobs
// ─────────────────────────────────────────────────────────────────────────────
async function startJob() {
  collectStep(currentStep);

  if (!jobConfig.camera_id) { showNotification('No camera selected.', 'error'); return; }
  if (!jobConfig.model) { showNotification('No model selected.', 'error'); return; }

  const payload = {
    camera_id: jobConfig.camera_id,
    model: jobConfig.model,
    conf: jobConfig.conf,
    iou: jobConfig.iou,
    tracker_file: jobConfig.tracker_file,
    rtsp_width: jobConfig.rtsp_width,
    rtsp_height: jobConfig.rtsp_height,
    rtsp_fps: jobConfig.rtsp_fps,
    rtsp_buffer_size: jobConfig.rtsp_buffer_size,
    rtsp_reconnect_delay: jobConfig.rtsp_reconnect_delay,
    rtsp_read_timeout: jobConfig.rtsp_read_timeout,
    shop_id: jobConfig.shop_id,
    crop_coords: jobConfig.crop_coords,
    line_coords: jobConfig.line_coords,
  };

  try {
    const r = await fetch('/jobs/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!r.ok) {
      const err = await r.json();
      showNotification('Error: ' + (err.error || 'unknown'), 'error');
      return;
    }
    if (jobConfig.shop_id) localStorage.setItem('shop_id', jobConfig.shop_id);

    showNotification('Job started successfully!', 'success');
    fetchJobs();
    resetWizard();
    document.querySelector('.jobs-list-card').scrollIntoView({ behavior: 'smooth' });
  } catch (e) {
    showNotification('Failed to start job: ' + e.message, 'error');
  }
}

function resetWizard() {
  // Reset config
  Object.assign(jobConfig, {
    camera_id: null, model: null, tracker_file: 'bytetrack.yaml',
    conf: 0.25, iou: 0.7, rtsp_width: null, rtsp_height: null,
    rtsp_fps: 15, rtsp_buffer_size: 1, rtsp_reconnect_delay: 3.0,
    rtsp_read_timeout: 5.0, shop_id: null, crop_coords: null, line_coords: null,
  });
  // Reset UI
  document.getElementById('crop-coords-display').hidden = true;
  document.getElementById('line-coords-display').hidden = true;
  document.getElementById('crop-area-check').checked = false;
  document.getElementById('line-counting-check').checked = false;
  document.getElementById('preset-warning').hidden = true;
  goToStep(1);
}

async function stopJob(job_id) {
  if (!confirm('Stop this job?')) return;
  try {
    await fetch('/jobs/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id }),
    });
    const stream = document.getElementById('stream');
    if (stream.src.includes(job_id)) closeViewer();
    fetchJobs();
  } catch (e) {
    showNotification('Failed to stop job.', 'error');
  }
}

function viewAI(job_id) {
  const viewer = document.getElementById('live-view-section');
  document.getElementById('stream').src = `/jobs/${job_id}/mjpeg`;
  viewer.style.display = 'block';
  viewer.scrollIntoView({ behavior: 'smooth' });
}

function closeViewer() {
  const viewer = document.getElementById('live-view-section');
  document.getElementById('stream').src = '';
  viewer.style.display = 'none';
}

// ─────────────────────────────────────────────────────────────────────────────
//  API Fetchers
// ─────────────────────────────────────────────────────────────────────────────
async function fetchCameras() {
  try {
    const r = await fetch('/cameras');
    const j = await r.json();
    camerasCache = j.cameras || {};
    const select = document.getElementById('camera-select');
    select.innerHTML = '';

    if (Object.keys(camerasCache).length === 0) {
      select.innerHTML = '<option disabled selected>No cameras available</option>';
      return;
    }

    // Pre-select from URL param or jobConfig
    const preSelected = jobConfig.camera_id ||
      new URLSearchParams(window.location.search).get('camera_id');

    for (const id in camerasCache) {
      const cam = camerasCache[id];
      const opt = document.createElement('option');
      opt.value = id;
      opt.textContent = `${cam.name} (${cam.rtsp})`;
      if (id === preSelected) opt.selected = true;
      select.appendChild(opt);
    }

    if (preSelected && camerasCache[preSelected]) {
      jobConfig.camera_id = preSelected;
    }

    // Refresh preset labels now that we have camera names
    renderPresetsDropdown();
  } catch (e) {
    console.error('Error fetching cameras:', e);
  }
}

async function fetchModels() {
  try {
    const r = await fetch('/models');
    const j = await r.json();
    const select = document.getElementById('model-select');
    select.innerHTML = '';
    if (!j.models || j.models.length === 0) {
      select.innerHTML = '<option disabled selected>No models available</option>';
      return;
    }
    j.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m;
      select.appendChild(opt);
    });
  } catch (e) {
    console.error('Error fetching models:', e);
  }
}

async function fetchTrackers() {
  try {
    const r = await fetch('/trackers');
    const trackers = await r.json();
    const select = document.getElementById('tracker-select');
    select.innerHTML = '';

    [
      { value: 'bytetrack.yaml', text: 'ByteTrack (Default)' },
      { value: 'botsort.yaml', text: 'BoTSORT (Default)' },
    ].forEach(opt => {
      const o = document.createElement('option');
      o.value = opt.value; o.textContent = opt.text;
      select.appendChild(o);
    });

    if (trackers.length > 0) {
      const sep = document.createElement('option');
      sep.disabled = true;
      sep.textContent = '── Saved Trackers ──';
      select.appendChild(sep);
      trackers.forEach(t => {
        const o = document.createElement('option');
        o.value = t.filename; o.textContent = t.name;
        select.appendChild(o);
      });
    }
  } catch (e) {
    console.error('Error fetching trackers:', e);
  }
}

async function fetchJobs() {
  try {
    const r = await fetch('/jobs');
    const j = await r.json();
    const d = document.getElementById('jobs-list');
    d.innerHTML = '';

    if (Object.keys(j).length === 0) {
      d.innerHTML = '<div class="empty-state">No active jobs running.</div>';
      return;
    }

    for (const id in j) {
      const job = j[id];
      const statusClass = job.status === 'running' ? 'status-running' : 'status-stopped';
      const camName = camerasCache[job.camera_id]
        ? camerasCache[job.camera_id].name
        : job.camera_id.slice(0, 8);

      const div = document.createElement('div');
      div.className = 'job-item';
      div.innerHTML = `
        <div class="job-header">
          <span class="job-title">Job: ${id.slice(0, 8)}…</span>
          <span class="job-status ${statusClass}">${job.status}</span>
        </div>
        <div class="job-details">
          <div class="detail-row">
            <span class="label">Camera</span>
            <span class="value">${camName}</span>
          </div>
          <div class="detail-row">
            <span class="label">Model</span>
            <span class="value">${job.model}</span>
          </div>
        </div>
        <div class="job-actions">
          <button class="action-btn view-btn" onclick="viewAI('${id}')">View Stream</button>
          <button class="action-btn stop-btn" onclick="stopJob('${id}')">Stop</button>
        </div>
      `;
      d.appendChild(div);
    }
  } catch (e) {
    console.error('Error fetching jobs:', e);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Notification
// ─────────────────────────────────────────────────────────────────────────────
let _notifTimer = null;
function showNotification(msg, type = 'info') {
  const el = document.getElementById('job-notification');
  el.className = `job-notification notif-${type}`;
  document.getElementById('job-notification-msg').textContent = msg;
  el.hidden = false;
  if (_notifTimer) clearTimeout(_notifTimer);
  _notifTimer = setTimeout(() => { el.hidden = true; }, 3500);
}
