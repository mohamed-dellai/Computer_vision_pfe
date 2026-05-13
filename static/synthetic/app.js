/* =============================================
   Synthetic Generator — app.js
   Canvas drawing, API calls, job polling
   ============================================= */

// ── State ────────────────────────────────────
const state = {
  backgroundPath: null,
  backgroundFilename: null,
  zones: [],          // [{points:[{x,y},...], color, id}]
  scaleLine: {yTop: 0.2, yBottom: 0.8}, // fraction of canvas height
  products: [],       // [{name, classId, color, models:[{filename,path}]}]
  currentTool: 'select',
  drawingPoints: [],
  selectedZoneIdx: -1,
  canvasScale: 1.0,
  imgNaturalW: 0,
  imgNaturalH: 0,
  activeJobId: null,
  pollTimer: null,
  dragScaleLine: null, // 'top'|'bottom'|null
};

const ZONE_COLORS = ['#58a6ff','#3fb950','#f78166','#d2a8ff','#ffa657','#79c0ff','#a5d6ff'];
const canvas = document.getElementById('main-canvas');
const ctx    = canvas.getContext('2d');
let bgImage  = null;
let calibrationImage = null;

// ── Background upload ────────────────────────
document.getElementById('bg-file').addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  uploadBackground(file);
});

const bgDrop = document.getElementById('bg-drop');
bgDrop.addEventListener('dragover', e => { e.preventDefault(); bgDrop.classList.add('drag-over'); });
bgDrop.addEventListener('dragleave', () => bgDrop.classList.remove('drag-over'));
bgDrop.addEventListener('drop', e => {
  e.preventDefault();
  bgDrop.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) uploadBackground(file);
});

async function uploadBackground(file) {
  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch('/synthetic/upload/background', {method:'POST', body:fd});
    const d = await r.json();
    if (!r.ok) { alert('Upload error: ' + d.error); return; }
    state.backgroundPath     = d.path;
    state.backgroundFilename = d.filename;
    document.getElementById('bg-name').textContent = file.name;
    document.getElementById('bg-preview-row').classList.remove('hidden');
    document.getElementById('canvas-placeholder').classList.add('hidden');
    loadBackgroundImage(`/synthetic/uploads/background/${d.filename}/file`);
  } catch(e) { alert('Upload failed: ' + e); }
}

function loadBackgroundImage(url) {
  bgImage = new Image();
  bgImage.onload = () => {
    state.imgNaturalW = bgImage.naturalWidth;
    state.imgNaturalH = bgImage.naturalHeight;
    // Clamp to fit container
    const container = document.getElementById('canvas-container');
    const maxW = container.clientWidth  - 32;
    const maxH = container.clientHeight - 32;
    const ratio = Math.min(maxW / bgImage.naturalWidth, maxH / bgImage.naturalHeight, 1);
    state.canvasScale = ratio;
    canvas.width  = Math.round(bgImage.naturalWidth  * ratio);
    canvas.height = Math.round(bgImage.naturalHeight * ratio);
    // Init scale lines
    state.scaleLine.yTop    = canvas.height * 0.2;
    state.scaleLine.yBottom = canvas.height * 0.8;
    document.getElementById('canvas-title').textContent = `${bgImage.naturalWidth} × ${bgImage.naturalHeight}`;
    redraw();
  };
  bgImage.src = url;
}

function clearBackground() {
  bgImage = null;
  calibrationImage = null;
  state.backgroundPath = null;
  state.backgroundFilename = null;
  canvas.width = 0; canvas.height = 0;
  document.getElementById('bg-preview-row').classList.add('hidden');
  document.getElementById('canvas-placeholder').classList.remove('hidden');
  document.getElementById('canvas-title').textContent = 'Upload a background image to begin';
  state.zones = [];
  renderZoneList();
}

// ── Canvas draw ──────────────────────────────
function redraw(mousePos) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state.currentTool === 'calibrate' && calibrationImage) {
    ctx.drawImage(calibrationImage, 0, 0, canvas.width, canvas.height);
  } else if (bgImage) {
    ctx.drawImage(bgImage, 0, 0, canvas.width, canvas.height);
  }

  // Draw scale lines
  ctx.setLineDash([6,4]);
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(255,220,0,0.85)';
  ctx.beginPath(); ctx.moveTo(0, state.scaleLine.yTop); ctx.lineTo(canvas.width, state.scaleLine.yTop); ctx.stroke();
  ctx.fillStyle = 'rgba(255,220,0,0.9)';
  ctx.font = '12px Inter,sans-serif';
  ctx.fillText(`▲ Top  scale=${document.getElementById('scale-top').value}`, 8, state.scaleLine.yTop - 5);
  ctx.strokeStyle = 'rgba(255,150,0,0.85)';
  ctx.beginPath(); ctx.moveTo(0, state.scaleLine.yBottom); ctx.lineTo(canvas.width, state.scaleLine.yBottom); ctx.stroke();
  ctx.fillStyle = 'rgba(255,150,0,0.9)';
  ctx.fillText(`▼ Bottom  scale=${document.getElementById('scale-bot').value}`, 8, state.scaleLine.yBottom + 15);
  ctx.setLineDash([]);

  // Draw completed zones
  state.zones.forEach((zone, i) => {
    if (zone.points.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(zone.points[0].x, zone.points[0].y);
    zone.points.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.closePath();
    ctx.fillStyle   = zone.color + '33';
    ctx.fill();
    ctx.strokeStyle = zone.color;
    ctx.lineWidth   = i === state.selectedZoneIdx ? 3 : 1.5;
    ctx.stroke();
    const cx = zone.points.reduce((s,p) => s+p.x, 0) / zone.points.length;
    const cy = zone.points.reduce((s,p) => s+p.y, 0) / zone.points.length;
    ctx.fillStyle = zone.color;
    ctx.font = 'bold 12px Inter,sans-serif';
    ctx.fillText(`Zone ${i+1}`, cx - 20, cy + 4);
  });

  // Active drawing polygon
  const pts = state.drawingPoints;
  if (pts.length > 0) {
    // Draw lines between points
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    // Preview line to mouse cursor
    if (mousePos) ctx.lineTo(mousePos.x, mousePos.y);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([4,3]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw points
    pts.forEach((p, i) => {
      const isFirst = i === 0;
      const snapRadius = 12;
      const nearFirst = mousePos && isFirst &&
        Math.hypot(mousePos.x - p.x, mousePos.y - p.y) < snapRadius;

      ctx.beginPath();
      ctx.arc(p.x, p.y, isFirst ? (pts.length >= 3 ? 8 : 5) : 4, 0, Math.PI*2);
      ctx.fillStyle = isFirst && pts.length >= 3 ? (nearFirst ? '#3fb950' : '#ffa657') : '#fff';
      ctx.fill();
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    // Closing-line preview when near first point
    if (mousePos && pts.length >= 3) {
      const snapRadius = 12;
      if (Math.hypot(mousePos.x - pts[0].x, mousePos.y - pts[0].y) < snapRadius) {
        ctx.beginPath();
        ctx.moveTo(pts[pts.length-1].x, pts[pts.length-1].y);
        ctx.lineTo(pts[0].x, pts[0].y);
        ctx.strokeStyle = '#3fb950';
        ctx.lineWidth   = 2;
        ctx.setLineDash([]);
        ctx.stroke();
        // "Close zone" label
        ctx.fillStyle = '#3fb950';
        ctx.font = 'bold 11px Inter,sans-serif';
        ctx.fillText('Click to close zone', pts[0].x + 12, pts[0].y - 8);
      }
    }
  }
}

// ── Canvas mouse events ──────────────────────
function canvasXY(e) {
  const r = canvas.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

let _lastMousePos = null;

canvas.addEventListener('click', e => {
  const {x, y} = canvasXY(e);

  // Scale line drag takes priority (handled in mousedown/mouseup)
  if (state.dragScaleLine) return;

  if (state.currentTool === 'draw') {
    const pts = state.drawingPoints;

    // If >= 3 points and click is near the first point → close zone
    if (pts.length >= 3) {
      const snapRadius = 12;
      if (Math.hypot(x - pts[0].x, y - pts[0].y) < snapRadius) {
        _finishZone();
        return;
      }
    }
    // Otherwise add point
    pts.push({x, y});
    _updateFinishBtn();
    redraw(_lastMousePos);

  } else if (state.currentTool === 'select') {
    state.selectedZoneIdx = -1;
    state.zones.forEach((zone, i) => {
      if (pointInPolygon(x, y, zone.points)) state.selectedZoneIdx = i;
    });
    renderZoneList(); redraw();
  } else if (state.currentTool === 'calibrate') {
    _runCalibration(x, y);
  }
});

canvas.addEventListener('mousedown', e => {
  const {x, y} = canvasXY(e);
  const tol = 10;
  if (Math.abs(y - state.scaleLine.yTop) < tol) { state.dragScaleLine = 'top'; return; }
  if (Math.abs(y - state.scaleLine.yBottom) < tol) { state.dragScaleLine = 'bottom'; return; }
});

canvas.addEventListener('mousemove', e => {
  const {x, y} = canvasXY(e);
  _lastMousePos = {x, y};

  if (state.dragScaleLine) {
    if (state.dragScaleLine === 'top') {
      state.scaleLine.yTop = Math.min(y, state.scaleLine.yBottom - 10);
    } else {
      state.scaleLine.yBottom = Math.max(y, state.scaleLine.yTop + 10);
    }
    redraw();
    return;
  }

  // Cursor hint
  const tol = 10;
  if (Math.abs(y - state.scaleLine.yTop) < tol || Math.abs(y - state.scaleLine.yBottom) < tol) {
    canvas.style.cursor = 'ns-resize';
  } else if (state.currentTool === 'draw') {
    // Snap cursor near first point
    const pts = state.drawingPoints;
    if (pts.length >= 3 && Math.hypot(x - pts[0].x, y - pts[0].y) < 12) {
      canvas.style.cursor = 'pointer';
    } else {
      canvas.style.cursor = 'crosshair';
    }
    redraw({x, y});  // live preview line
  } else {
    canvas.style.cursor = 'default';
  }
});

canvas.addEventListener('mouseup', () => { state.dragScaleLine = null; });

// Right-click: undo last point
canvas.addEventListener('contextmenu', e => {
  e.preventDefault();
  if (state.currentTool === 'draw' && state.drawingPoints.length > 0) {
    state.drawingPoints.pop();
    _updateFinishBtn();
    redraw(_lastMousePos);
  }
});

// Enter key: finish zone
document.addEventListener('keydown', e => {
  if (e.key === 'Enter' && state.currentTool === 'draw' && state.drawingPoints.length >= 3) {
    _finishZone();
  }
  if (e.key === 'Escape' && state.currentTool === 'draw') {
    state.drawingPoints = [];
    _updateFinishBtn();
    redraw();
  }
});

function _finishZone() {
  if (state.drawingPoints.length < 3) return;
  const color = ZONE_COLORS[state.zones.length % ZONE_COLORS.length];
  state.zones.push({ id: Date.now(), points: [...state.drawingPoints], color });
  state.drawingPoints = [];
  renderZoneList();
  _updateFinishBtn();
  redraw();
  // Flash zone list to confirm
  const zl = document.getElementById('zone-list');
  zl.style.transition = 'background .2s';
  zl.style.background = 'rgba(63,185,80,.15)';
  setTimeout(() => { zl.style.background = ''; }, 400);
}

function _updateFinishBtn() {
  const btn = document.getElementById('btn-finish-zone');
  if (!btn) return;
  const n = state.drawingPoints.length;
  btn.classList.toggle('hidden', n < 3);
  if (n >= 3) btn.textContent = `✅ Finish Zone (${n} pts)`;
}

function pointInPolygon(px, py, pts) {
  let inside = false;
  const n = pts.length;
  for (let i=0, j=n-1; i<n; j=i++) {
    const {x:xi,y:yi}=pts[i], {x:xj,y:yj}=pts[j];
    if (((yi>py)!==(yj>py)) && (px < (xj-xi)*(py-yi)/(yj-yi)+xi)) inside=!inside;
  }
  return inside;
}

// ── Tools ────────────────────────────────────
function setTool(tool) {
  state.currentTool = tool;
  state.drawingPoints = [];
  document.getElementById('btn-draw').classList.toggle('primary', tool==='draw');
  document.getElementById('btn-draw').classList.toggle('secondary', tool!=='draw');
  document.getElementById('btn-select').classList.toggle('primary', tool==='select');
  document.getElementById('btn-select').classList.toggle('secondary', tool!=='select');
  
  const btnCalib = document.getElementById('btn-calibrate');
  if (btnCalib) {
    btnCalib.classList.toggle('primary', tool==='calibrate');
    btnCalib.classList.toggle('secondary', tool!=='calibrate');
  }

  // Show/hide calibrate panel
  const calibPanel = document.getElementById('calibrate-panel');
  if (calibPanel) {
    calibPanel.classList.toggle('hidden', tool !== 'calibrate');
    if (tool === 'calibrate') _populateCalibProducts();
  }

  _updateFinishBtn();
  // Show/hide drawing instructions
  const hint = document.getElementById('draw-hint');
  if (hint) hint.classList.toggle('hidden', tool !== 'draw');
  
  // Clear calibration image if we leave the tool
  if (tool !== 'calibrate') calibrationImage = null;
  redraw();
}

function _populateCalibProducts() {
  const sel = document.getElementById('calib-product');
  if (!sel) return;
  sel.innerHTML = '';
  syncProductState();
  if (state.products.length === 0) {
    sel.innerHTML = '<option disabled selected>No products added</option>';
    return;
  }
  state.products.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.classId;
    opt.textContent = `${p.classId}: ${p.name}`;
    sel.appendChild(opt);
  });
}

async function _runCalibration(x, y) {
  if (!state.backgroundPath) { alert("Please upload a background first"); return; }
  syncProductState();
  if (state.products.length === 0) { alert("Please add at least one product class first"); return; }

  const pid = document.getElementById('calib-product').value;
  if (!pid) return;

  const rx = parseFloat(document.getElementById('calib-rx').value || 0);
  const ry = parseFloat(document.getElementById('calib-ry').value || 0);
  const rz = parseFloat(document.getElementById('calib-rz').value || 0);

  // Un-scale mouse coords back to background image size
  const realX = Math.round(x / state.canvasScale);
  const realY = Math.round(y / state.canvasScale);

  const cfg = buildConfig(); // Build full config to pass Scale & Lighting
  cfg.product_id = parseInt(pid);
  cfg.x = realX;
  cfg.y = realY;
  cfg.rx = rx;
  cfg.ry = ry;
  cfg.rz = rz;

  // Change cursor to waiting
  canvas.style.cursor = 'wait';
  try {
    const res = await fetch('/synthetic/calibrate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(cfg)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);

    const img = new Image();
    img.onload = () => {
      calibrationImage = img;
      redraw();
    };
    img.src = 'data:image/jpeg;base64,' + data.image;
  } catch (e) {
    alert("Calibration render failed: " + e.message);
  } finally {
    canvas.style.cursor = 'crosshair';
  }
}

function deleteSelectedZone() {
  if (state.selectedZoneIdx < 0) return;
  state.zones.splice(state.selectedZoneIdx, 1);
  state.selectedZoneIdx = -1;
  renderZoneList(); redraw();
}

function renderZoneList() {
  const el = document.getElementById('zone-list');
  el.innerHTML = '';
  if (state.zones.length === 0) {
    el.innerHTML = '<p class="hint" style="padding:4px 2px">No zones yet — draw one on the canvas</p>';
    return;
  }
  state.zones.forEach((z, i) => {
    const div = document.createElement('div');
    div.className = 'zone-item' + (i===state.selectedZoneIdx?' active':'');
    div.innerHTML = `<div class="zone-dot" style="background:${z.color}"></div>
      <span>Zone ${i+1} (${z.points.length} pts)</span>`;
    div.onclick = () => { state.selectedZoneIdx = i; renderZoneList(); redraw(); };
    el.appendChild(div);
  });
}


// ── Scale change ─────────────────────────────
function onScaleChange() {
  document.getElementById('val-scale-top').textContent = document.getElementById('scale-top').value;
  document.getElementById('val-scale-bot').textContent = document.getElementById('scale-bot').value;
  redraw();
}

// ── Zoom ─────────────────────────────────────
function zoomIn()    { state.canvasScale = Math.min(state.canvasScale * 1.25, 4); applyZoom(); }
function zoomOut()   { state.canvasScale = Math.max(state.canvasScale * 0.8,  0.1); applyZoom(); }
function zoomReset() { if (!bgImage) return; state.canvasScale=1; applyZoom(); }
function applyZoom() {
  if (!bgImage) return;
  canvas.style.width  = Math.round(bgImage.naturalWidth  * state.canvasScale) + 'px';
  canvas.style.height = Math.round(bgImage.naturalHeight * state.canvasScale) + 'px';
}

// ── Products ─────────────────────────────────
function addProductClass() {
  const tpl = document.getElementById('product-template');
  const clone = tpl.content.cloneNode(true);
  const idx = state.products.length;
  const color = ZONE_COLORS[idx % ZONE_COLORS.length];
  const card = clone.querySelector('.product-card');
  card.dataset.idx = idx;
  card.querySelector('.color-dot').style.background = color;
  card.querySelector('.product-id').value = idx;
  card.querySelector('.product-name').value = `product_${idx}`;

  // Wire up model file input
  const modelInput = card.querySelector('.model-file-input');
  modelInput.addEventListener('change', () => uploadModels(card, modelInput.files));

  document.getElementById('product-list').appendChild(clone);
  state.products.push({ name: `product_${idx}`, classId: idx, color, models: [] });
}

function removeProductClass(btn) {
  const card = btn.closest('.product-card');
  const idx = parseInt(card.dataset.idx);
  state.products.splice(idx, 1);
  card.remove();
  // Re-index remaining cards
  document.querySelectorAll('.product-card').forEach((c,i) => {
    c.dataset.idx = i;
    if (state.products[i]) state.products[i].classId = i;
  });
}

function triggerModelUpload(zone) {
  zone.querySelector('.model-file-input').click();
}

async function uploadModels(card, files) {
  const idx = parseInt(card.dataset.idx);
  const modelList = card.querySelector('.model-list');
  for (const file of files) {
    // Client-side extension check for instant feedback
    const ext = file.name.split('.').pop().toLowerCase();
    const allowed = ['obj','glb','gltf','stl','ply','dae'];
    if (!allowed.includes(ext)) {
      alert(`❌ "${file.name}" is not a 3D model file.\n\nStep 1 (Background) is for images (JPG/PNG).\nStep 2 (Product Classes) is for 3D models: .obj .glb .stl .ply\n\nPlease upload a 3D model file here.`);
      continue;
    }
    const fd = new FormData();
    fd.append('file', file);
    try {
      const r = await fetch('/synthetic/upload/model', {method:'POST', body:fd});
      const d = await r.json();
      if (!r.ok) { alert('Upload error: ' + d.error); continue; }
      state.products[idx].models.push({ filename: d.filename, path: d.path });
      const item = document.createElement('div');
      item.className = 'model-item';
      item.innerHTML = `<span title="${file.name}">${file.name}</span>
        <button class="btn-icon danger" onclick="removeModel(this,'${d.filename}',${idx})">✕</button>`;
      modelList.appendChild(item);
    } catch(e) { alert('Upload failed: ' + e); }
  }
}

function removeModel(btn, filename, productIdx) {
  btn.closest('.model-item').remove();
  if (state.products[productIdx]) {
    state.products[productIdx].models = state.products[productIdx].models.filter(m => m.filename !== filename);
  }
  fetch(`/synthetic/uploads/model/${filename}`, {method:'DELETE'});
}

// ── Sync product cards → state ────────────────
function syncProductState() {
  document.querySelectorAll('.product-card').forEach((card, i) => {
    if (!state.products[i]) return;
    state.products[i].name    = card.querySelector('.product-name').value.trim() || `product_${i}`;
    state.products[i].classId = parseInt(card.querySelector('.product-id').value) || i;
  });
}

// ── Build config object ───────────────────────
function buildConfig() {
  syncProductState();
  if (!state.backgroundPath) throw new Error('Please upload a background image first.');
  if (state.products.length === 0) throw new Error('Add at least one product class.');
  if (state.zones.length === 0) throw new Error('Draw at least one placement zone on the canvas.');
  state.products.forEach((p,i) => {
    if (p.models.length === 0) throw new Error(`Product "${p.name}" has no 3D model files.`);
  });

  return {
    background_path: state.backgroundPath,
    products: state.products.map(p => ({
      class_id:    p.classId,
      class_name:  p.name,
      model_paths: p.models.map(m => m.path),
      render_size: 512,
      weight:      1.0,
    })),
    placement_zones: state.zones.map(z => ({
      points: z.points.map(pt => [
        Math.round(pt.x / state.canvasScale),
        Math.round(pt.y / state.canvasScale),
      ])
    })),
    fixed_scale: parseFloat(document.getElementById('fixed-scale').value) || 1.0,
    rotation: {
      x_range: [parseFloat(document.getElementById('rx-min').value), parseFloat(document.getElementById('rx-max').value)],
      y_range: [parseFloat(document.getElementById('ry-min').value), parseFloat(document.getElementById('ry-max').value)],
      z_range: [parseFloat(document.getElementById('rz-min').value), parseFloat(document.getElementById('rz-max').value)],
    },
    lighting: {
      ambient_intensity:      parseFloat(document.getElementById('ambient').value),
      directional_intensity:  parseFloat(document.getElementById('dir-intensity').value),
      directional_angle_range:[-30, 30],
      intensity_variation:    parseFloat(document.getElementById('light-vary').value),
    },
    augmentation: {
      motion_blur_prob:       document.getElementById('aug-blur').checked  ? parseFloat(document.getElementById('aug-blur-prob').value)   : 0,
      brightness_prob:        document.getElementById('aug-bright').checked ? parseFloat(document.getElementById('aug-bright-prob').value) : 0,
      noise_prob:             document.getElementById('aug-noise').checked  ? parseFloat(document.getElementById('aug-noise-prob').value)  : 0,
      hue_shift_prob:         document.getElementById('aug-hue').checked    ? parseFloat(document.getElementById('aug-hue-prob').value)    : 0,
      jpeg_artifact_prob:     document.getElementById('aug-jpeg').checked   ? parseFloat(document.getElementById('aug-jpeg-prob').value)   : 0,
      motion_blur_kernel_range:[5,15],
      brightness_range:[0.7,1.3],
      noise_std_range:[5,25],
      hue_shift_range:[-10,10],
      jpeg_quality_range:[60,90],
    },
    num_images:             parseInt(document.getElementById('num-images').value),
    output_width:           parseInt(document.getElementById('out-width').value),
    output_height:          parseInt(document.getElementById('out-height').value),
    instances_min:          parseInt(document.getElementById('inst-min').value),
    instances_max:          parseInt(document.getElementById('inst-max').value),
    negative_sample_ratio:  parseFloat(document.getElementById('neg-ratio').value),
    min_visible_fraction:   0.7,
    edge_feather_px:        3,
    iou_overlap_limit:      0.3,
    dataset_name:           document.getElementById('dataset-name').value.trim() || 'Synthetic Dataset',
    dataset_description:    'Auto-generated synthetic dataset',
  };
}

// ── Preview ───────────────────────────────────
async function runPreview() {
  let cfg;
  try { cfg = buildConfig(); } catch(e) { alert(e.message); return; }
  cfg.count = 6;
  document.getElementById('preview-section').classList.remove('hidden');
  document.getElementById('preview-grid').innerHTML = '<p class="hint" style="padding:8px">Rendering preview… this may take a moment.</p>';
  try {
    const r = await fetch('/synthetic/preview', {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(cfg)
    });
    const d = await r.json();
    if (!r.ok) { alert('Preview error: ' + d.error); return; }
    const grid = document.getElementById('preview-grid');
    grid.innerHTML = '';
    d.previews.forEach(b64 => {
      const img = document.createElement('img');
      img.src = 'data:image/jpeg;base64,' + b64;
      img.className = 'preview-img';
      img.title = 'Click to open full size';
      img.onclick = () => window.open(img.src, '_blank');
      grid.appendChild(img);
    });
  } catch(e) { alert('Preview failed: ' + e); }
}

// ── Generation ────────────────────────────────
async function startGeneration() {
  let cfg;
  try { cfg = buildConfig(); } catch(e) { alert(e.message); return; }
  try {
    const r = await fetch('/synthetic/generate', {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(cfg)
    });
    const d = await r.json();
    if (!r.ok) { alert('Error: ' + d.error); return; }
    state.activeJobId = d.id;
    document.getElementById('progress-area').classList.remove('hidden');
    document.getElementById('job-result').classList.add('hidden');
    document.getElementById('jobs-section').classList.remove('hidden');
    pollJob(d.id, d.total);
  } catch(e) { alert('Failed to start: ' + e); }
}

function pollJob(jobId, total) {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    try {
      const r = await fetch(`/synthetic/generate/${jobId}/status`);
      const d = await r.json();
      if (!r.ok) return;
      const pct = total > 0 ? Math.round((d.generated / total) * 100) : 0;
      document.getElementById('progress-bar').style.width = pct + '%';
      document.getElementById('progress-label').textContent =
        `${d.generated} / ${total} images  (${pct}%)  — ${d.status}`;
      renderJobCard(d);
      if (['completed','failed','stopped'].includes(d.status)) {
        clearInterval(state.pollTimer);
        showJobResult(d);
      }
    } catch(e) {}
  }, 1500);
}

function showJobResult(job) {
  const el = document.getElementById('job-result');
  el.classList.remove('hidden','success','failure');
  if (job.status === 'completed') {
    el.classList.add('success');
    el.innerHTML = `✅ <b>Generation complete!</b><br>
      Generated: ${job.generated} images<br>
      Failed: ${job.failed}<br>
      ${job.dataset_id ? `Dataset ID: <code>${job.dataset_id}</code>` : ''}`;
  } else {
    el.classList.add('failure');
    el.innerHTML = `❌ <b>Generation ${job.status}</b><br>${job.error || ''}`;
  }
}

async function stopGeneration() {
  if (!state.activeJobId) return;
  await fetch(`/synthetic/generate/${state.activeJobId}/stop`, {method:'POST'});
  document.getElementById('progress-label').textContent = 'Stopping…';
}

function renderJobCard(job) {
  let card = document.getElementById('job-card-' + job.id);
  if (!card) {
    card = document.createElement('div');
    card.className = 'job-card';
    card.id = 'job-card-' + job.id;
    document.getElementById('jobs-list').prepend(card);
  }
  const pct = job.total > 0 ? Math.round((job.generated / job.total) * 100) : 0;
  card.innerHTML = `
    <div class="job-info">
      <div class="job-id">${job.id}</div>
      <div class="job-status ${job.status}">${job.status.toUpperCase()}  ${pct}%</div>
      <div class="hint">${job.generated}/${job.total} images • ${job.config?.dataset_name||''}</div>
    </div>`;
}

// ── Init: add one product by default ─────────
window.addEventListener('DOMContentLoaded', () => {
  addProductClass();
  setTool('select');
});
