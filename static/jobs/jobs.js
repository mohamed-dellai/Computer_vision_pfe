document.addEventListener('DOMContentLoaded', () => {
  fetchCameras();
  fetchModels();
  fetchJobs();
  fetchTrackers();
  setInterval(fetchJobs, 3000);
  
  const selectedTracker = sessionStorage.getItem('selectedTracker');
  if (selectedTracker) {
    document.getElementById('tracker-select').value = selectedTracker;
    sessionStorage.removeItem('selectedTracker');
    showNotification(`Using tracker: ${selectedTracker}`, 'info');
  }

  setupLineCounting();
});

let currentLineCoords = null;
let isDrawing = false;
let startPoint = null;
let currentPoint = null;
let snapshotImage = null;

function setupLineCounting() {
  const checkbox = document.getElementById('line-counting-check');
  const modal = document.getElementById('line-modal');
  const confirmBtn = document.getElementById('confirm-line');
  const cancelBtn = document.getElementById('cancel-line');
  const canvas = document.getElementById('line-canvas');
  
  checkbox.addEventListener('change', async (e) => {
    if (e.target.checked) {
      const cameraId = document.getElementById('camera-select').value;
      if (!cameraId) {
        alert("Please select a camera first.");
        e.target.checked = false;
        return;
      }
      
      try {
        await openLineModal(cameraId);
      } catch (err) {
        alert("Failed to get camera snapshot: " + err);
        e.target.checked = false;
      }
    } else {
      currentLineCoords = null;
      document.getElementById('line-coords-display').style.display = 'none';
      document.getElementById('coords-values').textContent = 'Not set';
    }
  });

  canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    isDrawing = true;
    startPoint = {x, y};
    currentPoint = {x, y};
    redrawCanvas();
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    currentPoint = {x, y};
    redrawCanvas();
  });

  canvas.addEventListener('mouseup', () => {
    isDrawing = false;
  });

  confirmBtn.addEventListener('click', () => {
    if (startPoint && currentPoint) {
        const canvas = document.getElementById('line-canvas');
        const scaleX = snapshotImage.width / canvas.width;
        const scaleY = snapshotImage.height / canvas.height;
        
        currentLineCoords = [
            Math.round(startPoint.x * scaleX),
            Math.round(startPoint.y * scaleY),
            Math.round(currentPoint.x * scaleX),
            Math.round(currentPoint.y * scaleY)
        ];
        
        document.getElementById('coords-values').textContent = 
            `(${currentLineCoords[0]}, ${currentLineCoords[1]}) to (${currentLineCoords[2]}, ${currentLineCoords[3]})`;
        document.getElementById('line-coords-display').style.display = 'block';
        modal.style.display = 'none';
    } else {
        alert("Please draw a line first.");
    }
  });

  cancelBtn.addEventListener('click', () => {
    checkbox.checked = false;
    currentLineCoords = null;
    modal.style.display = 'none';
  });
}

async function openLineModal(cameraId) {
    const modal = document.getElementById('line-modal');
    const canvas = document.getElementById('line-canvas');
    const ctx = canvas.getContext('2d');
    
    const rtspWidth = document.getElementById('rtsp_width').value;
    const rtspHeight = document.getElementById('rtsp_height').value;
    
    modal.style.display = 'flex';
    
    let url = `/cameras/${cameraId}/snapshot`;
    const params = new URLSearchParams();
    if (rtspWidth) params.append('width', rtspWidth);
    if (rtspHeight) params.append('height', rtspHeight);
    
    if (params.toString()) {
        url += `?${params.toString()}`;
    }

    const res = await fetch(url);
    if (!res.ok) throw new Error("Could not fetch snapshot");
    
    const blob = await res.blob();
    const img = new Image();
    img.src = URL.createObjectURL(blob);
    
    await new Promise(r => img.onload = r);
    snapshotImage = img;
    
    const maxWidth = 700;
    const maxHeight = 500;
    let width = img.width;
    let height = img.height;
    
    if (width > maxWidth) {
        height = height * (maxWidth / width);
        width = maxWidth;
    }
    if (height > maxHeight) {
        width = width * (maxHeight / height);
        height = maxHeight;
    }
    
    canvas.width = width;
    canvas.height = height;
    
    redrawCanvas();
}

function redrawCanvas() {
    const canvas = document.getElementById('line-canvas');
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
        
        ctx.fillStyle = '#ff0000';
        ctx.beginPath();
        ctx.arc(startPoint.x, startPoint.y, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(currentPoint.x, currentPoint.y, 4, 0, Math.PI * 2);
        ctx.fill();
    }
}

async function fetchCameras(){
  try {
    const r = await fetch('/cameras');
    const j = await r.json();
    const select = document.getElementById('camera-select');
    select.innerHTML = '';
    
    if (Object.keys(j.cameras).length === 0) {
      select.innerHTML = '<option disabled selected>No cameras available</option>';
      return;
    }
    
    const urlParams = new URLSearchParams(window.location.search);
    const preSelectedCam = urlParams.get('camera_id');
    
    for(const id in j.cameras){
      const cam = j.cameras[id];
      const option = document.createElement('option');
      option.value = id;
      option.text = cam.name + ' (' + cam.rtsp + ')';
      if (id === preSelectedCam) option.selected = true;
      select.appendChild(option);
    }
  } catch (e) {
    console.error("Error fetching cameras:", e);
  }
}

async function fetchModels(){
  try {
    const r = await fetch('/models');
    const j = await r.json();
    const select = document.getElementById('model-select');
    select.innerHTML = '';
    
    if (j.models.length === 0) {
      select.innerHTML = '<option disabled selected>No models available</option>';
      return;
    }
    
    j.models.forEach(m => {
      const option = document.createElement('option');
      option.value = m;
      option.text = m;
      select.appendChild(option);
    });
  } catch (e) {
    console.error("Error fetching models:", e);
  }
}

async function fetchTrackers(){
  try {
    const r = await fetch('/trackers');
    const trackers = await r.json();
    const select = document.getElementById('tracker-select');
    
    select.innerHTML = '';
    
    const defaultOptions = [
      {value: 'bytetrack.yaml', text: 'ByteTrack (Default)'},
      {value: 'botsort.yaml', text: 'BoTSORT (Default)'}
    ];
    
    defaultOptions.forEach(opt => {
      const option = document.createElement('option');
      option.value = opt.value;
      option.text = opt.text;
      select.appendChild(option);
    });
    
    if (trackers.length > 0) {
      const separator = document.createElement('option');
      separator.disabled = true;
      separator.text = '────────── Saved Trackers ──────────';
      select.appendChild(separator);
      
      trackers.forEach(tracker => {
        const option = document.createElement('option');
        option.value = tracker.filename;
        option.text = tracker.name;
        select.appendChild(option);
      });
    }
    
  } catch (e) {
    console.error("Error fetching trackers:", e);
  }
}

async function fetchJobs(){
  try {
    const r = await fetch('/jobs');
    const j = await r.json();
    const d = document.getElementById('jobs-list');
    d.innerHTML = '';
    
    if (Object.keys(j).length === 0) {
      d.innerHTML = '<div class="empty-state">No active jobs running.</div>';
      return;
    }
    
    for(const id in j){
      const job = j[id];
      const div = document.createElement('div');
      div.className = 'job-item';
      
      const statusClass = job.status === 'running' ? 'status-running' : 'status-stopped';
      
      div.innerHTML = `
        <div class="job-header">
          <span class="job-title">Job: ${id.substring(0,8)}...</span>
          <span class="job-status ${statusClass}">${job.status}</span>
        </div>
        <div class="job-details">
          <div class="detail-row">
            <span class="label">Camera:</span> <span class="value">${job.camera_id}</span>
          </div>
          <div class="detail-row">
            <span class="label">Model:</span> <span class="value">${job.model}</span>
          </div>
        </div>
        <div class="job-actions">
          <button class="action-btn view-btn" onclick="viewAI('${id}')">View Stream</button>
          <button class="action-btn stop-btn" onclick="stopJob('${id}')">Stop Job</button>
        </div>
      `;
      d.appendChild(div);
    }
  } catch (e) {
    console.error("Error fetching jobs:", e);
  }
}

async function startJob(){
  const camera_id = document.getElementById('camera-select').value;
  const model = document.getElementById('model-select').value;
  
  if (!camera_id || !model) {
    alert("Please select both a camera and a model.");
    return;
  }
  
  const conf = parseFloat(document.getElementById('conf').value) || 0.25;
  const iou = parseFloat(document.getElementById('iou').value) || 0.7;
  
  const tracker_file = document.getElementById('tracker-select').value;
  const rawRtspWidth = document.getElementById('rtsp_width').value.trim();
  const rawRtspHeight = document.getElementById('rtsp_height').value.trim();
  const rtsp_width = rawRtspWidth ? parseInt(rawRtspWidth) : null;
  const rtsp_height = rawRtspHeight ? parseInt(rawRtspHeight) : null;
  const rtsp_fps = parseInt(document.getElementById('rtsp_fps').value) || 15;
  const rtsp_buffer_size = parseInt(document.getElementById('rtsp_buffer_size').value) || 1;
  const rtsp_reconnect_delay = parseFloat(document.getElementById('rtsp_reconnect_delay').value) || 3.0;
  const rtsp_read_timeout = parseFloat(document.getElementById('rtsp_read_timeout').value) || 5.0;

  const shop_id_val = document.getElementById('shop-id').value.trim();
  const shop_id = shop_id_val || null;

  try {
    await fetch('/jobs/start',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        camera_id, model, conf, iou,
        rtsp_width, rtsp_height, rtsp_fps, rtsp_buffer_size,
        rtsp_reconnect_delay, rtsp_read_timeout,
        tracker_file,
        line_coords: currentLineCoords,
        shop_id
      })
    });
    fetchJobs();
    if (shop_id) localStorage.setItem('shop_id', shop_id);
    
    document.querySelector('.jobs-list-card').scrollIntoView({ behavior: 'smooth' });
    
  } catch (e) {
    alert("Failed to start job: " + e);
  }
}

async function stopJob(job_id){
  if(!confirm("Are you sure you want to stop this job?")) return;
  
  try {
    await fetch('/jobs/stop',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({job_id})
    });
    
    const stream = document.getElementById('stream');
    if (stream.src.includes(job_id)) {
      closeViewer();
    }
    
    fetchJobs();
  } catch (e) {
    alert("Failed to stop job");
  }
}

function viewAI(job_id){
  const viewer = document.getElementById('live-view-section');
  const stream = document.getElementById('stream');
  viewer.style.display = 'block';
  stream.src = `/jobs/${job_id}/mjpeg`;
  viewer.scrollIntoView({ behavior: 'smooth' });
}

function closeViewer() {
  const viewer = document.getElementById('live-view-section');
  const stream = document.getElementById('stream');
  stream.src = '';
  viewer.style.display = 'none';
}
