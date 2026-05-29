document.addEventListener('DOMContentLoaded', () => {
  fetchCameras();
  setInterval(fetchCameras, 4000);
});

async function fetchCameras() {
  try {
    const camRes = await fetch('/cameras');
    const camData = await camRes.json();

    const listEl = document.getElementById('cameras-list');
    listEl.innerHTML = '';

    if (!camData.cameras || Object.keys(camData.cameras).length === 0) {
      listEl.innerHTML = '<div class="empty-state">No cameras configured. Add one above.</div>';
      return;
    }

    for (const id in camData.cameras) {
      const cam = camData.cameras[id];

      const item = document.createElement('div');
      item.className = 'camera-item';
      item.innerHTML = `
        <div class="camera-info">
          <span class="camera-icon">📹</span>
          <div class="camera-details">
            <span class="camera-name">${cam.name}</span>
            <span class="camera-rtsp">${cam.rtsp}</span>
          </div>
        </div>
        <div class="camera-actions">
          <button class="action-btn danger" onclick="deleteCamera('${id}')">Delete</button>
          <button class="action-btn secondary" onclick="viewRaw('${id}')">View Raw</button>
        </div>
      `;
      listEl.appendChild(item);
    }
  } catch (e) {
    console.error('Error fetching cameras:', e);
  }
}

async function addCamera() {
  const name = document.getElementById('cam-name').value;
  const rtsp = document.getElementById('cam-rtsp').value;

  if (!name || !rtsp) {
    alert('Please fill in Camera Name and RTSP URL');
    return;
  }

  try {
    await fetch('/cameras', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, rtsp })
    });

    document.getElementById('cam-name').value = '';
    document.getElementById('cam-rtsp').value = '';

    fetchCameras();
  } catch (e) {
    alert('Failed to add camera');
  }
}

function viewRaw(cam_id) {
  const viewer = document.getElementById('live-view-section');
  const stream = document.getElementById('stream');
  viewer.style.display = 'block';
  stream.src = `/cameras/${cam_id}/mjpeg`;
  viewer.scrollIntoView({ behavior: 'smooth' });
}

function closeViewer() {
  const viewer = document.getElementById('live-view-section');
  const stream = document.getElementById('stream');
  stream.src = '';
  viewer.style.display = 'none';
}

async function deleteCamera(cam_id) {
  if (!confirm('Delete this camera?')) {
    return;
  }

  try {
    const response = await fetch(`/cameras/${cam_id}`, {
      method: 'DELETE'
    });

    if (!response.ok) {
      alert('Failed to delete camera');
      return;
    }

    closeViewer();
    fetchCameras();
  } catch (e) {
    console.error('Error deleting camera:', e);
    alert('Failed to delete camera');
  }
}
