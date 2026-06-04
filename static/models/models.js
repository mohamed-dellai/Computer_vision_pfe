document.addEventListener('DOMContentLoaded', () => {
  fetchModels();

  const fileInput = document.getElementById('model-file');
  const fileName = document.getElementById('file-name');

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      fileName.textContent = fileInput.files[0].name;
    } else {
      fileName.textContent = '';
    }
  });
});

async function fetchModels(){
  try {
    let r = await fetch('/models');
    let j = await r.json();
    let s = document.getElementById('models-list');
    let optSelect = document.getElementById('opt-source-select');
    s.innerHTML='';
    if (optSelect) {
      optSelect.innerHTML = '';
    }
    if (j.models.length === 0) {
      s.innerHTML = '<li class="empty-state">No models found. Upload one to get started.</li>';
      if (optSelect) {
        optSelect.innerHTML = '<option disabled selected>No models available</option>';
      }
      return;
    }
    j.models.forEach(m=>{
      const ext = modelExtension(m);
      let li = document.createElement('li');
      li.className = 'model-item';
      li.innerHTML = `
        <span class="model-icon">🧠</span>
        <span class="model-name">${m}</span>
        <span class="model-tag">${ext || 'model'}</span>
      `;
      s.appendChild(li);
      if (optSelect && ext === 'pt') {
        const option = document.createElement('option');
        option.value = m;
        option.text = m;
        optSelect.appendChild(option);
      }
    });
    if (optSelect && optSelect.options.length === 0) {
      optSelect.innerHTML = '<option disabled selected>No .pt models available</option>';
    }
  } catch (e) {
    console.error("Error fetching models:", e);
  }
}

function modelExtension(filename) {
  const match = String(filename || '').toLowerCase().match(/\.([^.]+)$/);
  return match ? match[1] : '';
}

async function uploadModel(){
  const file = document.getElementById('model-file').files[0];
  if(!file){ 
    alert('Please select a file first'); 
    return;
  }
  
  const formData = new FormData();
  formData.append('file', file);
  
  const status = document.getElementById('upload-status');
  status.textContent = 'Uploading...';
  status.className = 'status-text loading';
  
  try {
    const r = await fetch('/models/upload', {method:'POST', body:formData});
    const j = await r.json();
    
    if(r.ok){
      status.textContent = `Uploaded: ${j.filename}`;
      status.className = 'status-text success';
      document.getElementById('file-name').textContent = '';
      document.getElementById('model-file').value = '';
      setTimeout(fetchModels, 500);
    }else{
      status.textContent = `Error: ${j.error || 'Unknown error'}`;
      status.className = 'status-text error';
    }
  } catch (e) {
    status.textContent = 'Network error. Please try again.';
    status.className = 'status-text error';
  }
}

async function optimizeModel(){
  const source = document.getElementById('opt-source-select').value;
  const target = document.getElementById('opt-target-select').value;
  const imgsz = parseInt(document.getElementById('opt-imgsz').value, 10) || 640;
  const dynamic = document.getElementById('opt-dynamic').checked;
  const half = document.getElementById('opt-half').checked;
  const status = document.getElementById('opt-status');
  
  if(!source){ 
    alert('Please select a source model'); 
    return; 
  }
  
  status.textContent = 'Optimizing...';
  status.className = 'status-text loading';
  
  try {
    const r = await fetch('/models/optimize', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ source, target, imgsz, dynamic, half })
    });
    const j = await r.json();
    if(r.ok){
      status.textContent = `Exported: ${j.filename} (${j.format_label || j.format})`;
      status.className = 'status-text success';
      setTimeout(fetchModels, 500);
    } else {
      status.textContent = `Error: ${j.error || 'Export failed'}`;
      status.className = 'status-text error';
    }
  } catch (e) {
    status.textContent = 'Network error. Please try again.';
    status.className = 'status-text error';
  }
}
