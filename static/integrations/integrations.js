let currentConfig = {
  enabled: false,
  api_url: '',
  id: '',
  include_image: false,
  filter_classes: '',
  filter_events: ['in', 'out', 'unknown']
};

let autoSaveTimeout = null;

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('test-integration-btn').addEventListener('click', testIntegration);
  
  // Set up auto-save listeners
  const inputs = document.querySelectorAll('#integration-form input');
  inputs.forEach(input => {
    if (input.type === 'checkbox') {
      input.addEventListener('change', autoSaveIntegration);
    } else {
      input.addEventListener('blur', autoSaveIntegration);
      // Optional: Add debounce for typing
      input.addEventListener('input', () => {
        clearTimeout(autoSaveTimeout);
        autoSaveTimeout = setTimeout(autoSaveIntegration, 1000);
      });
    }
  });

  loadIntegration();
  fetchLogs();
  // Poll for logs every 3 seconds
  setInterval(fetchLogs, 3000);
});

async function loadIntegration() {
  try {
    const res = await fetch('/integrations/config');
    currentConfig = await res.json();
    renderForm();
    renderStatus();
  } catch (err) {
    notify('Failed to load integration.');
  }
}

function renderForm() {
  document.getElementById('integration-enabled').checked = !!currentConfig.enabled;
  document.getElementById('api-url').value = currentConfig.api_url || '';
  document.getElementById('integration-id').value = currentConfig.id || '';
  document.getElementById('include-image').checked = !!currentConfig.include_image;
  document.getElementById('filter-classes').value = currentConfig.filter_classes || '';
  
  const events = currentConfig.filter_events || [];
  const eventCheckboxes = document.querySelectorAll('input[name="filter-events"]');
  eventCheckboxes.forEach(cb => {
    cb.checked = events.includes(cb.value);
  });
}

function readForm() {
  const eventCheckboxes = document.querySelectorAll('input[name="filter-events"]:checked');
  const filter_events = Array.from(eventCheckboxes).map(cb => cb.value);

  return {
    enabled: document.getElementById('integration-enabled').checked,
    api_url: document.getElementById('api-url').value.trim(),
    id: document.getElementById('integration-id').value.trim(),
    include_image: document.getElementById('include-image').checked,
    filter_classes: document.getElementById('filter-classes').value.trim(),
    filter_events: filter_events,
  };
}

function renderStatus() {
  const status = document.getElementById('integration-status');
  const enabled = currentConfig.enabled && currentConfig.api_url;
  status.innerHTML = enabled
    ? `<strong>Active</strong><br>Events will be posted to ${escapeHtml(currentConfig.api_url)}<br>ID: ${escapeHtml(currentConfig.id || '-')}`
    : '<strong>Inactive</strong><br>No detected object events will be sent to an external API.';
}

async function autoSaveIntegration() {
  clearTimeout(autoSaveTimeout);
  const payload = readForm();
  try {
    const res = await fetch('/integrations/config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'save failed');
    currentConfig = data;
    renderStatus();
    
    // Show "Saved automatically" message
    const msg = document.getElementById('auto-save-msg');
    msg.classList.add('visible');
    setTimeout(() => msg.classList.remove('visible'), 3000);
  } catch (err) {
    notify(err.message || 'Failed to save integration.');
  }
}

async function testIntegration() {
  const payload = readForm();
  const result = document.getElementById('test-result');
  result.hidden = false;
  result.className = 'test-result';
  result.textContent = 'Testing connection...';

  try {
    const res = await fetch('/integrations/test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    result.classList.add(data.ok ? 'ok' : 'fail');
    result.textContent = data.ok
      ? `Connection OK. Status ${data.status_code}.`
      : `Connection failed. ${data.error || `Status ${data.status_code}: ${data.response || ''}`}`;
      
    // Refresh logs after a test
    setTimeout(fetchLogs, 500);
  } catch (err) {
    result.classList.add('fail');
    result.textContent = err.message || 'Connection test failed.';
  }
}

async function fetchLogs() {
  try {
    const res = await fetch('/integrations/logs');
    if (!res.ok) return;
    const logs = await res.json();
    const tbody = document.getElementById('logs-tbody');
    
    if (logs.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:var(--text-secondary);">No recent events</td></tr>';
      return;
    }
    
    tbody.innerHTML = '';
    logs.forEach(log => {
      const tr = document.createElement('tr');
      const timeStr = new Date(log.timestamp).toLocaleTimeString();
      const statusClass = log.status === 'success' ? 'status-success' : 'status-error';
      
      tr.innerHTML = `
        <td>${escapeHtml(timeStr)}</td>
        <td>${escapeHtml(log.class_name)}</td>
        <td>${escapeHtml(log.event_type)}</td>
        <td class="${statusClass}">${escapeHtml(log.status)} <span style="color:var(--text-secondary);font-size:0.75rem;margin-left:5px;">${escapeHtml(log.message)}</span></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    // silently fail for polling
  }
}

function notify(message) {
  const el = document.getElementById('integration-notification');
  el.textContent = message;
  el.hidden = false;
  setTimeout(() => { el.hidden = true; }, 3000);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[ch]));
}
