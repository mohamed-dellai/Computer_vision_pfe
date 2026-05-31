import json
import os
import threading
from typing import Dict, Optional, List
from datetime import datetime

_CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'data', 'integration.json')
_lock = threading.Lock()

_recent_logs = []
_MAX_LOGS = 50

def _empty_config() -> Dict[str, object]:
    return {
        'enabled': False,
        'api_url': '',
        'id': '',
        'include_image': False,
        'filter_classes': '',
        'filter_events': ['in', 'out', 'unknown']
    }

def load_integration() -> Dict[str, object]:
    if not os.path.exists(_CONFIG_FILE):
        return _empty_config()
    try:
        with _lock:
            with open(_CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        config = _empty_config()
        
        # Ensure filter_events is a list
        evs = data.get('filter_events')
        if not isinstance(evs, list):
            evs = ['in', 'out', 'unknown']

        config.update({
            'enabled': bool(data.get('enabled')),
            'api_url': str(data.get('api_url') or ''),
            'id': str(data.get('id') or ''),
            'include_image': bool(data.get('include_image')),
            'filter_classes': str(data.get('filter_classes') or ''),
            'filter_events': evs
        })
        return config
    except Exception:
        return _empty_config()

def save_integration(config: Dict[str, object]) -> Dict[str, object]:
    evs = config.get('filter_events')
    if not isinstance(evs, list):
        evs = ['in', 'out', 'unknown']

    normalized = {
        'enabled': bool(config.get('enabled')),
        'api_url': str(config.get('api_url') or '').strip(),
        'id': str(config.get('id') or '').strip(),
        'include_image': bool(config.get('include_image')),
        'filter_classes': str(config.get('filter_classes') or ''),
        'filter_events': evs
    }
    os.makedirs(os.path.dirname(_CONFIG_FILE), exist_ok=True)
    with _lock:
        with open(_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(normalized, f, indent=2)
    return normalized

def active_integration() -> Optional[Dict[str, object]]:
    config = load_integration()
    if not config.get('enabled') or not config.get('api_url'):
        return None
    return config

def add_log(class_name: str, event_type: str, status: str, message: str) -> None:
    with _lock:
        _recent_logs.insert(0, {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'class_name': class_name,
            'event_type': event_type,
            'status': status,
            'message': message
        })
        if len(_recent_logs) > _MAX_LOGS:
            _recent_logs.pop()

def get_logs() -> List[Dict[str, object]]:
    with _lock:
        return list(_recent_logs)
