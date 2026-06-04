from datetime import datetime

import requests
from flask import Blueprint, jsonify, request

import integration_config


def create_integrations_blueprint():
    bp = Blueprint('integrations', __name__, url_prefix='/integrations')

    @bp.route('/config', methods=['GET'])
    def get_config():
        return jsonify(integration_config.load_integration())

    @bp.route('/config', methods=['PUT'])
    def update_config():
        data = request.json or {}
        api_url = str(data.get('api_url') or '').strip()
        external_id = str(data.get('id') or '').strip()
        enabled = bool(data.get('enabled'))
        include_image = bool(data.get('include_image'))
        filter_classes = str(data.get('filter_classes') or '').strip()
        filter_events = data.get('filter_events', ['in', 'out', 'unknown'])

        if enabled and not api_url:
            return jsonify({'error': 'api_url is required when integration is enabled'}), 400

        config = integration_config.save_integration({
            'enabled': enabled,
            'api_url': api_url,
            'id': external_id,
            'include_image': include_image,
            'filter_classes': filter_classes,
            'filter_events': filter_events
        })
        return jsonify(config)

    @bp.route('/logs', methods=['GET'])
    def get_logs():
        return jsonify(integration_config.get_logs())

    @bp.route('/test', methods=['POST'])
    def test_config():
        data = request.json or integration_config.load_integration()
        api_url = str(data.get('api_url') or '').strip()
        external_id = str(data.get('id') or '').strip()
        if not api_url:
            return jsonify({'ok': False, 'error': 'api_url is required'}), 400

        payload = {
            'shopId': external_id,
            'name': 'connection-test',
            'class_name': 'test-object',
            'object_id': 999,
            'detection_time': datetime.utcnow().isoformat(),
            'location': '[0, 0, 100, 100]',
            'inOrOut': 'in',
            'confidence': 0.99,
            'test': True,
        }

        try:
            print(f"Sending test payload to {api_url}: {payload}")
            resp = requests.post(api_url, data=payload, timeout=5.0)
            body = resp.text[:1000] if resp.text else ''
            
            status_text = 'success' if resp.ok else 'error'
            integration_config.add_log('test-object', 'in', status_text, f"Test: {resp.status_code}")
            
            return jsonify({
                'ok': resp.ok,
                'status_code': resp.status_code,
                'response': body,
            }), 200 if resp.ok else 502
        except Exception as exc:
            integration_config.add_log('test-object', 'in', 'error', f"Test failed: {str(exc)}")
            return jsonify({'ok': False, 'error': str(exc)}), 502

    return bp
