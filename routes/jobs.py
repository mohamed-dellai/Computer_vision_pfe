from flask import Blueprint, jsonify, request, Response
import os
import uuid
import json
import threading
from datetime import datetime
import persistence
from worker import TrackingWorker
from config import RTSPConfig, TrackerConfig, JobConfig

_PRESETS_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'coord_presets.json')
_presets_lock = threading.Lock()


def _load_presets():
    if not os.path.exists(_PRESETS_FILE):
        return []
    try:
        with open(_PRESETS_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return []


def _save_presets(presets):
    os.makedirs(os.path.dirname(_PRESETS_FILE), exist_ok=True)
    with open(_PRESETS_FILE, 'w') as f:
        json.dump(presets, f, indent=2)


def create_jobs_blueprint(cameras, jobs, jobs_lock, available_models_func):
    bp = Blueprint('jobs', __name__, url_prefix='/jobs')

    # ── Presets ────────────────────────────────────────────────────────────────

    @bp.route('/presets', methods=['GET'])
    def list_presets():
        with _presets_lock:
            return jsonify(_load_presets())

    @bp.route('/presets', methods=['POST'])
    def create_preset():
        data = request.json or {}
        if not data.get('name', '').strip():
            return jsonify({'error': 'name is required'}), 400
        preset = {
            'id': str(uuid.uuid4()),
            'name': data['name'].strip(),
            'camera_id': data.get('camera_id') or None,
            'rtsp_width': data.get('rtsp_width') or None,
            'rtsp_height': data.get('rtsp_height') or None,
            'crop_coords': data.get('crop_coords') or None,
            'line_coords': data.get('line_coords') or None,
            'counting_method': data.get('counting_method') or ('zone' if data.get('zone_coords') else ('line' if data.get('line_coords') else 'none')),
            'zone_coords': data.get('zone_coords') or None,
            'zone_dwell_seconds': data.get('zone_dwell_seconds') or 3.0,
            'created_at': datetime.utcnow().isoformat(),
        }
        with _presets_lock:
            presets = _load_presets()
            presets.append(preset)
            _save_presets(presets)
        return jsonify(preset), 201

    @bp.route('/presets/<preset_id>', methods=['DELETE'])
    def delete_preset(preset_id):
        with _presets_lock:
            presets = _load_presets()
            presets = [p for p in presets if p['id'] != preset_id]
            _save_presets(presets)
        return jsonify({'deleted': preset_id})

    # ── Jobs ───────────────────────────────────────────────────────────────────

    @bp.route('', methods=['GET'])
    def list_jobs():
        with jobs_lock:
            summary = {
                jid: {
                    'camera_id': j['camera_id'],
                    'model': j['model'],
                    'status': j['worker'].status,
                    'counting_method': j.get('counting_method'),
                }
                for jid, j in jobs.items()
            }
        return jsonify(summary)

    @bp.route('/start', methods=['POST'])
    def start_job():
        data = request.json or {}
        camera_id = data.get('camera_id')
        model_name = data.get('model')

        if not camera_id or camera_id not in cameras:
            return jsonify({'error': 'unknown camera_id'}), 400
        if not model_name or model_name not in available_models_func():
            return jsonify({'error': 'model not available'}), 400

        try:
            rtsp_config = RTSPConfig.from_dict(data)
            tracker_config = TrackerConfig.from_dict(data)
            counting_method = data.get('counting_method') or ('zone' if data.get('zone_coords') else ('line' if data.get('line_coords') else 'none'))
            if counting_method not in ('line', 'zone', 'none'):
                return jsonify({'error': 'counting_method must be line, zone, or none'}), 400
            if counting_method == 'line' and not data.get('line_coords'):
                return jsonify({'error': 'line_coords are required for line counting'}), 400
            if counting_method == 'zone':
                zone_coords = data.get('zone_coords') or {}
                if not isinstance(zone_coords, dict) or not zone_coords.get('in') or not zone_coords.get('out'):
                    return jsonify({'error': 'zone_coords.in and zone_coords.out are required for zone counting'}), 400

            config = JobConfig(
                camera_id=camera_id,
                model_name=model_name,
                rtsp_url=cameras[camera_id]['rtsp'],
                conf=float(data.get('conf', 0.25)),
                iou=float(data.get('iou', 0.7)),
                rtsp_config=rtsp_config,
                tracker_config=tracker_config,
                line_coords=data.get('line_coords'),
                crop_coords=data.get('crop_coords'),
                counting_method=counting_method,
                zone_coords=data.get('zone_coords'),
                zone_dwell_seconds=float(data.get('zone_dwell_seconds', 3.0))
            )
        except Exception as e:
            return jsonify({'error': f'Invalid configuration: {str(e)}'}), 400

        jid = str(uuid.uuid4())
        worker = TrackingWorker(config)
        worker.start()

        with jobs_lock:
            job_data = config.to_dict()
            job_data['worker'] = worker
            jobs[jid] = job_data

        persistence.save_state(cameras, jobs, jobs_lock)
        response = config.to_dict()
        response['job_id'] = jid
        return jsonify(response), 201

    @bp.route('/stop', methods=['POST'])
    def stop_job():
        data = request.json or {}
        jid = data.get('job_id')
        if not jid or jid not in jobs:
            return jsonify({'error': 'unknown job_id'}), 400
        with jobs_lock:
            worker = jobs[jid]['worker']
            worker.stop()
            del jobs[jid]
        persistence.save_state(cameras, jobs, jobs_lock)
        return jsonify({'stopped': jid})

    @bp.route('/<job_id>/status', methods=['GET'])
    def job_status(job_id):
        j = jobs.get(job_id)
        if not j:
            return jsonify({'error': 'unknown job'}), 404
        return jsonify({
            'camera_id': j['camera_id'],
            'model': j['model'],
            'status': j['worker'].status,
            'counting_method': j.get('counting_method'),
        })

    @bp.route('/<job_id>/mjpeg', methods=['GET'])
    def mjpeg_stream(job_id):
        j = jobs.get(job_id)
        if not j:
            return 'unknown job', 404

        def generator():
            worker = j['worker']
            import time
            last_frame_id = -1
            while worker.is_running():
                if hasattr(worker, 'read_jpeg'):
                    frame, last_frame_id = worker.read_jpeg(timeout=1.0, last_frame_id=last_frame_id)
                else:
                    frame = getattr(worker, 'latest_jpeg', None)
                    time.sleep(0.05)

                if frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            yield b''

        return Response(generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

    return bp
