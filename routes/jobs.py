from flask import Blueprint, jsonify, request, Response
import os
import uuid
import persistence
from worker import TrackingWorker
from config import RTSPConfig, TrackerConfig, JobConfig

def create_jobs_blueprint(cameras, jobs, jobs_lock, available_models_func):
    bp = Blueprint('jobs', __name__, url_prefix='/jobs')

    @bp.route('', methods=['GET'])
    def list_jobs():
        with jobs_lock:
            summary = {jid: {'camera_id': j['camera_id'], 'model': j['model'], 'status': j['worker'].status} for jid, j in jobs.items()}
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
            
            config = JobConfig(
                camera_id=camera_id,
                model_name=model_name,
                rtsp_url=cameras[camera_id]['rtsp'],
                conf=float(data.get('conf', 0.25)),
                iou=float(data.get('iou', 0.7)),
                rtsp_config=rtsp_config,
                tracker_config=tracker_config,
                line_coords=data.get('line_coords')
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
        return jsonify({'camera_id': j['camera_id'], 'model': j['model'], 'status': j['worker'].status})

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
