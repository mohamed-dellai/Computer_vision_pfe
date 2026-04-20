from flask import Blueprint, jsonify, request, Response
import uuid
import cv2
import os
import threading
import persistence
from services.video_recorder import RawVideoRecorder

DEFAULT_RECORDINGS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data', 'recordings')
)


def create_cameras_blueprint(cameras, jobs, jobs_lock):
    bp = Blueprint('cameras', __name__, url_prefix='/cameras')

    @bp.route('', methods=['GET'])
    def list_cameras():
        return jsonify({'cameras': cameras})

    @bp.route('', methods=['POST'])
    def add_camera():
        data = request.json or {}
        name = data.get('name') or f'cam-{len(cameras)+1}'
        rtsp = data.get('rtsp')
        if not rtsp:
            return jsonify({'error': 'rtsp field required'}), 400
        cid = str(uuid.uuid4())
        cameras[cid] = {'name': name, 'rtsp': rtsp}
        persistence.save_state(cameras, jobs, jobs_lock)
        return jsonify({'camera_id': cid}), 201

    @bp.route('/<camera_id>/snapshot', methods=['GET'])
    def camera_snapshot(camera_id):
        cam = cameras.get(camera_id)
        if not cam:
            return jsonify({'error': 'unknown camera'}), 404
        rtsp_url = cam['rtsp']
        
        # Get RTSP configuration from request parameters
        width = request.args.get('width', type=int)
        height = request.args.get('height', type=int)
        
        rtsp_kwargs = {}
        if width: rtsp_kwargs['width'] = width
        if height: rtsp_kwargs['height'] = height
        
        # Use a temporary stream to capture a single frame
        from rtsp import RTSPVideoStream
        stream = RTSPVideoStream(rtsp_url, **rtsp_kwargs)
        stream.start()
        
        try:
            # Wait up to 5 seconds for a frame
            frame, _ = stream.read(timeout=5.0)
                
            if frame is None:
                print(f"Snapshot failed: timeout or no frame received from {rtsp_url}")
                return jsonify({'error': 'failed to grab frame (timeout or connection failed)'}), 500
                
            ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                print(f"Snapshot failed: could not encode frame from {rtsp_url} to JPEG")
                return jsonify({'error': 'failed to encode frame'}), 500
                
            return Response(buf.tobytes(), mimetype='image/jpeg')
        except Exception as e:
            print(f"Snapshot exception: {e}")
            return jsonify({'error': str(e)}), 500
        finally:
            stream.stop()

    @bp.route('/<camera_id>/mjpeg', methods=['GET'])
    def camera_mjpeg(camera_id):
        cam = cameras.get(camera_id)
        if not cam:
            return 'unknown camera', 404
        rtsp_url = cam['rtsp']

        width = request.args.get('width', type=int)
        height = request.args.get('height', type=int)
        if (width and not height) or (height and not width):
            return jsonify({'error': 'width and height must be provided together'}), 400
        rtsp_kwargs = {}
        if width and height:
            rtsp_kwargs['width'] = width
            rtsp_kwargs['height'] = height
        
        from rtsp import RTSPVideoStream
        
        def generator():
            stream = RTSPVideoStream(rtsp_url, **rtsp_kwargs)
            stream.start()
            last_frame_id = -1
            try:
                while True:
                    frame, last_frame_id = stream.read(timeout=0.5, last_frame_id=last_frame_id)
                    if frame is None:
                        if not stream.is_alive():
                            break
                        continue
                    
                    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if not ok:
                        continue
                        
                    frame_bytes = buf.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            finally:
                stream.stop()

        return Response(generator(), mimetype='multipart/x-mixed-replace; boundary=frame')



    return bp
