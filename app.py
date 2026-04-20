from flask import Flask
from worker import TrackingWorker
from config import RTSPConfig, TrackerConfig, JobConfig
import os
import sys
import threading
import persistence

print(f"==================================================")
print(f"STARTING SERVER WITH PYTHON EXECUTABLE:")
print(f"-> {sys.executable}")
print(f"==================================================")

# Import blueprint creators
from routes.cameras import create_cameras_blueprint
from routes.jobs import create_jobs_blueprint
from routes.models import create_models_blueprint
from routes.trackers import create_trackers_blueprint
from routes.training import create_training_blueprint
from routes.ui import create_ui_blueprint
from services.autodistill_grounded_sam2 import GroundedSam2AutodistillService
from services.training_runner import TrainingRunner
from services.training_store import TrainingStore

app = Flask(__name__)

# --- Configuration & State ---
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

TRACKERS_DIR = os.path.join(os.path.dirname(__file__), 'trackers')
if not os.path.exists(TRACKERS_DIR):
    os.makedirs(TRACKERS_DIR)

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
TRAINING_DIR = os.path.join(os.path.dirname(__file__), 'data', 'training')

ALLOWED_EXTENSIONS = {'pt', 'onnx', 'engine'}

os.makedirs(TRAINING_DIR, exist_ok=True)

# In-memory stores
cameras = {}
jobs = {}
jobs_lock = threading.Lock()

# --- Helper Functions ---
def available_models():
    files = []
    try:
        for f in os.listdir(MODELS_DIR):
            if any(f.endswith('.' + ext) for ext in ALLOWED_EXTENSIONS):
                files.append(f)
    except Exception:
        pass
    return files

# --- Initialization ---
loaded_cameras, loaded_jobs_config = persistence.load_state()
cameras.update(loaded_cameras)

training_store = TrainingStore(TRAINING_DIR)
training_runner = TrainingRunner(TRAINING_DIR, MODELS_DIR, training_store)
training_runner.initialize()
autodistill_service = GroundedSam2AutodistillService()

# Restart jobs
for jid, j_config in loaded_jobs_config.items():
    try:
        rtsp_config = RTSPConfig.from_dict(j_config)
        tracker_config = TrackerConfig.from_dict(j_config)
        
        config = JobConfig(
            camera_id=j_config['camera_id'],
            model_name=j_config['model'],
            rtsp_url=cameras[j_config['camera_id']]['rtsp'],
            conf=j_config.get('conf', 0.25),
            iou=j_config.get('iou', 0.7),
            rtsp_config=rtsp_config,
            tracker_config=tracker_config,
            line_coords=j_config.get('line_coords')
        )
        
        worker = TrackingWorker(config)
        worker.start()
        
        j_config['worker'] = worker
        jobs[jid] = j_config
        print(f"Restarted job {jid} for camera {j_config['camera_id']}")
    except Exception as e:
        print(f"Failed to restart job {jid}: {e}")

# --- Register Blueprints ---
app.register_blueprint(create_cameras_blueprint(cameras, jobs, jobs_lock))
app.register_blueprint(create_jobs_blueprint(cameras, jobs, jobs_lock, available_models))
app.register_blueprint(create_models_blueprint(MODELS_DIR, ALLOWED_EXTENSIONS, available_models))
app.register_blueprint(create_trackers_blueprint(TRACKERS_DIR))
app.register_blueprint(create_training_blueprint(training_store, training_runner, autodistill_service, cameras))
app.register_blueprint(create_ui_blueprint(STATIC_DIR))

# Startup visibility for debugging route registration issues.
print("Registered routes:")
for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
    print(f"{rule.rule} -> {','.join(sorted(rule.methods - {'HEAD', 'OPTIONS'}))}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
