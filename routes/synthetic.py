"""
routes/synthetic.py
===================
Flask blueprint: REST API for synthetic dataset generation.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import cv2
from flask import Blueprint, jsonify, request, Response
from werkzeug.utils import secure_filename

from services.synthetic_config import SyntheticConfig
from services.synthetic_generator import SyntheticDatasetGenerator

ALLOWED_MODEL_EXTS  = {".obj", ".glb", ".gltf", ".stl", ".ply", ".dae", ".mtl", ".png", ".jpg", ".jpeg"}
ALLOWED_IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_synthetic_blueprint(synthetic_dir: str, store=None):
    """
    synthetic_dir : base directory for all synthetic data
                    (backgrounds, 3D models, configs, generated datasets)
    store         : TrainingStore instance (optional; enables dataset registration)
    """
    bp = Blueprint("synthetic", __name__, url_prefix="/synthetic")

    backgrounds_dir = os.path.join(synthetic_dir, "backgrounds")
    models_dir      = os.path.join(synthetic_dir, "models")
    configs_dir     = os.path.join(synthetic_dir, "configs")
    datasets_dir    = os.path.join(synthetic_dir, "datasets")

    for d in (backgrounds_dir, models_dir, configs_dir, datasets_dir):
        os.makedirs(d, exist_ok=True)

    # In-memory job registry  {job_id -> job_dict}
    _jobs: Dict[str, Dict[str, Any]] = {}
    _jobs_lock = threading.Lock()
    _generators: Dict[str, SyntheticDatasetGenerator] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _job_response(job: Dict[str, Any]) -> Dict[str, Any]:
        """Strip non-serialisable keys before returning."""
        return {k: v for k, v in job.items() if k != "_generator"}

    # ------------------------------------------------------------------
    # Upload endpoints
    # ------------------------------------------------------------------

    @bp.route("/upload/background", methods=["POST"])
    def upload_background():
        """Upload a background image (real camera photo)."""
        f = request.files.get("file") or request.files.get("background")
        if not f or not f.filename:
            return jsonify({"error": "no file provided"}), 400
        # Use original filename for extension detection (secure_filename may strip it)
        _, ext = os.path.splitext(f.filename)
        if ext.lower() not in ALLOWED_IMAGE_EXTS:
            return jsonify({
                "error": f"Unsupported format '{ext}'. Background must be an image: "
                         f"{', '.join(sorted(ALLOWED_IMAGE_EXTS))}"
            }), 400
        safe = secure_filename(f.filename) or f"background{ext.lower()}"
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{safe}"
        dest = os.path.join(backgrounds_dir, filename)
        f.save(dest)
        return jsonify({"id": file_id, "filename": filename, "path": dest}), 201

    @bp.route("/upload/model", methods=["POST"])
    def upload_model():
        """Upload 3D model files (.obj, .glb, .mtl, .png …). Groups them in a UUID folder."""
        files = request.files.getlist("files")
        if not files:
            # Fallback for single file
            f = request.files.get("file") or request.files.get("model")
            if f:
                files = [f]
        
        if not files or not files[0].filename:
            return jsonify({"error": "no file provided"}), 400

        group_id = str(uuid.uuid4())
        group_dir = os.path.join(models_dir, group_id)
        os.makedirs(group_dir, exist_ok=True)

        saved_files = []
        for f in files:
            if not f.filename:
                continue
            original_name = f.filename
            _, ext = os.path.splitext(original_name)
            if ext.lower() not in ALLOWED_MODEL_EXTS:
                continue # skip unsupported files instead of failing entire batch

            safe = secure_filename(original_name) or f"model{ext.lower()}"
            dest = os.path.join(group_dir, safe)
            f.save(dest)
            
            # The filename we return includes the UUID folder so it can be deleted/referenced
            relative_path = f"{group_id}/{safe}"
            saved_files.append({
                "id": group_id,
                "filename": relative_path,
                "original_name": original_name,
                "path": dest,
            })

        if not saved_files:
            return jsonify({"error": "No valid supported files uploaded"}), 400

        return jsonify({"saved_files": saved_files}), 201

    @bp.route("/uploads", methods=["GET"])
    def list_uploads():
        """List uploaded backgrounds and 3D models."""
        backgrounds = [
            {"filename": fn, "path": os.path.join(backgrounds_dir, fn)}
            for fn in sorted(os.listdir(backgrounds_dir))
            if os.path.splitext(fn)[1].lower() in ALLOWED_IMAGE_EXTS
        ]
        models = []
        for root, _, files in os.walk(models_dir):
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() in ALLOWED_MODEL_EXTS:
                    rel_path = os.path.relpath(os.path.join(root, fn), models_dir)
                    # Use forward slash for consistency in URLs
                    rel_path = rel_path.replace("\\", "/")
                    models.append({
                        "filename": rel_path,
                        "path": os.path.join(root, fn)
                    })
        return jsonify({"backgrounds": backgrounds, "models": models})

    @bp.route("/uploads/background/<filename>", methods=["DELETE"])
    def delete_background(filename):
        path = os.path.join(backgrounds_dir, secure_filename(filename))
        if not os.path.exists(path):
            return jsonify({"error": "not found"}), 404
        os.remove(path)
        return jsonify({"deleted": filename})

    @bp.route("/uploads/model/<path:filepath>", methods=["DELETE"])
    def delete_model(filepath):
        # Prevent path traversal
        if ".." in filepath or filepath.startswith("/"):
            return jsonify({"error": "invalid path"}), 400
        path = os.path.join(models_dir, filepath)
        if not os.path.exists(path):
            return jsonify({"error": "not found"}), 404
        os.remove(path)
        return jsonify({"deleted": filepath})

    # ------------------------------------------------------------------
    # Config endpoints
    # ------------------------------------------------------------------

    @bp.route("/configs", methods=["GET"])
    def list_configs():
        configs = []
        for fn in sorted(os.listdir(configs_dir)):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(configs_dir, fn), "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                configs.append({"id": fn[:-5], "name": data.get("dataset_name", fn), "created_at": data.get("_created_at", "")})
            except Exception:
                pass
        return jsonify({"configs": configs})

    @bp.route("/configs", methods=["POST"])
    def save_config():
        data = request.json or {}
        try:
            cfg = SyntheticConfig.from_dict(data)
            cfg.validate()
        except (KeyError, ValueError, FileNotFoundError) as e:
            return jsonify({"error": str(e)}), 400

        config_id = str(uuid.uuid4())
        payload = cfg.to_dict()
        payload["_created_at"] = _utc_now()
        payload["_id"] = config_id

        path = os.path.join(configs_dir, f"{config_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return jsonify({"id": config_id, **payload}), 201

    @bp.route("/configs/<config_id>", methods=["GET"])
    def get_config(config_id):
        path = os.path.join(configs_dir, f"{secure_filename(config_id)}.json")
        if not os.path.exists(path):
            return jsonify({"error": "config not found"}), 404
        with open(path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))

    @bp.route("/configs/<config_id>", methods=["PUT"])
    def update_config(config_id):
        path = os.path.join(configs_dir, f"{secure_filename(config_id)}.json")
        if not os.path.exists(path):
            return jsonify({"error": "config not found"}), 404
        data = request.json or {}
        try:
            cfg = SyntheticConfig.from_dict(data)
            cfg.validate()
        except (KeyError, ValueError, FileNotFoundError) as e:
            return jsonify({"error": str(e)}), 400
        payload = cfg.to_dict()
        payload["_id"] = config_id
        payload["_updated_at"] = _utc_now()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return jsonify(payload)

    @bp.route("/configs/<config_id>", methods=["DELETE"])
    def delete_config(config_id):
        path = os.path.join(configs_dir, f"{secure_filename(config_id)}.json")
        if not os.path.exists(path):
            return jsonify({"error": "config not found"}), 404
        os.remove(path)
        return jsonify({"deleted": config_id})

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    @bp.route("/calibrate", methods=["POST"])
    def calibrate():
        """
        Generate a single calibration image with a specific product at exact coordinates and rotations.
        Returns a base64-encoded JPEG.
        """
        data = request.json or {}
        try:
            cfg = SyntheticConfig.from_dict(data)
            cfg.validate()
        except (KeyError, ValueError, FileNotFoundError) as e:
            return jsonify({"error": str(e)}), 400

        try:
            product_id = int(data.get("product_id", 0))
            x = int(data.get("x", 0))
            y = int(data.get("y", 0))
            rx = float(data.get("rx", 0.0))
            ry = float(data.get("ry", 0.0))
            rz = float(data.get("rz", 0.0))
            
            gen = SyntheticDatasetGenerator(cfg, store=None)
            image = gen.render_calibration_image(product_id, x, y, rx, ry, rz)
        except Exception as e:
            return jsonify({"error": f"Calibration failed: {e}"}), 500

        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            return jsonify({"image": b64})
        else:
            return jsonify({"error": "Failed to encode image"}), 500


    @bp.route("/preview", methods=["POST"])
    def preview():
        """
        Generate a small batch of preview images.
        Returns a list of base64-encoded JPEGs.
        """
        data = request.json or {}
        count = min(int(data.get("count", 6)), 12)
        try:
            cfg = SyntheticConfig.from_dict(data)
            cfg.validate()
        except (KeyError, ValueError, FileNotFoundError) as e:
            return jsonify({"error": str(e)}), 400

        try:
            gen = SyntheticDatasetGenerator(cfg, store=None)
            images = gen.preview(count=count)
        except Exception as e:
            return jsonify({"error": f"Preview failed: {e}"}), 500

        previews = []
        for img in images:
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                previews.append(base64.b64encode(buf.tobytes()).decode("utf-8"))

        return jsonify({"previews": previews, "count": len(previews)})

    # ------------------------------------------------------------------
    # Generation jobs
    # ------------------------------------------------------------------

    @bp.route("/generate", methods=["POST"])
    def start_generation():
        """Start a background generation job."""
        data = request.json or {}
        try:
            cfg = SyntheticConfig.from_dict(data)
            cfg.validate()
        except (KeyError, ValueError, FileNotFoundError) as e:
            return jsonify({"error": str(e)}), 400

        job_id   = str(uuid.uuid4())
        out_dir  = os.path.join(datasets_dir, job_id)
        os.makedirs(out_dir, exist_ok=True)

        job: Dict[str, Any] = {
            "id": job_id,
            "status": "starting",
            "created_at": _utc_now(),
            "total": cfg.num_images,
            "generated": 0,
            "failed": 0,
            "dataset_id": None,
            "output_dir": out_dir,
            "error": None,
            "config": cfg.to_dict(),
        }

        with _jobs_lock:
            _jobs[job_id] = job

        def _run():
            gen = SyntheticDatasetGenerator(cfg, store=store)
            with _jobs_lock:
                _generators[job_id] = gen
                _jobs[job_id]["status"] = "running"
                _jobs[job_id]["started_at"] = _utc_now()

            def _progress(done: int, total: int):
                with _jobs_lock:
                    if job_id in _jobs:
                        _jobs[job_id]["generated"] = done

            try:
                result = gen.generate(out_dir, progress_callback=_progress)
                with _jobs_lock:
                    if job_id in _jobs:
                        _jobs[job_id].update({
                            "status": "completed",
                            "generated": result.generated,
                            "failed": result.failed,
                            "dataset_id": result.dataset_id,
                            "ended_at": _utc_now(),
                            "errors": result.errors[:20],  # cap errors list
                        })
            except Exception as exc:
                with _jobs_lock:
                    if job_id in _jobs:
                        _jobs[job_id].update({
                            "status": "failed",
                            "error": str(exc),
                            "ended_at": _utc_now(),
                        })
            finally:
                with _jobs_lock:
                    _generators.pop(job_id, None)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        return jsonify(_job_response(job)), 201

    @bp.route("/generate/<job_id>/status", methods=["GET"])
    def job_status(job_id):
        with _jobs_lock:
            job = _jobs.get(job_id)
        if not job:
            return jsonify({"error": "job not found"}), 404
        return jsonify(_job_response(job))

    @bp.route("/generate", methods=["GET"])
    def list_jobs():
        with _jobs_lock:
            jobs = [_job_response(j) for j in _jobs.values()]
        jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
        return jsonify({"jobs": jobs})

    @bp.route("/generate/<job_id>/stop", methods=["POST"])
    def stop_job(job_id):
        with _jobs_lock:
            job  = _jobs.get(job_id)
            gen  = _generators.get(job_id)
        if not job:
            return jsonify({"error": "job not found"}), 404
        if gen:
            gen.stop()
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "stopping"
        return jsonify({"stopping": True, "job_id": job_id})

    # ------------------------------------------------------------------
    # Serve background image for the UI canvas
    # ------------------------------------------------------------------

    @bp.route("/uploads/background/<filename>/file", methods=["GET"])
    def get_background_file(filename):
        path = os.path.join(backgrounds_dir, secure_filename(filename))
        if not os.path.exists(path):
            return jsonify({"error": "not found"}), 404
        import mimetypes
        mimetype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as f:
            return Response(f.read(), mimetype=mimetype)

    return bp
