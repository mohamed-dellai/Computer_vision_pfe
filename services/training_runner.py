import os
import re
import subprocess
import threading
import uuid
import shutil
from datetime import datetime, timezone
from typing import Dict, Optional

import torch
import yaml


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrainingRunner:
    PRESET_MODELS = {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"}
    AUGMENTATION_DEFAULTS = {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.4,
        "close_mosaic": 10,
    }
    AUGMENTATION_RANGES = {
        "hsv_h": (0.0, 1.0),
        "hsv_s": (0.0, 1.0),
        "hsv_v": (0.0, 1.0),
        "degrees": (0.0, 180.0),
        "translate": (0.0, 1.0),
        "scale": (0.0, 10.0),
        "shear": (0.0, 180.0),
        "perspective": (0.0, 1.0),
        "flipud": (0.0, 1.0),
        "fliplr": (0.0, 1.0),
        "mosaic": (0.0, 1.0),
        "mixup": (0.0, 1.0),
        "copy_paste": (0.0, 1.0),
        "erasing": (0.0, 1.0),
        "close_mosaic": (0, 1000),
    }

    def __init__(self, base_dir: str, models_dir: str, store):
        self.base_dir = base_dir
        self.models_dir = models_dir
        self.store = store
        self._lock = threading.RLock()
        self._runtime: Dict[str, Dict[str, object]] = {}

    def initialize(self) -> None:
        active = self.store.get_active_job_id()
        if not active:
            return
        try:
            job = self.store.get_job(active)
        except KeyError:
            self.store.set_active_job(None)
            return
        if job.get("status") in {"starting", "running"}:
            self.store.update_job(
                active,
                {
                    "status": "interrupted",
                    "ended_at": _utc_now(),
                    "message": "training process was interrupted during restart",
                },
            )

    def _resolve_model_source(self, model_source: str) -> str:
        model_source = (model_source or "").strip()
        if not model_source:
            raise ValueError("model_source is required")
        if model_source in self.PRESET_MODELS:
            # Prefer validated local copies when present.
            candidates = [
                os.path.join(self.models_dir, model_source),
                os.path.abspath(model_source),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    self._validate_pt_checkpoint(candidate, preset=True)
                    return candidate
            # Allow Ultralytics to auto-download preset weights.
            return model_source

        model_name = os.path.basename(model_source)
        if not model_name.endswith(".pt"):
            raise ValueError("custom model must be a .pt file")
        abs_path = os.path.join(self.models_dir, model_name)
        if not os.path.exists(abs_path):
            raise ValueError(f"model not found: {model_name}")
        self._validate_pt_checkpoint(abs_path, preset=False)
        return abs_path

    def _validate_pt_checkpoint(self, path: str, preset: bool) -> None:
        try:
            torch.load(path, map_location="cpu")
        except Exception as exc:
            if preset:
                raise ValueError(
                    f"preset model file is corrupted: {path}. "
                    f"Delete it and retry to auto-redownload. Details: {exc}"
                ) from exc
            raise ValueError(
                f"custom model file is corrupted: {path}. "
                f"Re-export/re-upload a valid .pt checkpoint. Details: {exc}"
            ) from exc

    def _validate_class_ids(self, classes):
        if not classes:
            raise ValueError("dataset has no classes")
        ids = sorted(c["id"] for c in classes)
        expected = list(range(len(classes)))
        if ids != expected:
            raise ValueError("class IDs must be contiguous and start at 0 for YOLO training")
        return [c["name"] for c in sorted(classes, key=lambda x: x["id"])]

    def _validate_ratios(self, train_ratio: float, val_ratio: float, test_ratio: float) -> None:
        for ratio in (train_ratio, val_ratio, test_ratio):
            if ratio < 0.0 or ratio > 1.0:
                raise ValueError("split ratios must be between 0 and 1")
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError("split ratios must sum to 1.0")

    def _parse_augmentation_config(self, payload: Dict[str, object]) -> Optional[Dict[str, object]]:
        raw = payload.get("augmentation")
        if raw in (None, "", False):
            return None
        if not isinstance(raw, dict):
            raise ValueError("augmentation must be an object")

        enabled = raw.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in {"1", "true", "yes", "on"}
        if not enabled:
            return None

        unknown = sorted(k for k in raw.keys() if k not in self.AUGMENTATION_DEFAULTS and k != "enabled")
        if unknown:
            raise ValueError(f"unsupported augmentation option(s): {', '.join(unknown)}")

        parsed = {}
        for key, default in self.AUGMENTATION_DEFAULTS.items():
            value = raw.get(key, default)
            if value in (None, ""):
                value = default

            min_value, max_value = self.AUGMENTATION_RANGES[key]
            try:
                if isinstance(default, int):
                    parsed_value = int(value)
                else:
                    parsed_value = float(value)
            except Exception as exc:
                raise ValueError(f"augmentation.{key} must be a number") from exc

            if parsed_value < min_value or parsed_value > max_value:
                raise ValueError(f"augmentation.{key} must be between {min_value} and {max_value}")

            parsed[key] = parsed_value

        return parsed

    def start_job(self, payload: Dict[str, object]) -> Dict[str, object]:
        dataset_id = str(payload.get("dataset_id") or "")
        version_id = str(payload.get("annotation_version_id") or "")
        model_source = str(payload.get("model_source") or "")
        epochs = int(payload.get("epochs", 50))
        batch_size = int(payload.get("batch_size", 16))
        imgsz = int(payload.get("imgsz", 640))
        split_train = float(payload.get("split_train", 0.8))
        split_val = float(payload.get("split_val", 0.1))
        split_test = float(payload.get("split_test", 0.1))
        seed = int(payload.get("seed", 42))
        augmentation = self._parse_augmentation_config(payload)

        self._validate_ratios(split_train, split_val, split_test)
        model_path = self._resolve_model_source(model_source)
        material = self.store.get_training_material(dataset_id, version_id)
        stats = material["stats"]
        if stats["labeled_images"] <= 0:
            raise ValueError("selected annotation version has zero labeled images")

        names = self._validate_class_ids(material["classes"])
        splits = self.store.split_images(dataset_id, version_id, split_train, split_val, split_test, seed)
        labeled_splits = self._filter_to_labeled_images(splits, material["labels_dir"])
        total_labeled = sum(len(v) for v in labeled_splits.values())
        if total_labeled <= 0:
            raise ValueError("selected annotation version has no non-empty label files")

        with self._lock:
            active = self.store.get_active_job_id()
            if active:
                active_job = self.store.get_job(active)
                if active_job.get("status") in {"starting", "running"}:
                    raise RuntimeError("another training job is already running")

            job_id = str(uuid.uuid4())
            now = _utc_now()
            job_dir = os.path.join(self.base_dir, "jobs", job_id)
            splits_dir = os.path.join(job_dir, "splits")
            runs_dir = os.path.join(job_dir, "runs")
            prepared_dir = os.path.join(job_dir, "prepared_dataset")
            os.makedirs(splits_dir, exist_ok=True)
            os.makedirs(runs_dir, exist_ok=True)
            os.makedirs(prepared_dir, exist_ok=True)

            train_txt = os.path.join(splits_dir, "train.txt")
            val_txt = os.path.join(splits_dir, "val.txt")
            test_txt = os.path.join(splits_dir, "test.txt")

            staged_splits = self._stage_yolo_dataset(prepared_dir, labeled_splits, material["labels_dir"])
            self._write_list(train_txt, staged_splits["train"])
            self._write_list(val_txt, staged_splits["val"])
            self._write_list(test_txt, staged_splits["test"])

            dataset_yaml_path = os.path.join(job_dir, "dataset.yaml")
            with open(dataset_yaml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    {
                        "train": train_txt,
                        "val": val_txt,
                        "test": test_txt,
                        "nc": len(names),
                        "names": names,
                    },
                    f,
                    sort_keys=False,
                )

            logs_path = os.path.join(job_dir, "logs.txt")
            state_path = os.path.join(job_dir, "state.json")
            cmd = [
                "yolo",
                "detect",
                "train",
                f"model={model_path}",
                f"data={dataset_yaml_path}",
                f"epochs={epochs}",
                f"batch={batch_size}",
                f"imgsz={imgsz}",
                f"project={runs_dir}",
                "name=run",
            ]
            if augmentation:
                for key, value in augmentation.items():
                    cmd.append(f"{key}={value}")

            job = {
                "id": job_id,
                "dataset_id": dataset_id,
                "annotation_version_id": version_id,
                "model_source": model_source,
                "resolved_model": model_path,
                "epochs": epochs,
                "batch_size": batch_size,
                "imgsz": imgsz,
                "split_train": split_train,
                "split_val": split_val,
                "split_test": split_test,
                "seed": seed,
                "augmentation": augmentation,
                "status": "starting",
                "progress_pct": 0.0,
                "epoch_current": 0,
                "epoch_total": epochs,
                "metrics": {},
                "created_at": now,
                "started_at": now,
                "ended_at": None,
                "job_dir": job_dir,
                "logs_path": logs_path,
                "state_path": state_path,
                "command": cmd,
                "split_counts": {k: len(v) for k, v in staged_splits.items()},
                "message": "",
            }
            self.store.create_job(job)
            self._runtime[job_id] = {"stop_event": threading.Event(), "process": None}
            thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
            self._runtime[job_id]["thread"] = thread
            thread.start()
            return job

    def _write_list(self, path: str, rows) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(f"{row}\n")

    def _filter_to_labeled_images(self, splits, labels_dir: str):
        filtered = {"train": [], "val": [], "test": []}
        for split_name in filtered.keys():
            for image_path in splits.get(split_name, []):
                image_id = os.path.splitext(os.path.basename(image_path))[0]
                label_path = os.path.join(labels_dir, f"{image_id}.txt")
                if not os.path.exists(label_path):
                    continue
                try:
                    if os.path.getsize(label_path) <= 0:
                        continue
                except OSError:
                    continue
                filtered[split_name].append(image_path)
        return filtered

    def _stage_yolo_dataset(self, prepared_dir: str, splits, labels_dir: str):
        out = {"train": [], "val": [], "test": []}
        for split_name in out.keys():
            img_dir = os.path.join(prepared_dir, "images", split_name)
            lbl_dir = os.path.join(prepared_dir, "labels", split_name)
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)

            for src_image in splits.get(split_name, []):
                image_name = os.path.basename(src_image)
                image_id = os.path.splitext(image_name)[0]
                src_label = os.path.join(labels_dir, f"{image_id}.txt")
                if not os.path.exists(src_label):
                    continue

                dst_image = os.path.join(img_dir, image_name)
                dst_label = os.path.join(lbl_dir, f"{image_id}.txt")

                shutil.copy2(src_image, dst_image)
                shutil.copy2(src_label, dst_label)
                out[split_name].append(dst_image)
        return out

    def _append_job_log(self, logs_path: str, line: str) -> None:
        os.makedirs(os.path.dirname(logs_path), exist_ok=True)
        with open(logs_path, "a", encoding="utf-8", errors="replace") as logf:
            logf.write(f"{line.rstrip()}\n")

    def _parse_progress(self, line: str, epoch_total: int) -> Optional[int]:
        # Ultralytics commonly logs "1/100" per epoch; parse that pattern.
        match = re.search(r"\b(\d+)\s*/\s*(\d+)\b", line)
        if not match:
            return None
        cur = int(match.group(1))
        tot = int(match.group(2))
        if tot <= 0:
            return None
        if epoch_total and tot != epoch_total:
            return None
        if cur < 0:
            return None
        return cur

    def _run_job(self, job_id: str) -> None:
        try:
            job = self.store.get_job(job_id)
        except KeyError:
            return

        runtime = self._runtime.get(job_id)
        if not runtime:
            return
        stop_event = runtime["stop_event"]
        process = None
        logs_path = str(job.get("logs_path") or "")
        try:
            self._append_job_log(logs_path, f"[{_utc_now()}] Starting training job {job_id}")
            self._append_job_log(logs_path, f"[{_utc_now()}] Command: {' '.join(job['command'])}")
            with open(logs_path, "a", encoding="utf-8", errors="replace") as logf:
                self.store.update_job(job_id, {"status": "running", "message": ""}, clear_active_if_terminal=False)
                process = subprocess.Popen(
                    job["command"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    cwd=self.models_dir,
                )
                with self._lock:
                    if job_id in self._runtime:
                        self._runtime[job_id]["process"] = process

                epoch_total = int(job.get("epoch_total") or 0)
                epoch_current = 0
                for line in process.stdout:
                    if line is None:
                        continue
                    logf.write(line)
                    logf.flush()
                    parsed = self._parse_progress(line, epoch_total)
                    if parsed is not None and parsed >= epoch_current:
                        epoch_current = parsed
                        progress = 0.0 if epoch_total <= 0 else min(100.0, (epoch_current / epoch_total) * 100.0)
                        self.store.update_job(
                            job_id,
                            {"epoch_current": epoch_current, "progress_pct": progress},
                            clear_active_if_terminal=False,
                        )
                    if stop_event.is_set():
                        break

                return_code = process.wait(timeout=10)
                if stop_event.is_set():
                    status = "stopped"
                    message = "training stopped by user"
                    self._append_job_log(logs_path, f"[{_utc_now()}] Stop requested by user")
                elif return_code == 0:
                    status = "completed"
                    message = "training completed"
                    self._append_job_log(logs_path, f"[{_utc_now()}] Training completed successfully")
                else:
                    status = "failed"
                    message = f"training exited with code {return_code}"
                    self._append_job_log(
                        logs_path,
                        f"[{_utc_now()}] Training failed with exit code {return_code}",
                    )
                self.store.update_job(
                    job_id,
                    {
                        "status": status,
                        "message": message,
                        "ended_at": _utc_now(),
                        "progress_pct": 100.0 if status == "completed" else self.store.get_job(job_id).get("progress_pct", 0.0),
                    },
                )
        except Exception as exc:
            if logs_path:
                self._append_job_log(logs_path, f"[{_utc_now()}] Exception: {repr(exc)}")
            self.store.update_job(
                job_id,
                {"status": "failed", "message": str(exc), "ended_at": _utc_now()},
            )
        finally:
            if process and process.poll() is None:
                try:
                    process.terminate()
                except Exception:
                    pass
            with self._lock:
                self._runtime.pop(job_id, None)

    def list_jobs(self):
        return self.store.list_jobs()

    def get_job(self, job_id: str):
        return self.store.get_job(job_id)

    def stop_job(self, job_id: str) -> Dict[str, object]:
        with self._lock:
            runtime = self._runtime.get(job_id)
            if runtime:
                runtime["stop_event"].set()
                process = runtime.get("process")
                if process and process.poll() is None:
                    try:
                        process.terminate()
                    except Exception:
                        pass
            else:
                job = self.store.get_job(job_id)
                if job.get("status") not in {"starting", "running"}:
                    return job
        return self.store.update_job(
            job_id,
            {"status": "stopped", "message": "stop requested", "ended_at": _utc_now()},
        )

    def get_logs(self, job_id: str, offset: int = 0, limit: int = 20000) -> Dict[str, object]:
        job = self.store.get_job(job_id)
        logs_path = job.get("logs_path")
        if not logs_path or not os.path.exists(logs_path):
            return {"offset": 0, "next_offset": 0, "chunk": ""}
        with open(logs_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(max(0, offset))
            chunk = f.read(limit)
            next_offset = f.tell()
        return {"offset": offset, "next_offset": next_offset, "chunk": chunk}
