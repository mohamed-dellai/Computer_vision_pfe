import json
import os
import random
import shutil
import threading
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import cv2
from werkzeug.utils import secure_filename


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrainingStore:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.datasets_dir = os.path.join(base_dir, "datasets")
        self.jobs_dir = os.path.join(base_dir, "jobs")
        self.registry_path = os.path.join(base_dir, "registry.json")
        self._lock = threading.RLock()

        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.jobs_dir, exist_ok=True)
        self._ensure_registry()

    def _ensure_registry(self) -> None:
        with self._lock:
            if not os.path.exists(self.registry_path):
                self._save_registry({"datasets": {}, "jobs": {}, "active_job_id": None})
                return
            data = self._load_registry()
            changed = False
            if "datasets" not in data:
                data["datasets"] = {}
                changed = True
            if "jobs" not in data:
                data["jobs"] = {}
                changed = True
            if "active_job_id" not in data:
                data["active_job_id"] = None
                changed = True
            if changed:
                self._save_registry(data)

    def _load_registry(self) -> Dict[str, Any]:
        with open(self.registry_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_registry(self, data: Dict[str, Any]) -> None:
        tmp = f"{self.registry_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.registry_path)

    def _dataset_dir(self, dataset_id: str) -> str:
        return os.path.join(self.datasets_dir, dataset_id)

    def _dataset_images_dir(self, dataset_id: str) -> str:
        return os.path.join(self._dataset_dir(dataset_id), "images")

    def _dataset_versions_dir(self, dataset_id: str) -> str:
        return os.path.join(self._dataset_dir(dataset_id), "annotation_versions")

    def _version_labels_dir(self, dataset_id: str, version_id: str) -> str:
        return os.path.join(self._dataset_versions_dir(dataset_id), version_id, "labels")

    def _require_dataset(self, data: Dict[str, Any], dataset_id: str) -> Dict[str, Any]:
        dataset = data["datasets"].get(dataset_id)
        if not dataset:
            raise KeyError("dataset not found")
        return dataset

    def _require_version(self, dataset: Dict[str, Any], version_id: str) -> Dict[str, Any]:
        version = dataset["annotation_versions"].get(version_id)
        if not version:
            raise KeyError("annotation version not found")
        return version

    def list_datasets(self) -> List[Dict[str, Any]]:
        with self._lock:
            data = self._load_registry()
            out = []
            for dataset in data["datasets"].values():
                out.append({
                    "id": dataset["id"],
                    "name": dataset["name"],
                    "description": dataset.get("description", ""),
                    "created_at": dataset["created_at"],
                    "updated_at": dataset["updated_at"],
                    "image_count": len(dataset.get("images", {})),
                    "class_count": len(dataset.get("classes", [])),
                    "annotation_version_count": len(dataset.get("annotation_versions", {})),
                })
            out.sort(key=lambda x: x["created_at"], reverse=True)
            return out

    def create_dataset(self, name: str, description: str = "") -> Dict[str, Any]:
        if not name or not str(name).strip():
            raise ValueError("name is required")

        with self._lock:
            data = self._load_registry()
            dataset_id = str(uuid.uuid4())
            now = _utc_now()
            dataset = {
                "id": dataset_id,
                "name": str(name).strip(),
                "description": description or "",
                "created_at": now,
                "updated_at": now,
                "images": {},
                "classes": [],
                "annotation_versions": {},
            }
            data["datasets"][dataset_id] = dataset
            os.makedirs(self._dataset_images_dir(dataset_id), exist_ok=True)
            os.makedirs(self._dataset_versions_dir(dataset_id), exist_ok=True)
            self._save_registry(data)
            return dataset

    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        with self._lock:
            data = self._load_registry()
            dataset = self._require_dataset(data, dataset_id)
            return dataset

    def update_dataset(self, dataset_id: str, name: Optional[str], description: Optional[str]) -> Dict[str, Any]:
        with self._lock:
            data = self._load_registry()
            dataset = self._require_dataset(data, dataset_id)
            if name is not None:
                if not str(name).strip():
                    raise ValueError("name cannot be empty")
                dataset["name"] = str(name).strip()
            if description is not None:
                dataset["description"] = description
            dataset["updated_at"] = _utc_now()
            self._save_registry(data)
            return dataset

    def delete_dataset(self, dataset_id: str) -> None:
        with self._lock:
            data = self._load_registry()
            self._require_dataset(data, dataset_id)
            del data["datasets"][dataset_id]
            self._save_registry(data)
            shutil.rmtree(self._dataset_dir(dataset_id), ignore_errors=True)

    def add_images(self, dataset_id: str, files) -> List[Dict[str, Any]]:
        saved = []
        with self._lock:
            data = self._load_registry()
            dataset = self._require_dataset(data, dataset_id)
            images_dir = self._dataset_images_dir(dataset_id)
            os.makedirs(images_dir, exist_ok=True)

            for file in files:
                if not file or not file.filename:
                    continue
                secure = secure_filename(file.filename)
                _, ext = os.path.splitext(secure)
                ext = ext.lower()
                if ext not in ALLOWED_IMAGE_EXTENSIONS:
                    continue
                image_id = str(uuid.uuid4())
                filename = f"{image_id}{ext}"
                abs_path = os.path.join(images_dir, filename)
                file.save(abs_path)
                width, height = self._read_image_size(abs_path)
                img = {
                    "id": image_id,
                    "filename": filename,
                    "original_name": secure,
                    "uploaded_at": _utc_now(),
                }
                if width and height:
                    img["width"] = width
                    img["height"] = height
                dataset["images"][image_id] = img
                saved.append(img)

            dataset["updated_at"] = _utc_now()
            self._save_registry(data)
        return saved

    def add_images_from_video(self, dataset_id: str, video_file, every_nth_frame: int = 1) -> Dict[str, Any]:
        if video_file is None or not getattr(video_file, "filename", None):
            raise ValueError("video file is required")
        if every_nth_frame < 1:
            raise ValueError("every_nth_frame must be >= 1")

        secure = secure_filename(video_file.filename)
        _, ext = os.path.splitext(secure)
        ext = ext.lower()
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            raise ValueError("unsupported video format")

        temp_path = None
        extracted = []
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                temp_path = tmp.name
            video_file.save(temp_path)

            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                raise ValueError("failed to open uploaded video")

            try:
                frame_idx = 0
                saved_idx = 0
                with self._lock:
                    data = self._load_registry()
                    dataset = self._require_dataset(data, dataset_id)
                    images_dir = self._dataset_images_dir(dataset_id)
                    os.makedirs(images_dir, exist_ok=True)

                    while True:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            break

                        if frame_idx % every_nth_frame == 0:
                            image_id = str(uuid.uuid4())
                            filename = f"{image_id}.jpg"
                            abs_path = os.path.join(images_dir, filename)
                            ok = cv2.imwrite(abs_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                            if ok:
                                height, width = frame.shape[:2]
                                item = {
                                    "id": image_id,
                                    "filename": filename,
                                    "original_name": f"{os.path.splitext(secure)[0]}_frame_{frame_idx:06d}.jpg",
                                    "uploaded_at": _utc_now(),
                                    "width": int(width),
                                    "height": int(height),
                                }
                                dataset["images"][image_id] = item
                                extracted.append(item)
                                saved_idx += 1
                        frame_idx += 1

                    dataset["updated_at"] = _utc_now()
                    self._save_registry(data)

                return {
                    "extracted_count": saved_idx,
                    "every_nth_frame": every_nth_frame,
                    "video_name": secure,
                    "images": extracted,
                }
            finally:
                cap.release()
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except FileNotFoundError:
                    pass

    @staticmethod
    def _read_image_size(path: str):
        image = cv2.imread(path)
        if image is None:
            return None, None
        height, width = image.shape[:2]
        return int(width), int(height)

    @staticmethod
    def _parse_split_ratio(value, default: float) -> float:
        if value is None or value == "":
            return default
        try:
            ratio = float(value)
        except Exception as exc:
            raise ValueError("split values must be numbers between 0.05 and 0.95") from exc
        if ratio < 0.05 or ratio > 0.95:
            raise ValueError("split values must be between 0.05 and 0.95")
        return ratio

    @staticmethod
    def _parse_crop_edge(value, default: float, name: str) -> float:
        if value is None or value == "":
            return default
        try:
            edge = float(value)
        except Exception as exc:
            raise ValueError(f"{name} must be a number between 0 and 1") from exc
        if edge < 0.0 or edge > 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
        return edge

    def transform_images_crop(
        self,
        dataset_id: str,
        image_ids: List[str],
        mode: str,
        split_x=None,
        split_y=None,
        crop_left=None,
        crop_top=None,
        crop_right=None,
        crop_bottom=None,
        create_new_dataset: bool = False,
        new_dataset_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        mode = (mode or "").strip().lower()
        if mode not in {"static", "vertical", "horizontal", "grid"}:
            raise ValueError("mode must be static, vertical, horizontal, or grid")
        if not isinstance(image_ids, list) or not image_ids:
            raise ValueError("image_ids must be a non-empty list")
        split_x_ratio = self._parse_split_ratio(split_x, 0.5)
        split_y_ratio = self._parse_split_ratio(split_y, 0.5)
        crop_left_ratio = self._parse_crop_edge(crop_left, 0.05, "crop_left")
        crop_top_ratio = self._parse_crop_edge(crop_top, 0.05, "crop_top")
        crop_right_ratio = self._parse_crop_edge(crop_right, 0.95, "crop_right")
        crop_bottom_ratio = self._parse_crop_edge(crop_bottom, 0.95, "crop_bottom")
        if crop_right_ratio - crop_left_ratio < 0.01:
            raise ValueError("crop_right must be greater than crop_left")
        if crop_bottom_ratio - crop_top_ratio < 0.01:
            raise ValueError("crop_bottom must be greater than crop_top")

        created = []
        skipped = []
        with self._lock:
            data = self._load_registry()
            dataset = self._require_dataset(data, dataset_id)
            images = dataset.get("images", {})
            
            target_dataset = dataset
            target_dataset_id = dataset_id
            version_id_map = {}
            
            if create_new_dataset:
                target_dataset_id = str(uuid.uuid4())
                now = _utc_now()
                target_dataset = {
                    "id": target_dataset_id,
                    "name": str(new_dataset_name).strip() if new_dataset_name else f"{dataset.get('name', 'Dataset')} (Cropped)",
                    "description": f"Cropped from dataset {dataset.get('name', dataset_id)}",
                    "created_at": now,
                    "updated_at": now,
                    "images": {},
                    "classes": [dict(c) for c in dataset.get("classes", [])],
                    "annotation_versions": {},
                }
                data["datasets"][target_dataset_id] = target_dataset
                
                os.makedirs(self._dataset_images_dir(target_dataset_id), exist_ok=True)
                os.makedirs(self._dataset_versions_dir(target_dataset_id), exist_ok=True)
                
                for src_vid, src_v in dataset.get("annotation_versions", {}).items():
                    new_vid = str(uuid.uuid4())
                    version_id_map[src_vid] = new_vid
                    target_dataset["annotation_versions"][new_vid] = {
                        "id": new_vid,
                        "name": src_v.get("name", "Version 1"),
                        "source_version_id": None,
                        "created_at": now,
                        "updated_at": now,
                    }
                    os.makedirs(self._version_labels_dir(target_dataset_id, new_vid), exist_ok=True)

            images_dir = self._dataset_images_dir(dataset_id)
            target_images_dir = self._dataset_images_dir(target_dataset_id)
            os.makedirs(target_images_dir, exist_ok=True)

            seen = set()
            for image_id in image_ids:
                if image_id in seen:
                    continue
                seen.add(image_id)

                source = images.get(image_id)
                if not source:
                    skipped.append({"image_id": image_id, "reason": "image not found"})
                    continue

                source_path = os.path.join(images_dir, source["filename"])
                image = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    skipped.append({"image_id": image_id, "reason": "failed to read image"})
                    continue

                height, width = image.shape[:2]
                if mode == "static":
                    left_px = int(round(width * crop_left_ratio))
                    top_px = int(round(height * crop_top_ratio))
                    right_px = int(round(width * crop_right_ratio))
                    bottom_px = int(round(height * crop_bottom_ratio))
                    if left_px < 0 or top_px < 0 or right_px > width or bottom_px > height:
                        skipped.append({"image_id": image_id, "reason": "crop rectangle is outside image bounds"})
                        continue
                    if right_px <= left_px or bottom_px <= top_px:
                        skipped.append({"image_id": image_id, "reason": "crop rectangle is too small"})
                        continue
                    crops = [
                        ("static", image[top_px:bottom_px, left_px:right_px]),
                    ]
                else:
                    split_x_px = int(round(width * split_x_ratio))
                    split_y_px = int(round(height * split_y_ratio))
                    if split_x_px <= 0 or split_x_px >= width:
                        skipped.append({"image_id": image_id, "reason": "vertical split is outside image bounds"})
                        continue
                    if split_y_px <= 0 or split_y_px >= height:
                        skipped.append({"image_id": image_id, "reason": "horizontal split is outside image bounds"})
                        continue

                if mode != "static":
                    if mode == "vertical":
                        crops = [
                            ("left", image[:, :split_x_px]),
                            ("right", image[:, split_x_px:]),
                        ]
                    elif mode == "horizontal":
                        crops = [
                            ("top", image[:split_y_px, :]),
                            ("bottom", image[split_y_px:, :]),
                        ]
                    else:
                        crops = [
                            ("top_left", image[:split_y_px, :split_x_px]),
                            ("top_right", image[:split_y_px, split_x_px:]),
                            ("bottom_left", image[split_y_px:, :split_x_px]),
                            ("bottom_right", image[split_y_px:, split_x_px:]),
                        ]

                base_name = os.path.splitext(source.get("original_name") or source.get("filename") or "image")[0]
                safe_base = secure_filename(base_name) or "image"
                for label, crop in crops:
                    crop_id = str(uuid.uuid4())
                    filename = f"{crop_id}.jpg"
                    abs_path = os.path.join(target_images_dir, filename)
                    output_crop = crop
                    if output_crop.ndim == 3 and output_crop.shape[2] == 4:
                        output_crop = cv2.cvtColor(output_crop, cv2.COLOR_BGRA2BGR)
                    ok = cv2.imwrite(abs_path, output_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    if not ok:
                        skipped.append({"image_id": image_id, "reason": f"failed to write {label} crop"})
                        continue

                    crop_h, crop_w = output_crop.shape[:2]
                    item = {
                        "id": crop_id,
                        "filename": filename,
                        "original_name": f"{safe_base}_crop_{label}.jpg",
                        "uploaded_at": _utc_now(),
                        "width": int(crop_w),
                        "height": int(crop_h),
                        "source_image_id": image_id,
                        "transform": {
                            "type": "crop",
                            "mode": mode,
                            "part": label,
                            "split_x": split_x_ratio,
                            "split_y": split_y_ratio,
                            "crop_left": crop_left_ratio,
                            "crop_top": crop_top_ratio,
                            "crop_right": crop_right_ratio,
                            "crop_bottom": crop_bottom_ratio,
                        },
                    }
                    target_dataset["images"][crop_id] = item
                    created.append(item)
                    
                    if create_new_dataset:
                        cl, ct, cr, cb = crop_left_ratio, crop_top_ratio, crop_right_ratio, crop_bottom_ratio
                        if mode == "vertical":
                            if label == "left":
                                cl, cr, ct, cb = 0.0, split_x_ratio, 0.0, 1.0
                            elif label == "right":
                                cl, cr, ct, cb = split_x_ratio, 1.0, 0.0, 1.0
                        elif mode == "horizontal":
                            if label == "top":
                                cl, cr, ct, cb = 0.0, 1.0, 0.0, split_y_ratio
                            elif label == "bottom":
                                cl, cr, ct, cb = 0.0, 1.0, split_y_ratio, 1.0
                        elif mode == "grid":
                            if label == "top_left":
                                cl, cr, ct, cb = 0.0, split_x_ratio, 0.0, split_y_ratio
                            elif label == "top_right":
                                cl, cr, ct, cb = split_x_ratio, 1.0, 0.0, split_y_ratio
                            elif label == "bottom_left":
                                cl, cr, ct, cb = 0.0, split_x_ratio, split_y_ratio, 1.0
                            elif label == "bottom_right":
                                cl, cr, ct, cb = split_x_ratio, 1.0, split_y_ratio, 1.0
                                
                        for src_vid, new_vid in version_id_map.items():
                            src_label_file = os.path.join(self._version_labels_dir(dataset_id, src_vid), f"{image_id}.txt")
                            annots = self._read_annotations_from_file(src_label_file)
                            if annots:
                                new_annots = self._recalculate_annotations(annots, width, height, cl, ct, cr, cb)
                                if new_annots:
                                    lines = self._annotations_to_lines(new_annots, {c["id"] for c in target_dataset["classes"]})
                                    new_label_file = os.path.join(self._version_labels_dir(target_dataset_id, new_vid), f"{crop_id}.txt")
                                    with open(new_label_file, "w", encoding="utf-8") as f:
                                        f.write("\n".join(lines) + "\n")

            dataset["updated_at"] = _utc_now()
            if create_new_dataset:
                target_dataset["updated_at"] = _utc_now()
            self._save_registry(data)

        return {
            "mode": mode,
            "split_x": split_x_ratio,
            "split_y": split_y_ratio,
            "crop_left": crop_left_ratio,
            "crop_top": crop_top_ratio,
            "crop_right": crop_right_ratio,
            "crop_bottom": crop_bottom_ratio,
            "created_count": len(created),
            "created": created,
            "skipped": skipped,
            "target_dataset_id": target_dataset_id,
        }

    def list_images(self, dataset_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            data = self._load_registry()
            dataset = self._require_dataset(data, dataset_id)
            images = list(dataset.get("images", {}).values())
            images.sort(key=lambda x: x["uploaded_at"], reverse=True)
            return images

    def get_image_abs_path(self, dataset_id: str, image_id: str) -> str:
        with self._lock:
            data = self._load_registry()
            dataset = self._require_dataset(data, dataset_id)
            image = dataset.get("images", {}).get(image_id)
            if not image:
                raise KeyError("image not found")
            return os.path.join(self._dataset_images_dir(dataset_id), image["filename"])

    def delete_image(self, dataset_id: str, image_id: str) -> None:
        with self._lock:
            data = self._load_registry()
            dataset = self._require_dataset(data, dataset_id)
            image = dataset.get("images", {}).get(image_id)
            if not image:
                raise KeyError("image not found")

            img_path = os.path.join(self._dataset_images_dir(dataset_id), image["filename"])
            try:
                os.remove(img_path)
            except FileNotFoundError:
                pass

            for version_id in dataset.get("annotation_versions", {}).keys():
                label_file = os.path.join(self._version_labels_dir(dataset_id, version_id), f"{image_id}.txt")
                try:
                    os.remove(label_file)
                except FileNotFoundError:
                    pass

            del dataset["images"][image_id]
            dataset["updated_at"] = _utc_now()
            self._save_registry(data)

    def list_classes(self, dataset_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            data = self._load_registry()
            dataset = self._require_dataset(data, dataset_id)
            classes = list(dataset.get("classes", []))
            classes.sort(key=lambda x: x["id"])
            return classes

    def add_class(self, dataset_id: str, name: str, color: str, class_id: Optional[int] = None) -> Dict[str, Any]:
        if not name or not str(name).strip():
            raise ValueError("class name is required")
        if not color:
            raise ValueError("class color is required")

        with self._lock:
            data = self._load_registry()
            dataset = self._require_dataset(data, dataset_id)
            classes = dataset["classes"]

            existing_ids = {c["id"] for c in classes}
            if class_id is None:
                cid = 0
                while cid in existing_ids:
                    cid += 1
            else:
                cid = int(class_id)
                if cid in existing_ids:
                    raise ValueError("class id already exists")
            if any(c["name"] == name for c in classes):
                raise ValueError("class name already exists")

            item = {"id": cid, "name": name.strip(), "color": color}
            classes.append(item)
            classes.sort(key=lambda x: x["id"])
            dataset["updated_at"] = _utc_now()
            self._save_registry(data)
            return item

    def update_class(
        self,
        dataset_id: str,
        class_id: int,
        name: Optional[str] = None,
        color: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            data = self._load_registry()
            dataset = self._require_dataset(data, dataset_id)
            cls = None
            for c in dataset["classes"]:
                if c["id"] == int(class_id):
                    cls = c
                    break
            if not cls:
                raise KeyError("class not found")

            if name is not None:
                if not str(name).strip():
                    raise ValueError("class name cannot be empty")
                if any(c["name"] == name and c["id"] != cls["id"] for c in dataset["classes"]):
                    raise ValueError("class name already exists")
                cls["name"] = name.strip()
            if color is not None:
                cls["color"] = color
            dataset["updated_at"] = _utc_now()
            self._save_registry(data)
            return cls

    def _iter_label_class_ids(self, dataset_id: str, dataset: Dict[str, Any]):
        for version_id in dataset.get("annotation_versions", {}).keys():
            labels_dir = self._version_labels_dir(dataset_id, version_id)
            if not os.path.exists(labels_dir):
                continue
            for name in os.listdir(labels_dir):
                if not name.endswith(".txt"):
                    continue
                label_file = os.path.join(labels_dir, name)
                try:
                    with open(label_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            first = line.split()[0]
                            yield int(first)
                except Exception:
                    continue

    def delete_class(self, dataset_id: str, class_id: int) -> None:
        with self._lock:
            data = self._load_registry()
            dataset = self._require_dataset(data, dataset_id)
            class_id = int(class_id)
            if class_id in set(self._iter_label_class_ids(dataset_id, dataset)):
                raise RuntimeError("class is used in annotations")

            original_count = len(dataset["classes"])
            dataset["classes"] = [c for c in dataset["classes"] if c["id"] != class_id]
            if len(dataset["classes"]) == original_count:
                raise KeyError("class not found")
            dataset["updated_at"] = _utc_now()
            self._save_registry(data)

    def list_annotation_versions(self, dataset_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            data = self._load_registry()
            dataset = self._require_dataset(data, dataset_id)
            versions = list(dataset.get("annotation_versions", {}).values())
            versions.sort(key=lambda x: x["created_at"], reverse=True)
            return versions

    def create_annotation_version(
        self, dataset_id: str, name: str, source_version_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if not name or not str(name).strip():
            raise ValueError("version name is required")
        with self._lock:
            data = self._load_registry()
            dataset = self._require_dataset(data, dataset_id)
            versions = dataset["annotation_versions"]
            if any(v["name"] == name for v in versions.values()):
                raise ValueError("annotation version name already exists")

            version_id = str(uuid.uuid4())
            now = _utc_now()
            version = {
                "id": version_id,
                "name": name.strip(),
                "source_version_id": source_version_id,
                "created_at": now,
                "updated_at": now,
            }
            versions[version_id] = version

            target_labels_dir = self._version_labels_dir(dataset_id, version_id)
            os.makedirs(target_labels_dir, exist_ok=True)
            if source_version_id:
                if source_version_id not in versions:
                    raise KeyError("source annotation version not found")
                source_dir = self._version_labels_dir(dataset_id, source_version_id)
                if os.path.exists(source_dir):
                    for name in os.listdir(source_dir):
                        if not name.endswith(".txt"):
                            continue
                        shutil.copy2(os.path.join(source_dir, name), os.path.join(target_labels_dir, name))

            dataset["updated_at"] = _utc_now()
            self._save_registry(data)
            return version

    def update_annotation_version(self, dataset_id: str, version_id: str, name: str) -> Dict[str, Any]:
        if not name or not str(name).strip():
            raise ValueError("version name is required")
        with self._lock:
            data = self._load_registry()
            dataset = self._require_dataset(data, dataset_id)
            version = self._require_version(dataset, version_id)
            if any(v["name"] == name and v["id"] != version_id for v in dataset["annotation_versions"].values()):
                raise ValueError("annotation version name already exists")
            version["name"] = name.strip()
            version["updated_at"] = _utc_now()
            dataset["updated_at"] = _utc_now()
            self._save_registry(data)
            return version

    def delete_annotation_version(self, dataset_id: str, version_id: str) -> None:
        with self._lock:
            data = self._load_registry()
            dataset = self._require_dataset(data, dataset_id)
            self._require_version(dataset, version_id)
            del dataset["annotation_versions"][version_id]
            dataset["updated_at"] = _utc_now()
            self._save_registry(data)
            shutil.rmtree(os.path.join(self._dataset_versions_dir(dataset_id), version_id), ignore_errors=True)

    def get_annotations(self, dataset_id: str, version_id: str, image_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            data = self._load_registry()
            dataset = self._require_dataset(data, dataset_id)
            self._require_version(dataset, version_id)
            if image_id not in dataset.get("images", {}):
                raise KeyError("image not found")
            label_file = os.path.join(self._version_labels_dir(dataset_id, version_id), f"{image_id}.txt")
            if not os.path.exists(label_file):
                return []

            result = []
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    result.append({
                        "class_id": int(parts[0]),
                        "x": float(parts[1]),
                        "y": float(parts[2]),
                        "w": float(parts[3]),
                        "h": float(parts[4]),
                    })
            return result

    @staticmethod
    def _read_annotations_from_file(label_file: str) -> List[Dict[str, Any]]:
        if not os.path.exists(label_file):
            return []
        result = []
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                result.append({
                    "class_id": int(parts[0]),
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "w": float(parts[3]),
                    "h": float(parts[4]),
                })
        return result

    @staticmethod
    def _annotations_to_lines(annotations: List[Dict[str, Any]], class_ids: set) -> List[str]:
        lines = []
        for i, item in enumerate(annotations):
            try:
                cid = int(item["class_id"])
                x = float(item["x"])
                y = float(item["y"])
                w = float(item["w"])
                h = float(item["h"])
            except Exception as exc:
                raise ValueError(f"invalid annotation at index {i}: {exc}")
            if cid not in class_ids:
                raise ValueError(f"class_id {cid} does not exist")
            for value, key in ((x, "x"), (y, "y"), (w, "w"), (h, "h")):
                if value < 0.0 or value > 1.0:
                    raise ValueError(f"{key} must be between 0 and 1")
            if w <= 0.0 or h <= 0.0:
                raise ValueError("w and h must be > 0")
            lines.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        return lines

    @staticmethod
    def _recalculate_annotations(
        annotations: List[Dict[str, Any]], 
        img_w: int, 
        img_h: int, 
        crop_left: float, 
        crop_top: float, 
        crop_right: float, 
        crop_bottom: float
    ) -> List[Dict[str, Any]]:
        result = []
        crop_abs_left = crop_left * img_w
        crop_abs_top = crop_top * img_h
        crop_abs_right = crop_right * img_w
        crop_abs_bottom = crop_bottom * img_h
        crop_w = crop_abs_right - crop_abs_left
        crop_h = crop_abs_bottom - crop_abs_top
        
        if crop_w <= 0 or crop_h <= 0:
            return []

        for ann in annotations:
            # Denormalize to absolute pixels
            abs_cx = ann["x"] * img_w
            abs_cy = ann["y"] * img_h
            abs_w = ann["w"] * img_w
            abs_h = ann["h"] * img_h
            
            box_left = abs_cx - abs_w / 2
            box_top = abs_cy - abs_h / 2
            box_right = abs_cx + abs_w / 2
            box_bottom = abs_cy + abs_h / 2
            
            # Intersection with crop box
            inter_left = max(box_left, crop_abs_left)
            inter_top = max(box_top, crop_abs_top)
            inter_right = min(box_right, crop_abs_right)
            inter_bottom = min(box_bottom, crop_abs_bottom)
            
            # Completely outside
            if inter_left >= inter_right or inter_top >= inter_bottom:
                continue
                
            # New relative absolute coordinates inside the crop box
            new_box_left = inter_left - crop_abs_left
            new_box_top = inter_top - crop_abs_top
            new_box_right = inter_right - crop_abs_left
            new_box_bottom = inter_bottom - crop_abs_top
            
            # Renormalize to 0.0 - 1.0 based on crop dimensions
            new_w = (new_box_right - new_box_left) / crop_w
            new_h = (new_box_bottom - new_box_top) / crop_h
            new_cx = (new_box_left + new_box_right) / 2 / crop_w
            new_cy = (new_box_top + new_box_bottom) / 2 / crop_h
            
            new_cx = max(0.0, min(1.0, new_cx))
            new_cy = max(0.0, min(1.0, new_cy))
            new_w = max(0.0, min(1.0, new_w))
            new_h = max(0.0, min(1.0, new_h))
            
            if new_w > 0 and new_h > 0:
                result.append({
                    "class_id": ann["class_id"],
                    "x": new_cx,
                    "y": new_cy,
                    "w": new_w,
                    "h": new_h
                })
        return result

    def get_annotations_map(
        self, dataset_id: str, version_id: str, image_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            data = self._load_registry()
            dataset = self._require_dataset(data, dataset_id)
            self._require_version(dataset, version_id)

            images = dataset.get("images", {})
            for image_id in image_ids:
                if image_id not in images:
                    raise KeyError("image not found")

            labels_dir = self._version_labels_dir(dataset_id, version_id)
            result = {}
            for image_id in image_ids:
                label_file = os.path.join(labels_dir, f"{image_id}.txt")
                result[image_id] = self._read_annotations_from_file(label_file)
            return result

    def save_annotations(
        self, dataset_id: str, version_id: str, image_id: str, annotations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        with self._lock:
            data = self._load_registry()
            dataset = self._require_dataset(data, dataset_id)
            self._require_version(dataset, version_id)
            if image_id not in dataset.get("images", {}):
                raise KeyError("image not found")

            class_ids = {c["id"] for c in dataset.get("classes", [])}
            lines = self._annotations_to_lines(annotations, class_ids)

            labels_dir = self._version_labels_dir(dataset_id, version_id)
            os.makedirs(labels_dir, exist_ok=True)
            label_file = os.path.join(labels_dir, f"{image_id}.txt")
            if lines:
                with open(label_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
            else:
                try:
                    os.remove(label_file)
                except FileNotFoundError:
                    pass

            dataset["annotation_versions"][version_id]["updated_at"] = _utc_now()
            dataset["updated_at"] = _utc_now()
            self._save_registry(data)
            return {"saved": len(lines)}

    def save_annotations_bulk(
        self,
        dataset_id: str,
        version_id: str,
        annotations_by_image: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        with self._lock:
            data = self._load_registry()
            dataset = self._require_dataset(data, dataset_id)
            self._require_version(dataset, version_id)

            images = dataset.get("images", {})
            class_ids = {c["id"] for c in dataset.get("classes", [])}

            lines_by_image = {}
            for image_id, annotations in annotations_by_image.items():
                if image_id not in images:
                    raise KeyError("image not found")
                lines_by_image[image_id] = self._annotations_to_lines(annotations, class_ids)

            labels_dir = self._version_labels_dir(dataset_id, version_id)
            os.makedirs(labels_dir, exist_ok=True)

            saved_boxes = 0
            for image_id, lines in lines_by_image.items():
                label_file = os.path.join(labels_dir, f"{image_id}.txt")
                if lines:
                    with open(label_file, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines) + "\n")
                else:
                    try:
                        os.remove(label_file)
                    except FileNotFoundError:
                        pass
                saved_boxes += len(lines)

            dataset["annotation_versions"][version_id]["updated_at"] = _utc_now()
            dataset["updated_at"] = _utc_now()
            self._save_registry(data)
            return {"saved_images": len(lines_by_image), "saved_boxes": saved_boxes}

    def annotation_stats(self, dataset_id: str, version_id: str) -> Dict[str, Any]:
        with self._lock:
            data = self._load_registry()
            dataset = self._require_dataset(data, dataset_id)
            self._require_version(dataset, version_id)
            labels_dir = self._version_labels_dir(dataset_id, version_id)
            class_counts = {}
            bbox_count = 0
            labeled_images = 0
            if os.path.exists(labels_dir):
                for name in os.listdir(labels_dir):
                    if not name.endswith(".txt"):
                        continue
                    path = os.path.join(labels_dir, name)
                    has_boxes = False
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split()
                            if not parts:
                                continue
                            cid = int(parts[0])
                            class_counts[cid] = class_counts.get(cid, 0) + 1
                            bbox_count += 1
                            has_boxes = True
                    if has_boxes:
                        labeled_images += 1
            return {
                "total_images": len(dataset.get("images", {})),
                "labeled_images": labeled_images,
                "bbox_count": bbox_count,
                "class_counts": class_counts,
            }

    def get_training_material(self, dataset_id: str, version_id: str) -> Dict[str, Any]:
        with self._lock:
            data = self._load_registry()
            dataset = self._require_dataset(data, dataset_id)
            self._require_version(dataset, version_id)
            stats = self.annotation_stats(dataset_id, version_id)
            classes = sorted(dataset.get("classes", []), key=lambda x: x["id"])
            images = list(dataset.get("images", {}).values())
            labels_dir = self._version_labels_dir(dataset_id, version_id)
            images_dir = self._dataset_images_dir(dataset_id)
            return {
                "dataset": dataset,
                "classes": classes,
                "images": images,
                "labels_dir": labels_dir,
                "images_dir": images_dir,
                "stats": stats,
            }

    def split_images(
        self,
        dataset_id: str,
        version_id: str,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> Dict[str, List[str]]:
        material = self.get_training_material(dataset_id, version_id)
        image_paths = []
        for image in material["images"]:
            abs_path = os.path.join(material["images_dir"], image["filename"])
            image_paths.append(abs_path)

        rng = random.Random(seed)
        rng.shuffle(image_paths)
        total = len(image_paths)
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)
        if n_train + n_val > total:
            n_val = max(0, total - n_train)
        train = image_paths[:n_train]
        val = image_paths[n_train : n_train + n_val]
        test = image_paths[n_train + n_val :]
        if total > 0 and not train:
            train = [image_paths[0]]
            val = image_paths[1:1]
            test = image_paths[1:]
        return {"train": train, "val": val, "test": test}

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            data = self._load_registry()
            jobs = list(data.get("jobs", {}).values())
            jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return jobs

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            data = self._load_registry()
            job = data.get("jobs", {}).get(job_id)
            if not job:
                raise KeyError("job not found")
            return job

    def get_active_job_id(self) -> Optional[str]:
        with self._lock:
            data = self._load_registry()
            return data.get("active_job_id")

    def set_active_job(self, job_id: Optional[str]) -> None:
        with self._lock:
            data = self._load_registry()
            data["active_job_id"] = job_id
            self._save_registry(data)

    def create_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            data = self._load_registry()
            job_id = job_data["id"]
            data["jobs"][job_id] = job_data
            data["active_job_id"] = job_id
            self._save_registry(data)
            return job_data

    def update_job(self, job_id: str, updates: Dict[str, Any], clear_active_if_terminal: bool = True) -> Dict[str, Any]:
        with self._lock:
            data = self._load_registry()
            job = data.get("jobs", {}).get(job_id)
            if not job:
                raise KeyError("job not found")
            job.update(updates)
            terminal = {"completed", "failed", "stopped", "interrupted"}
            if clear_active_if_terminal and data.get("active_job_id") == job_id and job.get("status") in terminal:
                data["active_job_id"] = None
            self._save_registry(data)
            return job
