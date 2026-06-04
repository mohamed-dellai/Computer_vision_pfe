import os
import shutil
from importlib import import_module
from pathlib import Path

from werkzeug.utils import secure_filename


EXPORT_FORMATS = {
    "onnx": {
        "ultralytics_format": "onnx",
        "label": "ONNX",
    },
    "engine": {
        "ultralytics_format": "engine",
        "label": "TensorRT",
    },
}

TARGET_ALIASES = {
    "onnx": "onnx",
    "engine": "engine",
    "tensorrt": "engine",
    "trt": "engine",
}


def normalize_export_target(target):
    normalized = TARGET_ALIASES.get(str(target or "").strip().lower())
    if not normalized:
        supported = ", ".join(sorted(TARGET_ALIASES))
        raise ValueError(f"unsupported target. choose one of: {supported}")
    return normalized


class ModelOptimizer:
    def __init__(self, models_dir):
        self.models_dir = models_dir

    def optimize(self, source, target, imgsz, opset=12, dynamic=False, half=False, simplify=False, device=None):
        src_path = os.path.join(self.models_dir, secure_filename(source))
        if not os.path.exists(src_path):
            raise FileNotFoundError("source model not found")
        if Path(src_path).suffix.lower() != ".pt":
            raise ValueError("Ultralytics export requires a .pt source model")

        target = normalize_export_target(target)
        export_cfg = EXPORT_FORMATS[target]
        model = self._load_yolo_class()(src_path)

        export_args = {
            "format": export_cfg["ultralytics_format"],
            "imgsz": imgsz,
        }
        if device not in (None, ""):
            export_args["device"] = device

        if target == "onnx":
            export_args.update({
                "opset": opset,
                "dynamic": dynamic,
                "simplify": simplify,
            })
            if half:
                export_args["half"] = True
        elif target == "engine":
            self._require_tensorrt()
            export_args.update({
                "dynamic": dynamic,
                "half": half,
            })

        out_path = model.export(**export_args)
        if not isinstance(out_path, str) or not os.path.exists(out_path):
            raise RuntimeError("export did not produce a file")

        dest_path = self._move_export_to_models_dir(out_path)
        return {
            "filename": os.path.basename(dest_path),
            "format": target,
            "format_label": export_cfg["label"],
            "path": dest_path,
        }

    def _load_yolo_class(self):
        try:
            ultralytics = import_module("ultralytics")
            return getattr(ultralytics, "YOLO")
        except Exception as exc:
            raise RuntimeError("Ultralytics is not installed or cannot be imported") from exc

    def _require_tensorrt(self):
        try:
            tensorrt = import_module("tensorrt")
        except Exception as exc:
            raise RuntimeError(
                "TensorRT export requires NVIDIA TensorRT, CUDA, and a compatible NVIDIA GPU. "
                "Install TensorRT or choose ONNX."
            ) from exc
        if not hasattr(tensorrt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
            version = getattr(tensorrt, "__version__", "unknown")
            raise RuntimeError(
                f"TensorRT {version} is not compatible with the current Ultralytics exporter. "
                "Install a TensorRT 10.x CUDA 12 package, for example: tensorrt-cu12<11."
            )

    def _move_export_to_models_dir(self, out_path):
        out_path = os.path.abspath(out_path)
        models_dir = os.path.abspath(self.models_dir)
        dest_path = os.path.join(models_dir, os.path.basename(out_path))

        if os.path.abspath(os.path.dirname(out_path)) == models_dir:
            return out_path

        os.makedirs(models_dir, exist_ok=True)
        if os.path.isdir(out_path):
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            shutil.move(out_path, dest_path)
            return dest_path

        if os.path.exists(dest_path):
            os.remove(dest_path)
        shutil.move(out_path, dest_path)
        return dest_path
