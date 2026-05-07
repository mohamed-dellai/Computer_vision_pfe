import importlib
import importlib.util
import os
import threading
import traceback
from typing import Any, Dict, List, Optional, Tuple

import cv2


PROVIDER_LABELS = {
    "owlvit": "OWL-ViT",
    "dino": "GroundingDINO",
    "yolo_world": "YOLO-World",
    "florence2": "Florence-2",
}

DEFAULT_YOLO_WORLD_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "yolov8x-worldv2.pt")
)


def _resolve_yolo_world_model_path() -> str:
    return os.getenv("AUTODISTILL_YOLO_WORLD_MODEL", DEFAULT_YOLO_WORLD_MODEL_PATH).strip()


def _empty_detections() -> Dict[str, object]:
    return {"xyxy": [], "class_id": [], "data": {"class_name": []}}


class _GroundingDinoOnlyProvider:
    def __init__(self, ontology, box_threshold: float, text_threshold: float):
        self.ontology = ontology
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        helpers = importlib.import_module("autodistill_grounded_sam.helpers")
        self._load_grounding_dino = getattr(helpers, "load_grounding_dino")
        self._combine_detections = getattr(helpers, "combine_detections")
        autodistill_helpers = importlib.import_module("autodistill.helpers")
        self._load_image = getattr(autodistill_helpers, "load_image")
        self._dino_model = self._load_grounding_dino()

    def predict(self, input: Any):
        image = self._load_image(input, return_format="cv2")
        detections_list = []
        for description in self.ontology.prompts():
            detections = self._dino_model.predict_with_classes(
                image=image,
                classes=[description],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
            detections_list.append(detections)
        return self._combine_detections(
            detections_list, overwrite_class_ids=range(len(detections_list))
        )


class _OwlVitProvider:
    def __init__(self, ontology, box_threshold: float = 0.35, text_threshold: float = 0.25):
        self.ontology = ontology
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        transformers = importlib.import_module("transformers")
        autodistill_helpers = importlib.import_module("autodistill.helpers")
        self._load_image = getattr(autodistill_helpers, "load_image")

        # create zero-shot object detection pipeline for OWL-ViT
        pipeline = getattr(transformers, "pipeline")
        model_id = os.getenv("AUTODISTILL_OWL_MODEL", "google/owlvit-base-patch32").strip()
        device_env = os.getenv("AUTODISTILL_DEVICE", "").strip()
        if device_env:
            # transformers pipeline expects int device id for GPU, -1 for CPU
            try:
                device = int(device_env)
            except Exception:
                device = 0 if device_env.startswith("cuda") or device_env.startswith("gpu") else -1
        else:
            # default: use GPU 0 if available via torch, otherwise CPU
            try:
                torch = importlib.import_module("torch")
                device = 0 if torch.cuda.is_available() else -1
            except Exception:
                device = -1

        try:
            print(f"[OWL-ViT] initializing pipeline model={model_id} device={device}")
            self._detector = pipeline(
                "zero-shot-object-detection",
                model=model_id,
                device=device,
            )
            print(f"[OWL-ViT] pipeline initialized: {self._detector}")
        except Exception as e:
            print(f"[OWL-ViT] failed to initialize pipeline: {e}")
            import traceback as _tb
            _tb.print_exc()
            raise

    def predict(self, input: Any):
        image = self._load_image(input, return_format="PIL")
        prompts = list(self.ontology.prompts())

        try:
            w, h = (None, None)
            try:
                w, h = image.size
            except Exception:
                pass
            print(f"[OWL-ViT] predict called image_size={(w,h)} prompts_count={len(prompts)} prompts_sample={prompts[:5]}")

            # run detector once with all candidate labels for efficiency
            results = self._detector(image, prompts)
            print(f"[OWL-ViT] raw results type={type(results)} len={len(results) if hasattr(results, '__len__') else 'n/a'}")

            # results: list of detections with keys 'score','label','box'
            xyxy = []
            class_ids = []
            class_names = []

            # image size for possible normalized boxes
            if w is None or h is None:
                try:
                    w, h = image.size
                except Exception:
                    w, h = (0, 0)

            for i, det in enumerate(results):
                try:
                    print(f"[OWL-ViT] det[{i}] = {det}")
                except Exception:
                    print(f"[OWL-ViT] det[{i}] (unprintable)")

                score = float(det.get("score", 0.0))
                label = det.get("label")
                box = det.get("box") or det.get("bbox")

                if label is None or box is None:
                    print(f"[OWL-ViT] det[{i}] missing label or box; skipping")
                    continue

                if score < self.box_threshold:
                    print(f"[OWL-ViT] det[{i}] score={score} below threshold={self.box_threshold}; skipping")
                    continue

                # box may be [x0,y0,x1,y1] in absolute pixels or normalized [0..1]
                try:
                    bx = [float(v) for v in box]
                except Exception as e:
                    print(f"[OWL-ViT] det[{i}] invalid box format {box}: {e}")
                    continue
                if max(bx) <= 1.01 and w > 0 and h > 0:
                    # normalized coordinates
                    x0 = bx[0] * w
                    y0 = bx[1] * h
                    x1 = bx[2] * w
                    y1 = bx[3] * h
                else:
                    x0, y0, x1, y1 = bx[0], bx[1], bx[2], bx[3]

                print(f"[OWL-ViT] det[{i}] -> box={(x0,y0,x1,y1)} label={label} score={score}")

                xyxy.append([x0, y0, x1, y1])

                # resolve class id as index in prompts
                try:
                    class_idx = prompts.index(label)
                except ValueError:
                    class_idx = None

                class_ids.append(class_idx)
                class_names.append(label)

            if not xyxy:
                print("[OWL-ViT] no detections passed thresholds")
                return {"xyxy": [], "class_id": [], "data": {"class_name": []}}

            print(f"[OWL-ViT] returning {len(xyxy)} boxes")
            return {"xyxy": xyxy, "class_id": class_ids, "data": {"class_name": class_names}}
        except Exception as exc:
            print(f"[OWL-ViT] predict exception: {exc}")
            import traceback as _tb
            _tb.print_exc()
            raise


class _YoloWorldProvider:
    def __init__(self, ontology, box_threshold: float = 0.35, text_threshold: float = 0.25):
        self.ontology = ontology
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        ultralytics = importlib.import_module("ultralytics")
        model_cls = getattr(ultralytics, "YOLO")
        model_path = _resolve_yolo_world_model_path()
        self._model = model_cls(model_path)
        self._prompts = list(ontology.prompts())
        self._model.set_classes(self._prompts)
        self._device = os.getenv("AUTODISTILL_DEVICE", "").strip() or None

    def predict(self, input: Any):
        kwargs = {
            "conf": self.box_threshold,
            "verbose": False,
        }
        if self._device:
            kwargs["device"] = self._device

        results = self._model.predict(input, **kwargs)
        if not results:
            return _empty_detections()

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return _empty_detections()

        class_ids = []
        raw_cls = getattr(boxes, "cls", None)
        if raw_cls is not None:
            if hasattr(raw_cls, "detach"):
                raw_cls = raw_cls.detach().cpu()
            if hasattr(raw_cls, "tolist"):
                raw_cls = raw_cls.tolist()
            class_ids = [int(value) for value in raw_cls]

        class_names = [
            self._prompts[class_id] if 0 <= class_id < len(self._prompts) else str(class_id)
            for class_id in class_ids
        ]

        xyxy = boxes.xyxy
        if hasattr(xyxy, "detach"):
            xyxy = xyxy.detach().cpu()
        if hasattr(xyxy, "tolist"):
            xyxy = xyxy.tolist()

        return {
            "xyxy": xyxy,
            "class_id": class_ids,
            "data": {"class_name": class_names},
        }


class _Florence2Provider:
    TASK_PROMPT = "<CAPTION_TO_PHRASE_GROUNDING>"

    def __init__(self, ontology, box_threshold: float = 0.35, text_threshold: float = 0.25):
        self.ontology = ontology
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._prompts = list(ontology.prompts())
        self._task_prompt = os.getenv("AUTODISTILL_FLORENCE2_TASK", self.TASK_PROMPT).strip()

        torch = importlib.import_module("torch")
        pil_image = importlib.import_module("PIL.Image")
        transformers = importlib.import_module("transformers")
        auto_processor_cls = getattr(transformers, "AutoProcessor")
        auto_model_cls = getattr(transformers, "AutoModelForCausalLM")

        self._torch = torch
        self._image_cls = getattr(pil_image, "Image")
        self._model_id = os.getenv("AUTODISTILL_FLORENCE2_MODEL", "microsoft/Florence-2-base-ft").strip()
        self._device = os.getenv("AUTODISTILL_DEVICE", "").strip()
        if not self._device:
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._torch_dtype = torch.float16 if self._device.startswith("cuda") else torch.float32

        self._processor = auto_processor_cls.from_pretrained(
            self._model_id,
            trust_remote_code=True,
        )
        self._model = auto_model_cls.from_pretrained(
            self._model_id,
            torch_dtype=self._torch_dtype,
            trust_remote_code=True,
        ).to(self._device)
        self._model.eval()

    def predict(self, input: Any):
        image = self._load_image(input)
        xyxy_rows = []
        class_ids = []
        class_names = []

        for class_index, prompt in enumerate(self._prompts):
            prompt_boxes = self._predict_prompt(image, prompt)
            if not prompt_boxes:
                continue
            xyxy_rows.extend(prompt_boxes)
            class_ids.extend([class_index] * len(prompt_boxes))
            class_names.extend([prompt] * len(prompt_boxes))

        if not xyxy_rows:
            return _empty_detections()

        return {
            "xyxy": xyxy_rows,
            "class_id": class_ids,
            "data": {"class_name": class_names},
        }

    def _load_image(self, input: Any):
        if isinstance(input, self._image_cls):
            return input.convert("RGB")

        pil_image = importlib.import_module("PIL.Image")
        image_module = getattr(pil_image, "open")
        return image_module(input).convert("RGB")

    def _predict_prompt(self, image, prompt: str) -> List[List[float]]:
        task_prompt = self._task_prompt
        text_input = "" if task_prompt == "<OD>" else prompt
        full_prompt = task_prompt + text_input

        inputs = self._processor(text=full_prompt, images=image, return_tensors="pt")
        inputs = inputs.to(self._device, self._torch_dtype)

        with self._torch.inference_mode():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
            )

        generated_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]
        parsed = self._processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height),
        )
        result = parsed.get(task_prompt, parsed)
        bboxes = result.get("bboxes", [])
        return [list(map(float, box)) for box in bboxes]


class GroundedSam2AutodistillService:
    DEFAULT_PROVIDER = "owlvit"
    PROVIDER_ALIASES = {
        "dino": "dino",
        "groundingdino": "dino",
        "grounding_dino": "dino",
        "yolo-world": "yolo_world",
        "yolo_world": "yolo_world",
        "yoloworld": "yolo_world",
        "florence": "florence2",
        "florence2": "florence2",
        "florence-2": "florence2",
        "florence_2": "florence2",
        "owl": "owlvit",
        "owlvit": "owlvit",
        "owl-vit": "owlvit",
        "owl_vit": "owlvit",
    }

    def __init__(self):
        self._lock = threading.RLock()
        self._model_cache = {}
        self._caption_ontology_cls = None

    @staticmethod
    def _debug(message: str):
        print(f"[AUTODISTILL DEBUG] {message}")

    def availability(self) -> Dict[str, object]:
        try:
            providers = []
            provider_errors = {}
            default_provider_name = ""

            for provider_id in PROVIDER_LABELS:
                try:
                    provider_name = self._check_provider_available(provider_id)
                    providers.append(provider_id)
                    if provider_id == self.DEFAULT_PROVIDER:
                        default_provider_name = provider_name
                except Exception as exc:
                    provider_errors[provider_id] = str(exc)

            if self.DEFAULT_PROVIDER not in providers:
                error = provider_errors.get(self.DEFAULT_PROVIDER, "default provider is unavailable")
                return {"available": False, "error": error, "provider_errors": provider_errors}

            return {
                "available": True,
                "provider": default_provider_name,
                "default_provider": self.DEFAULT_PROVIDER,
                "providers": providers,
                "provider_labels": PROVIDER_LABELS,
                "provider_errors": provider_errors,
            }
        except Exception as exc:
            return {"available": False, "error": str(exc)}

    def _check_provider_available(self, provider: str) -> str:
        normalized = self.PROVIDER_ALIASES.get(provider, provider)

        if normalized == "dino":
            if importlib.util.find_spec("autodistill_grounded_sam") is None:
                raise RuntimeError("GroundingDINO provider is unavailable: autodistill-grounded-sam is not installed")
            return "autodistill_grounding_dino.local_adapter"

        if normalized == "owlvit":
            missing = [
                module_name
                for module_name in ("transformers",)
                if importlib.util.find_spec(module_name) is None
            ]
            if missing:
                raise RuntimeError(f"OWL-ViT provider is unavailable: missing {', '.join(missing)}")
            return "transformers_owlvit.local_adapter"

        if normalized == "yolo_world":
            if importlib.util.find_spec("ultralytics") is None:
                raise RuntimeError("YOLO-World provider is unavailable: ultralytics is not installed")
            model_path = _resolve_yolo_world_model_path()
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"YOLO-World model not found: {model_path}")
            return "ultralytics_yolo_world.local_adapter"

        if normalized == "florence2":
            missing = [
                module_name
                for module_name in ("torch", "PIL", "einops", "timm", "transformers")
                if importlib.util.find_spec(module_name) is None
            ]
            if missing:
                raise RuntimeError(f"Florence-2 provider is unavailable: missing {', '.join(missing)}")
            return "transformers_florence2.local_adapter"

        allowed = ", ".join(PROVIDER_LABELS.keys())
        raise ValueError(f"provider must be one of: {allowed}")

    def run(
        self,
        image_items: List[Dict[str, str]],
        ontology_map: Dict[str, str],
        ontology_class_ids: List[int],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        provider: str = DEFAULT_PROVIDER,
    ) -> Dict[str, object]:
        if not image_items:
            return {
                "processed_images": 0,
                "labeled_images": 0,
                "predicted_boxes": 0,
                "annotations_by_image": {},
                "errors": [],
            }

        self._debug(
            f"run called with provider={provider}, images={len(image_items)}, "
            f"ontology_terms={len(ontology_map)}, class_ids={len(ontology_class_ids)}"
        )
        self._ensure_caption_ontology()
        provider_cls, provider_name = self._resolve_provider(provider)
        self._debug(f"resolved provider -> {provider_name} ({provider_cls.__name__})")
        model = self._get_model(
            provider_cls=provider_cls,
            provider_name=provider_name,
            ontology_map=ontology_map,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        annotations_by_image = {}
        errors = []
        predicted_boxes = 0
        labeled_images = 0

        for item in image_items:
            image_id = item["id"]
            image_path = item["path"]
            try:
                annotations = self._predict_image_annotations(
                    model=model,
                    image_path=image_path,
                    ontology_class_ids=ontology_class_ids,
                    ontology_map=ontology_map,
                )
                annotations_by_image[image_id] = annotations
                predicted_boxes += len(annotations)
                if annotations:
                    labeled_images += 1
            except Exception as exc:
                print(f"=========================================")
                print(f"AUTODISTILL PREDICTION ERROR on image {image_id}:")
                print(f"{type(exc).__name__}: {exc}")
                import traceback
                traceback.print_exc()
                print(f"=========================================")
                annotations_by_image[image_id] = []
                errors.append({"image_id": image_id, "error": str(exc)})

        return {
            "processed_images": len(image_items),
            "labeled_images": labeled_images,
            "predicted_boxes": predicted_boxes,
            "annotations_by_image": annotations_by_image,
            "errors": errors,
            "provider": provider_name,
        }

    def _ensure_caption_ontology(self):
        if self._caption_ontology_cls is not None:
            return

        with self._lock:
            if self._caption_ontology_cls is not None:
                return

            try:
                detection_module = importlib.import_module("autodistill.detection")
                self._caption_ontology_cls = getattr(detection_module, "CaptionOntology")
            except Exception as exc:
                raise RuntimeError(
                    "failed to import autodistill.detection. "
                    "Install or repair autodistill with: pip install -U autodistill "
                    f"(details: {exc})"
                ) from exc

    def _resolve_provider(self, provider: str):
        raw_provider = (provider or self.DEFAULT_PROVIDER).strip().lower()
        normalized = self.PROVIDER_ALIASES.get(raw_provider, raw_provider)
        self._debug(f"resolving provider '{normalized}'")

        if normalized == "dino":
            try:
                importlib.import_module("autodistill_grounded_sam.helpers")
                return _GroundingDinoOnlyProvider, "autodistill_grounding_dino.local_adapter"
            except Exception as exc:
                raise RuntimeError(f"GroundingDINO provider is unavailable: {exc}") from exc

        if normalized == "owlvit":
            try:
                transformers = importlib.import_module("transformers")
                getattr(transformers, "pipeline")
                return _OwlVitProvider, "transformers_owlvit.local_adapter"
            except Exception as exc:
                raise RuntimeError(f"OWL-ViT provider is unavailable: {exc}") from exc

        if normalized == "yolo_world":
            try:
                ultralytics = importlib.import_module("ultralytics")
                getattr(ultralytics, "YOLO")
                model_path = _resolve_yolo_world_model_path()
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"YOLO-World model not found: {model_path}")
                return _YoloWorldProvider, "ultralytics_yolo_world.local_adapter"
            except Exception as exc:
                raise RuntimeError(
                    "YOLO-World provider is unavailable. Install or repair ultralytics, "
                    "then provide a YOLO-World .pt model such as yolov8x-worldv2.pt "
                    f"(details: {exc})"
                ) from exc

        if normalized == "florence2":
            try:
                importlib.import_module("torch")
                importlib.import_module("PIL.Image")
                importlib.import_module("einops")
                importlib.import_module("timm")
                transformers = importlib.import_module("transformers")
                getattr(transformers, "AutoProcessor")
                getattr(transformers, "AutoModelForCausalLM")
                return _Florence2Provider, "transformers_florence2.local_adapter"
            except Exception as exc:
                raise RuntimeError(
                    "Florence-2 provider is unavailable. Install required packages with: "
                    "pip install timm einops "
                    f"(details: {exc})"
                ) from exc

        allowed = ", ".join(PROVIDER_LABELS.keys())
        raise ValueError(f"provider must be one of: {allowed}")

    def _get_model(
        self,
        provider_cls,
        provider_name: str,
        ontology_map: Dict[str, str],
        box_threshold: float,
        text_threshold: float,
    ):
        key = (
            provider_name,
            tuple(ontology_map.items()),
            round(box_threshold, 4),
            round(text_threshold, 4),
        )
        with self._lock:
            model = self._model_cache.get(key)
            if model is not None:
                return model
            model = self._build_model(
                provider_cls=provider_cls,
                provider_name=provider_name,
                ontology_map=ontology_map,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            self._model_cache[key] = model
            return model

    def _build_model(
        self,
        provider_cls,
        provider_name: str,
        ontology_map: Dict[str, str],
        box_threshold: float,
        text_threshold: float,
    ):
        ontology = self._caption_ontology_cls(ontology_map)

        init_variants = [
            {"ontology": ontology, "box_threshold": box_threshold, "text_threshold": text_threshold},
            {"ontology": ontology, "box_threshold": box_threshold},
            {"ontology": ontology},
        ]

        last_exc = None
        for kwargs in init_variants:
            try:
                self._debug(f"initializing provider {provider_name} with args={list(kwargs.keys())}")
                return provider_cls(**kwargs)
            except TypeError as exc:
                self._debug(
                    f"provider init TypeError for {provider_name} with args={list(kwargs.keys())}: {exc}"
                )
                last_exc = exc
                continue
            except Exception as exc:
                self._debug(
                    f"provider init failed for {provider_name} with args={list(kwargs.keys())}: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._debug(traceback.format_exc())
                raise RuntimeError(f"failed to initialize provider {provider_name}: {exc}") from exc
        if last_exc:
            raise RuntimeError(f"failed to initialize provider {provider_name}: {last_exc}") from last_exc
        raise RuntimeError("failed to initialize grounding dino provider")

    def _predict_image_annotations(
        self,
        model,
        image_path: str,
        ontology_class_ids: List[int],
        ontology_map: Dict[str, str],
    ) -> List[Dict[str, float]]:
        detections = model.predict(image_path)
        xyxy_rows, class_idx_rows, class_name_rows = self._extract_detection_rows(detections)
        if not xyxy_rows:
            return []

        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError("failed to read image for bbox normalization")
        image_h, image_w = image.shape[:2]
        if image_w <= 0 or image_h <= 0:
            raise RuntimeError("invalid image dimensions")

        class_lookup = {name.lower(): cid for name, cid in zip(ontology_map.keys(), ontology_class_ids)}

        annotations = []
        for i, row in enumerate(xyxy_rows):
            class_id = self._resolve_class_id(i, class_idx_rows, class_name_rows, ontology_class_ids, class_lookup)
            if class_id is None:
                continue

            x1, y1, x2, y2 = [float(v) for v in row]
            x1 = max(0.0, min(x1, float(image_w - 1)))
            y1 = max(0.0, min(y1, float(image_h - 1)))
            x2 = max(0.0, min(x2, float(image_w - 1)))
            y2 = max(0.0, min(y2, float(image_h - 1)))

            if x2 <= x1 or y2 <= y1:
                continue

            box_w = x2 - x1
            box_h = y2 - y1
            x_center = x1 + (box_w / 2.0)
            y_center = y1 + (box_h / 2.0)

            annotations.append({
                "class_id": int(class_id),
                "x": self._clamp01(x_center / image_w),
                "y": self._clamp01(y_center / image_h),
                "w": self._clamp01(box_w / image_w),
                "h": self._clamp01(box_h / image_h),
            })

        return annotations

    def _extract_detection_rows(self, detections) -> Tuple[List[List[float]], List[Optional[int]], List[Optional[str]]]:
        obj = detections
        if hasattr(obj, "detections"):
            obj = getattr(obj, "detections")

        xyxy = self._get_attr_or_item(obj, "xyxy")
        if xyxy is None:
            return [], [], []
        if hasattr(xyxy, "tolist"):
            xyxy = xyxy.tolist()
        xyxy_rows = [list(row) for row in xyxy]

        class_idx = self._get_attr_or_item(obj, "class_id")
        if class_idx is not None and hasattr(class_idx, "tolist"):
            class_idx = class_idx.tolist()
        if class_idx is None:
            class_idx_rows = [None] * len(xyxy_rows)
        else:
            class_idx_rows = [None if v is None else int(v) for v in class_idx]
            if len(class_idx_rows) < len(xyxy_rows):
                class_idx_rows.extend([None] * (len(xyxy_rows) - len(class_idx_rows)))

        class_names = [None] * len(xyxy_rows)
        data = self._get_attr_or_item(obj, "data")
        if isinstance(data, dict):
            names = data.get("class_name")
            if names is not None:
                if hasattr(names, "tolist"):
                    names = names.tolist()
                for i, value in enumerate(names[: len(class_names)]):
                    class_names[i] = str(value)

        return xyxy_rows, class_idx_rows, class_names

    @staticmethod
    def _get_attr_or_item(obj, key):
        if hasattr(obj, key):
            return getattr(obj, key)
        if isinstance(obj, dict):
            return obj.get(key)
        return None

    @staticmethod
    def _resolve_class_id(
        index: int,
        class_idx_rows: List[Optional[int]],
        class_name_rows: List[Optional[str]],
        ontology_class_ids: List[int],
        class_lookup: Dict[str, int],
    ) -> Optional[int]:
        idx = class_idx_rows[index] if index < len(class_idx_rows) else None
        if idx is not None and 0 <= idx < len(ontology_class_ids):
            return ontology_class_ids[idx]

        name = class_name_rows[index] if index < len(class_name_rows) else None
        if name:
            return class_lookup.get(name.strip().lower())

        if len(ontology_class_ids) == 1:
            return ontology_class_ids[0]
        return None

    @staticmethod
    def _clamp01(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value
