import importlib
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2


class _GroundingDinoOnlyProvider:
    def __init__(self, ontology, box_threshold: float , text_threshold: float):
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


class GroundedSam2AutodistillService:
    DEFAULT_PROVIDER = "dino"

    def __init__(self):
        self._lock = threading.RLock()
        self._model_cache = {}
        self._caption_ontology_cls = None

    def availability(self) -> Dict[str, object]:
        try:
            self._ensure_caption_ontology()
            provider_cls, provider_name = self._resolve_provider(self.DEFAULT_PROVIDER)
            return {
                "available": True,
                "provider": provider_name,
                "default_provider": self.DEFAULT_PROVIDER,
                "providers": ["dino", "grounded_sam", "grounded_sam2", "auto"],
            }
        except Exception as exc:
            return {"available": False, "error": str(exc)}

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

        self._ensure_caption_ontology()
        provider_cls, provider_name = self._resolve_provider(provider)
        model = self._get_model(
            provider_cls=provider_cls,
            provider_name=provider_name,
            ontology_map=ontology_map,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            allow_fallback=(provider == "auto"),
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
        normalized = (provider or self.DEFAULT_PROVIDER).strip().lower()
        if normalized not in {"dino", "grounded_sam", "grounded_sam2", "auto"}:
            raise ValueError("provider must be one of: dino, grounded_sam, grounded_sam2, auto")

        if normalized == "dino":
            try:
                importlib.import_module("autodistill_grounded_sam.helpers")
                return _GroundingDinoOnlyProvider, "autodistill_grounding_dino.local_adapter"
            except Exception as exc:
                raise RuntimeError(f"GroundingDINO provider is unavailable: {exc}") from exc

        if normalized == "grounded_sam":
            return self._resolve_first([
                ("autodistill_grounded_sam", "GroundedSAM"),
            ], "GroundedSAM provider is unavailable")

        if normalized == "grounded_sam2":
            return self._resolve_first([
                ("autodistill_grounded_sam_2", "GroundedSAM2"),
                ("autodistill_grounded_sam2", "GroundedSAM2"),
            ], "GroundedSAM2 provider is unavailable")

        # auto
        return self._resolve_first([
            ("autodistill_grounded_sam", "GroundedSAM"),
            ("autodistill_grounded_sam_2", "GroundedSAM2"),
            ("autodistill_grounded_sam2", "GroundedSAM2"),
        ], "no provider available for auto", include_dino_fallback=True)

    def _resolve_first(self, candidates, not_found_error: str, include_dino_fallback: bool = False):
        errors = []
        for module_name, class_name in candidates:
            try:
                module = importlib.import_module(module_name)
                provider_cls = getattr(module, class_name, None)
                if provider_cls is None:
                    continue
                return provider_cls, f"{module_name}.{class_name}"
            except Exception as exc:
                errors.append(f"{module_name}.{class_name}: {exc}")
        if include_dino_fallback:
            try:
                importlib.import_module("autodistill_grounded_sam.helpers")
                return _GroundingDinoOnlyProvider, "autodistill_grounding_dino.local_adapter"
            except Exception as exc:
                errors.append(f"autodistill_grounding_dino.local_adapter: {exc}")
        details = "; ".join(errors[:3]) if errors else "module/class not found"
        raise RuntimeError(f"{not_found_error} ({details})")

    def _get_model(
        self,
        provider_cls,
        provider_name: str,
        ontology_map: Dict[str, str],
        box_threshold: float,
        text_threshold: float,
        allow_fallback: bool,
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
                allow_fallback=allow_fallback,
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
        allow_fallback: bool,
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
                return provider_cls(**kwargs)
            except TypeError as exc:
                last_exc = exc
                continue
            except Exception as exc:
                # GroundedSAM2 often fails at runtime on missing optional deps (e.g. flash_attn).
                # Try a stable fallback provider when available.
                if allow_fallback:
                    fallback_model = self._try_groundedsam_fallback(
                        ontology_map, box_threshold, text_threshold
                    )
                    if fallback_model is not None:
                        return fallback_model
                message = str(exc)
                if "flash_attn" in message:
                    raise RuntimeError(
                        "GroundedSAM2 failed because 'flash_attn' is missing in this environment. "
                        "On Windows this is commonly unsupported; use GroundedSAM fallback or run "
                        "GroundedSAM2 in a Linux/CUDA environment."
                    ) from exc
                raise RuntimeError(f"failed to initialize provider {provider_name}: {exc}") from exc
        if last_exc:
            raise RuntimeError(f"failed to initialize provider {provider_name}: {last_exc}") from last_exc
        raise RuntimeError("failed to initialize grounded sam provider")

    def _try_groundedsam_fallback(
        self,
        ontology_map: Dict[str, str],
        box_threshold: float,
        text_threshold: float,
    ):
        try:
            module = importlib.import_module("autodistill_grounded_sam")
            fallback_cls = getattr(module, "GroundedSAM", None)
            if fallback_cls is None:
                return None
        except Exception:
            return None

        ontology = self._caption_ontology_cls(ontology_map)
        init_variants = [
            {"ontology": ontology, "box_threshold": box_threshold, "text_threshold": text_threshold},
            {"ontology": ontology, "box_threshold": box_threshold},
            {"ontology": ontology},
        ]
        for kwargs in init_variants:
            try:
                return fallback_cls(**kwargs)
            except TypeError:
                continue
            except Exception:
                continue
        return None

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
