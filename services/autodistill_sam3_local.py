import os
import importlib
from typing import Any, Dict, List

import numpy as np
import torch


class Sam3LocalProvider:
    """
    Local SAM3 provider that mimics the minimal `predict()` contract used by
    GroundedSam2AutodistillService._extract_detection_rows.
    """

    def __init__(self, ontology, box_threshold: float = 0.35, text_threshold: float = 0.25):
        self.ontology = ontology
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)

        from PIL import Image  # pylint: disable=import-outside-toplevel
        import transformers  # pylint: disable=import-outside-toplevel

        self._Image = Image
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = (os.getenv("SAM3_MODEL_ID") or "facebook/sam3").strip()

        Sam3Model = getattr(transformers, "Sam3Model", None)
        Sam3Processor = getattr(transformers, "Sam3Processor", None)
        if Sam3Model is None or Sam3Processor is None:
            try:
                Sam3Model = importlib.import_module(
                    "transformers.models.sam3.modeling_sam3"
                ).Sam3Model
                Sam3Processor = importlib.import_module(
                    "transformers.models.sam3.processing_sam3"
                ).Sam3Processor
            except Exception as exc:
                version = getattr(transformers, "__version__", "unknown")
                raise RuntimeError(
                    "SAM3 classes are unavailable in the installed transformers package "
                    f"(version={version}). Install a newer transformers build that includes "
                    "Sam3Model/Sam3Processor."
                ) from exc

        self._processor = Sam3Processor.from_pretrained(model_id)
        self._model = Sam3Model.from_pretrained(model_id).to(self._device)
        self._model.eval()

    @staticmethod
    def _to_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        if hasattr(value, "tolist"):
            return value.tolist()
        return list(value)

    @staticmethod
    def _bbox_from_mask(mask_2d: np.ndarray) -> List[float]:
        ys, xs = np.where(mask_2d)
        if xs.size == 0 or ys.size == 0:
            return []
        x1 = float(xs.min())
        y1 = float(ys.min())
        x2 = float(xs.max())
        y2 = float(ys.max())
        if x2 <= x1 or y2 <= y1:
            return []
        return [x1, y1, x2, y2]

    def _predict_boxes_for_prompt(self, image, prompt: str, target_size) -> List[List[float]]:
        inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        processed = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=max(0.0, min(1.0, self.box_threshold)),
            mask_threshold=max(0.0, min(1.0, self.text_threshold)),
            target_sizes=[target_size],
        )
        if not processed:
            return []

        result = processed[0]
        seg = result.get("segmentation")
        segments_info = result.get("segments_info") or []
        if seg is None or not segments_info:
            return []

        seg_np = np.array(seg)
        boxes: List[List[float]] = []
        for info in segments_info:
            seg_id = info.get("id")
            if seg_id is None:
                continue
            bbox = self._bbox_from_mask(seg_np == int(seg_id))
            if bbox:
                boxes.append(bbox)
        return boxes

    def predict(self, input: Any) -> Dict[str, Any]:
        image = self._Image.open(input).convert("RGB")
        target_size = (image.height, image.width)

        xyxy_rows: List[List[float]] = []
        class_ids: List[int] = []
        class_names: List[str] = []

        prompts = list(self.ontology.prompts())
        for idx, prompt in enumerate(prompts):
            boxes = self._predict_boxes_for_prompt(image=image, prompt=str(prompt), target_size=target_size)
            for b in boxes:
                xyxy_rows.append(b)
                class_ids.append(idx)
                class_names.append(str(prompt))

        return {
            "xyxy": xyxy_rows,
            "class_id": class_ids,
            "data": {
                "class_name": class_names
            },
        }
