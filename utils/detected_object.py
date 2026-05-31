import os
import json
import threading
from datetime import datetime
from typing import Optional, Dict

import cv2
import numpy as np
import requests

import integration_config

class DetectedObject:
    def __init__(self, class_name: str, object_id: int, detection_time: datetime, location: list, image: np.ndarray, inOrOut: str, confidence: float):
        self.class_name = class_name
        self.object_id = object_id
        self.detection_time = detection_time
        self.location = location  # [x, y, w, h]
        self.image = image
        self.inOrOut = inOrOut
        self.confidence = confidence

    def __repr__(self):
        return f"DetectedObject(class_name={self.class_name}, object_id={self.object_id}, detection_time={self.detection_time}, location={self.location}, inOrOut={self.inOrOut}, confidence={self.confidence})"

    def post_event(
        self,
        endpoint: Optional[str] = None,
        external_id: Optional[str] = None,
        name: Optional[str] = None,
        timeout: float = 3.0,
        extra_fields: Optional[Dict[str, object]] = None,
    ) -> None:
        active = integration_config.active_integration()
        url = endpoint or (active or {}).get("api_url") or os.getenv("MENU_OBJECT", "")
        if not url:
            return
            
        if active:
            # Check class filter
            filter_classes_str = active.get("filter_classes", "")
            if filter_classes_str:
                allowed_classes = [c.strip() for c in filter_classes_str.split(",") if c.strip()]
                if allowed_classes and self.class_name not in allowed_classes:
                    return
            
            # Check event filter
            filter_events = active.get("filter_events", [])
            if filter_events and self.inOrOut not in filter_events:
                return
                
            # Check image toggle
            if not active.get("include_image"):
                self.image = None

        integration_id = external_id or (active or {}).get("id") or os.getenv("MENU_ID") or os.getenv("MENU_SHOP_ID", "")

        payload = {
            "id": integration_id,
            "name": name or os.getenv("MENU_OBJECT_NAME", "test-item"),
            "class_name": self.class_name,
            "object_id": self.object_id,
            "detection_time": self.detection_time.isoformat(),
            "location": json.dumps(self.location),
            "inOrOut": self.inOrOut,
            "confidence": self.confidence,
        }
        if extra_fields:
            payload.update(extra_fields)

        files = None
        if self.image is not None:
            ok, buf = cv2.imencode(".jpg", self.image)
            if ok:
                files = {"image": ("frame.jpg", buf.tobytes(), "image/jpeg")}

        def _post():
            try:
                resp = requests.post(url, data=payload, files=files, timeout=timeout)
                try:
                    resp_text = resp.text
                except Exception:
                    resp_text = "<unreadable response>"

                status_text = "success" if resp.ok else "error"
                message = f"Status {resp.status_code}" if resp.ok else f"Status {resp.status_code}: {resp_text[:100]}"
                integration_config.add_log(self.class_name, self.inOrOut, status_text, message)

                if resp.ok:
                    print(f"Posted detected object to {url} - status={resp.status_code}")
                else:
                    print(f"Posting detected object to {url} returned status={resp.status_code}, response={resp_text[:1000]}")
            except Exception as e:
                integration_config.add_log(self.class_name, self.inOrOut, "error", f"Failed: {str(e)}")
                print(f"Warning: Failed posting detected object: {e}")

        threading.Thread(target=_post, daemon=True).start()
