import os
import json
import threading
from datetime import datetime
from typing import Optional, Dict

import cv2
import numpy as np
import requests

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
        print(f"DetectedObject(class_name={self.class_name}, object_id={self.object_id}, detection_time={self.detection_time}, location={self.location}, inOrOut={self.inOrOut}, confidence={self.confidence})")

    def post_event(
        self,
        endpoint: Optional[str] = None,
        shop_id: Optional[str] = None,
        name: Optional[str] = None,
        timeout: float = 3.0,
        extra_fields: Optional[Dict[str, object]] = None,
    ) -> None:
        url = endpoint or os.getenv(
            "MENU_OBJECT",
            "http://pfe.coffenard.shop/menu/vision/detected-object"
        )
        if not url:
            return

        payload = {
            "shopId": shop_id or os.getenv("MENU_SHOP_ID", "69a1aed0cf3ebf947eac72c6"),
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

                if resp.ok:
                    print(f"Posted detected object to {url} - status={resp.status_code}")
                else:
                    print(
                        f"Posting detected object to {url} returned status={resp.status_code}, response={resp_text[:1000]}"
                    )
            except Exception as e:
                print(f"Warning: Failed posting detected object: {e}")

        threading.Thread(target=_post, daemon=True).start()
