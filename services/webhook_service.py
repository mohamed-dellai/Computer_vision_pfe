import time
import queue
import threading
import uuid
import requests
from dataclasses import dataclass, field
from typing import Optional, Dict

import integration_config

@dataclass(order=True)
class WebhookEvent:
    execute_at: float
    event_id: str = field(compare=False)
    url: str = field(compare=False)
    payload: dict = field(compare=False)
    files: Optional[dict] = field(compare=False)
    timeout: float = field(compare=False)
    class_name: str = field(compare=False)
    event_type: str = field(compare=False)
    retry_count: int = field(default=0, compare=False)

class WebhookDispatcher:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(WebhookDispatcher, cls).__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self):
        self.q = queue.PriorityQueue()
        self.RATE_LIMIT_DELAY = 0.3  # 300ms between requests
        self.RETRY_DELAY = 10.0      # 10s before retry
        self.MAX_RETRIES = 3
        self.last_request_time = 0.0
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def enqueue(self, url: str, payload: dict, files: Optional[dict], timeout: float, class_name: str, event_type: str):
        event_id = str(uuid.uuid4())
        payload['event_id'] = event_id
        
        event = WebhookEvent(
            execute_at=time.time(),
            event_id=event_id,
            url=url,
            payload=payload,
            files=files,
            timeout=timeout,
            class_name=class_name,
            event_type=event_type
        )
        self.q.put(event)

    def _worker(self):
        while True:
            try:
                event = self.q.get()
                
                # Check if it's time to execute this event
                now = time.time()
                if event.execute_at > now:
                    # Put it back and sleep a bit so we don't spin, but we need to check queue
                    self.q.put(event)
                    time.sleep(0.1)
                    continue

                # Rate limiting
                time_since_last = time.time() - self.last_request_time
                if time_since_last < self.RATE_LIMIT_DELAY:
                    time.sleep(self.RATE_LIMIT_DELAY - time_since_last)
                
                # Update last request time before making the request
                self.last_request_time = time.time()

                # Attempt delivery
                self._dispatch(event)

                self.q.task_done()
            except Exception as e:
                print(f"Webhook worker encountered an error: {e}")
                time.sleep(1.0) # sleep briefly to prevent tight failure loop

    def _dispatch(self, event: WebhookEvent):
        try:
            print(f"Sending webhook payload to {event.url}: {event.payload}")
            resp = requests.post(event.url, data=event.payload, files=event.files, timeout=event.timeout)
            try:
                resp_text = resp.text
            except Exception:
                resp_text = "<unreadable response>"

            if resp.ok:
                message = f"Status {resp.status_code}"
                integration_config.add_log(event.class_name, event.event_type, "success", message)
                print(f"Posted detected object to {event.url} - status={resp.status_code}")
            else:
                self._handle_failure(event, f"Status {resp.status_code}: {resp_text[:100]}")
        except Exception as e:
            self._handle_failure(event, f"Network/Timeout: {str(e)}")

    def _handle_failure(self, event: WebhookEvent, error_msg: str):
        if event.retry_count < self.MAX_RETRIES:
            event.retry_count += 1
            event.execute_at = time.time() + self.RETRY_DELAY
            integration_config.add_log(
                event.class_name, 
                event.event_type, 
                "error", 
                f"Attempt {event.retry_count} failed, retrying in 10s: {error_msg}"
            )
            # Make sure we don't resend the file pointer at EOF. 
            # In our case files dict holds bytes directly from imencode.tobytes(), so it's safe to reuse.
            self.q.put(event)
        else:
            integration_config.add_log(
                event.class_name, 
                event.event_type, 
                "error", 
                f"Permanently failed after {self.MAX_RETRIES} retries: {error_msg}"
            )
            print(f"Warning: Dropping webhook event {event.event_id} after {self.MAX_RETRIES} retries.")

# Global instance for ease of use
dispatcher = WebhookDispatcher()
