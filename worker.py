import threading
import time
import cv2
from rtsp import RTSPVideoStream
from processor import VideoProcessor

class TrackingWorker(threading.Thread):
    def __init__(self, config):
        super().__init__(daemon=True)
        self.config = config
        self._stop_event = threading.Event()
        self.status = "created"
        
        self._jpeg_condition = threading.Condition()
        self.latest_jpeg = None
        self.jpeg_frame_id = 0
        
        self.stats = {
            'total_unique': 0,
            'class_counts': {},
            'line_counts': {},
        }
        self.processor = None

    def start(self):
        self.status = "starting"
        super().start()

    def run(self):
        try:
            self.processor = VideoProcessor(self.config)
            self.status = "running"
        except Exception as e:
            self.status = f"error: {e}"
            return

        rtsp_cfg = self.config.rtsp_config
        stream = RTSPVideoStream(
            rtsp_url=self.config.rtsp_url,
            width=rtsp_cfg.width,
            height=rtsp_cfg.height,
            fps=rtsp_cfg.fps,
            reconnect_delay=rtsp_cfg.reconnect_delay,
            buffer_size=rtsp_cfg.buffer_size,
            read_timeout=rtsp_cfg.read_timeout
        )

        try:
            stream.start()
            last_frame_id = -1
            while not self._stop_event.is_set():
                im0, last_frame_id = stream.read(timeout=rtsp_cfg.read_timeout, last_frame_id=last_frame_id)
                if im0 is None:
                    print("No frame received from stream, retrying...")
                    time.sleep(0.2)
                    continue

                annotated_frame = self.processor.process_frame(im0)
                self.stats = self.processor.stats
                ok, jpeg = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    with self._jpeg_condition:
                        self.latest_jpeg = jpeg.tobytes()
                        self.jpeg_frame_id += 1
                        self._jpeg_condition.notify_all()

        except Exception as e:
            self.status = f"error: {e}"
        finally:
            try:
                stream.stop()
            except Exception:
                pass
            if self.processor is not None:
                try:
                    self.processor.close()
                except Exception:
                    pass
            if self.status == 'running':
                self.status = 'stopped'

    def stop(self):
        self._stop_event.set()
        with self._jpeg_condition:
            self._jpeg_condition.notify_all()

    def read_jpeg(self, timeout=None, last_frame_id=-1):
        start = time.time()
        with self._jpeg_condition:
            while self.jpeg_frame_id <= last_frame_id:
                if self._stop_event.is_set():
                    return None, last_frame_id
                if timeout is not None:
                    elapsed = time.time() - start
                    if elapsed >= timeout:
                        return None, last_frame_id
                    self._jpeg_condition.wait(timeout - elapsed)
                else:
                    self._jpeg_condition.wait()
            return self.latest_jpeg, self.jpeg_frame_id

    def is_running(self):
        return self.status == 'running'
