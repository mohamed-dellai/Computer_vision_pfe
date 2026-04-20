import threading
import time
import cv2
import numpy as np
import os

# Force TCP for RTSP to prevent packet loss, stuttering, and smearing over UDP
# stimeout (in microseconds) prevents indefinite hangs if the camera drops silently
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

class RTSPVideoStream(threading.Thread):
    def __init__(self, rtsp_url, width=None, height=None, fps=15, reconnect_delay=3.0, 
                 buffer_size=1, read_timeout=5.0):
        super().__init__(daemon=True)
        self.rtsp_url = rtsp_url
        self.width = int(width) if width else None
        self.height = int(height) if height else None
        self.fps = int(fps)
        self.reconnect_delay = reconnect_delay
        self.buffer_size = int(buffer_size)
        self.read_timeout = read_timeout

        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        
        self.latest_frame = None
        self.frame_count = 0
        self.is_connected = False
        self._cap = None

    def run(self):
        print(f"Starting RTSP reader for {self.rtsp_url}")
        while not self._stop_event.is_set():
            try:
                if self._cap is None or not self._cap.isOpened():
                    self._connect()

                if not self._cap.isOpened():
                    time.sleep(self.reconnect_delay)
                    continue

                ret, frame = self._cap.read()
                if not ret:
                    print("RTSP frame read failed (stream ended or lost), reconnecting...")
                    self._disconnect()
                    time.sleep(self.reconnect_delay)
                    continue

                # Only resize when explicit output dimensions are provided.
                # If width/height are None, keep the camera-native resolution.
                if (
                    self.width is not None
                    and self.height is not None
                    and (frame.shape[0] != self.height or frame.shape[1] != self.width)
                ):
                    frame = cv2.resize(frame, (self.width, self.height))

                with self._condition:
                    self.latest_frame = frame.copy()
                    self.frame_count += 1
                    self.is_connected = True
                    self._condition.notify_all()
                
            except Exception as e:
                print(f"RTSP loop error: {e}")
                self._disconnect()
                time.sleep(self.reconnect_delay)

        self._disconnect()
        print("RTSP reader thread stopped")

    def _connect(self):
        print(f"Connecting to RTSP: {self.rtsp_url}")
        try:
            # Force FFMPEG backend for best RTSP stability
            self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            if self._cap.isOpened():
                # Set buffer size to minimize latency
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
                print(f"RTSP connected: {self.rtsp_url} (buffer_size={self.buffer_size}, backend=FFMPEG)")
            else:
                print(f"Failed to open RTSP: {self.rtsp_url}")
        except Exception as e:
            print(f"RTSP connection error: {e}")
            self._cap = None

    def _disconnect(self):
        with self._condition:
            self.is_connected = False
            self._condition.notify_all()
            
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def read(self, timeout=None, last_frame_id=-1):
        """
        Returns (latest_frame, frame_id). 
        If timeout is provided, waits up to timeout seconds for a new frame 
        (where frame_id > last_frame_id) to exist.
        """
        start = time.time()
        with self._condition:
            while self.frame_count <= last_frame_id or self.latest_frame is None:
                if self._stop_event.is_set():
                    return None, last_frame_id
                    
                if timeout is not None:
                    elapsed = time.time() - start
                    if elapsed >= timeout:
                        return None, last_frame_id
                    self._condition.wait(timeout - elapsed)
                else:
                    self._condition.wait()
                    
            if self.latest_frame is not None:
                return self.latest_frame.copy(), self.frame_count
            return None, last_frame_id

    def stop(self):
        self._stop_event.set()
    
    def is_open(self):
        return self.is_connected
