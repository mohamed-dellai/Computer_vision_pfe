from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

@dataclass
class RTSPConfig:
    width: Optional[int] = None
    height: Optional[int] = None
    fps: int = 15
    reconnect_delay: float = 3.0
    buffer_size: int = 1
    read_timeout: float = 5.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RTSPConfig':
        def _optional_int(value):
            if value is None or value == '':
                return None
            return int(value)

        return cls(
            width=_optional_int(data.get('rtsp_width')),
            height=_optional_int(data.get('rtsp_height')),
            fps=int(data.get('rtsp_fps', 15)),
            reconnect_delay=float(data.get('rtsp_reconnect_delay', 3.0)),
            buffer_size=int(data.get('rtsp_buffer_size', 1)),
            read_timeout=float(data.get('rtsp_read_timeout', 5.0))
        )

@dataclass
class TrackerConfig:
    tracker_file: str = 'bytetrack.yaml'

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrackerConfig':
        # Only accept tracker_file parameter - no individual parameters
        tracker_file = data.get('tracker_file', 'bytetrack.yaml')
        
        # Validate that the tracker file exists
        import os
        tracker_path = os.path.join(os.path.dirname(__file__), 'trackers', tracker_file)
        
        if not os.path.exists(tracker_path):
            raise ValueError(f"Tracker configuration file not found: {tracker_file}")
        
        return cls(tracker_file=tracker_path)

@dataclass
class JobConfig:
    camera_id: str
    model_name: str
    rtsp_url: str
    conf: float
    iou: float
    rtsp_config: RTSPConfig
    tracker_config: TrackerConfig
    line_coords: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Returns a flat dictionary representation of the config."""
        return {
            'camera_id': self.camera_id,
            'model': self.model_name,
            'conf': self.conf,
            'iou': self.iou,
            'line_coords': self.line_coords,
            # Flatten RTSP config
            'rtsp_width': self.rtsp_config.width,
            'rtsp_height': self.rtsp_config.height,
            'rtsp_fps': self.rtsp_config.fps,
            'rtsp_reconnect_delay': self.rtsp_config.reconnect_delay,
            'rtsp_buffer_size': self.rtsp_config.buffer_size,
            'rtsp_read_timeout': self.rtsp_config.read_timeout,
            # Only tracker file path
            'tracker_file': self.tracker_config.tracker_file,
        }
