import os
import cv2
import numpy as np
from ultralytics import YOLO
from utils.detected_object import DetectedObject
from datetime import datetime

class VideoProcessor:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.stats = {
            'total_unique': 0,
            'class_counts': {},  # class_name -> count
            'line_counts': {},   # class_name -> {'in': count, 'out': count}
        }
        self.seen_ids = set()
        self.track_history = {} # track_id -> (cx, cy)
        self.detected_events = []  # list of DetectedObject instances

        self._init_model()

    def _init_model(self):
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'models', self.config.model_name)
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
            
    def close(self):
        """Clean up resources."""
        # Ultralytics models don't strictly require a close method,
        # but this provides a hook for any cleanup we might need later
        # (e.g. if we were using a different inference engine).
        self.model = None
        self.track_history.clear()
        self.seen_ids.clear()

    def process_frame(self, frame):
        """
        Process a single frame: detect, track, update stats, and draw.
        Returns the annotated frame.
        """
        try:
            tracker_file = self.config.tracker_config.tracker_file
            results = self.model.track(
                frame, 
                persist=True, 
                verbose=False,
                tracker=tracker_file,
                stream=True,
                conf=self.config.conf,
                iou=self.config.iou
            )

            annotated_frame = frame
            
            # Draw counting line if configured
            if self.config.line_coords:
                x1, y1, x2, y2 = self.config.line_coords
                cv2.line(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Label ends A/B for reference
                cv2.putText(annotated_frame, "A", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(annotated_frame, "B", (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            for result in results:
               

                # Plot/annotate first, then pass annotated frame so events include overlays
                annotated_frame = result.plot()
                self._update_stats(result, annotated_frame)
                self._draw_stats(annotated_frame)
                break # Process only the first result (usually only one per frame)
            
            return annotated_frame

        except Exception as e:
            print(f"Error: Frame processing error: {e}")
            return frame

    def _update_stats(self, result, frame):
        boxes = getattr(result, "boxes", None)
        if boxes is None or not hasattr(boxes, "id") or boxes.id is None:
            return

        ids = boxes.id.cpu().numpy().astype(int).tolist()
        cls_idxs = boxes.cls.cpu().numpy().astype(int).tolist() if hasattr(boxes, "cls") else []
        xywhs = boxes.xywh.cpu().numpy().tolist() if hasattr(boxes, "xywh") else []
        confs = boxes.conf.cpu().numpy().tolist() if hasattr(boxes, "conf") else []

        for i, (tid, cid) in enumerate(zip(ids, cls_idxs)):
            cname = self.model.names.get(cid, str(cid))
            
            # Unique ID counting
            if tid not in self.seen_ids:
                self.seen_ids.add(tid)
                self.stats['total_unique'] = len(self.seen_ids)
                self.stats['class_counts'][cname] = self.stats['class_counts'].get(cname, 0) + 1
                
                # Init line counts for this class if needed
                if cname not in self.stats['line_counts']:
                    self.stats['line_counts'][cname] = {'in': 0, 'out': 0}

                print(f"Info: New object detected: ID {tid} ({cname})")
                try:
                    location = []
                    if i < len(xywhs):
                        cx, cy = int(xywhs[i][0]), int(xywhs[i][1])
                        w, h = int(xywhs[i][2]), int(xywhs[i][3])
                        x = max(0, int(cx - w / 2))
                        y = max(0, int(cy - h / 2))
                        location = [x, y, int(w), int(h)]

                    conf = confs[i] if i < len(confs) else 0.0

                    dobj = DetectedObject(
                        class_name=cname,
                        object_id=tid,
                        detection_time=datetime.now(),
                        location=location,
                        image=frame.copy(),
                        inOrOut="unknown",
                        confidence=float(conf)
                    )
                    dobj.post_event()
                except Exception as e:
                    print(f"Warning: Failed sending DetectedObject: {e}")

            # Line crossing counting
            if self.config.line_coords and i < len(xywhs):
                cx, cy = int(xywhs[i][0]), int(xywhs[i][1])
                w, h = int(xywhs[i][2]), int(xywhs[i][3])
                curr_point = (cx, cy)

                if tid in self.track_history:
                    prev_point = self.track_history[tid]
                    direction = self._check_crossing(prev_point, curr_point, self.config.line_coords)

                    if direction:
                        # Init if not exists (redundant but safe)
                        if cname not in self.stats['line_counts']:
                             self.stats['line_counts'][cname] = {'in': 0, 'out': 0}
                             
                        self.stats['line_counts'][cname][direction] += 1
                        print(f"Info: Object crossed line ({direction}): ID {tid} ({cname})")

                        # Create DetectedObject event and store it
                        try:
                            # Convert xywh center to top-left x,y and ensure ints
                            x = int(cx - w/2)
                            y = int(cy - h/2)
                            x = max(0, x)
                            y = max(0, y)
                           


                            conf = confs[i] if i < len(confs) else 0.0

                            dobj = DetectedObject(
                                class_name=cname,
                                object_id=tid,
                                detection_time=datetime.now(),
                                location=[x, y, int(w), int(h)],
                                image= frame.copy(),
                                inOrOut=direction,
                                confidence=float(conf)
                            )

                            self.detected_events.append(dobj)
                        except Exception as e:
                            print(f"Warning: Failed creating DetectedObject: {e}")

                self.track_history[tid] = curr_point

    def _check_crossing(self, p1, p2, line_coords):
        """
        Check if movement from p1 to p2 crosses the line.
        Returns 'in' or 'out' if crossed, None otherwise.
        
        Direction Convention:
        - The line is defined from Start (A) to End (B).
        - "In": Crossing from the Right side to the Left side (relative to A->B).
        - "Out": Crossing from the Left side to the Right side (relative to A->B).
        """
        x1, y1, x2, y2 = line_coords
        l1, l2 = (x1, y1), (x2, y2)
        
        # Helper for orientation
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            
        intersect = ccw(p1, l1, l2) != ccw(p2, l1, l2) and ccw(p1, p2, l1) != ccw(p1, p2, l2)
        
        if intersect:
            # We determine direction based on whether the object moved 
            # generally "downwards" (In) or "upwards" (Out) relative to the screen.
            # You can easily swap these if your camera is upside down!
            y_movement = p2[1] - p1[1]
            
            # If the object moved downwards (positive Y on screen), it's "In"
            if y_movement > 0:
                return 'in'
            # If the object moved upwards (negative Y on screen), it's "Out"
            elif y_movement < 0:
                return 'out'
            else:
                # If they moved perfectly horizontal, use X movement
                x_movement = p2[0] - p1[0]
                return 'in' if x_movement > 0 else 'out'
            
        return None

    def _draw_stats(self, frame):
        y_offset = 30
        
        # Draw class counts (Total Unique)
        for cname, count in self.stats['class_counts'].items():
            text = f"Total {cname}: {count}"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (10, y_offset - h - 5), (10 + w, y_offset + 5), (0, 0, 0), -1)
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_offset += 35

        # Draw line counts if enabled
        if self.config.line_coords:
             y_offset += 10
             title = "Line Crossings:"
             (w, h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
             cv2.rectangle(frame, (10, y_offset - h - 5), (10 + w, y_offset + 5), (0, 0, 0), -1)
             cv2.putText(frame, title, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
             y_offset += 35
             
             for cname, counts in self.stats['line_counts'].items():
                in_count = counts['in']
                out_count = counts['out']
                text = f"{cname}: In {in_count} | Out {out_count}"
                
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (10, y_offset - h - 5), (10 + w, y_offset + 5), (0, 0, 0), -1)
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                y_offset += 35
