import os
import cv2
import numpy as np
from ultralytics import YOLO
from utils.detected_object import DetectedObject
from datetime import datetime
from zone_counter import ZoneCounter

class VideoProcessor:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.stats = {
            'total_unique': 0,
            'class_counts': {},  # class_name -> count
            'line_counts': {},   # class_name -> {'in': count, 'out': count}
            'zone_counts': {},   # class_name -> {'in': count, 'out': count, 'net': count}
        }
        self.seen_ids = set()
        self.track_history = {} # track_id -> (cx, cy)
        self.track_sides = {} # track_id -> last non-zero side of the counting line
        self.detected_events = []  # list of DetectedObject instances
        self.counting_method = getattr(config, 'counting_method', None)
        if not self.counting_method:
            self.counting_method = 'zone' if getattr(config, 'zone_coords', None) else ('line' if getattr(config, 'line_coords', None) else 'none')
        self.zone_counter = None
        if self.counting_method == 'zone' and getattr(config, 'zone_coords', None):
            dwell_seconds = getattr(config, 'zone_dwell_seconds', 3.0)
            self.zone_counter = ZoneCounter(config.zone_coords, dwell_seconds=dwell_seconds)

        # Line coordinates are expected in the same coordinate space as the
        # frames passed to process_frame(). The worker crops the frame when
        # `config.crop_coords` is set, so the frontend should supply
        # coordinates relative to the cropped frame. No additional
        # transformation is needed here.

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
        self.track_sides.clear()
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

            for result in results:
               

                # Plot/annotate only after tracking has consumed the clean frame.
                annotated_frame = result.plot()
                self._update_stats(result, annotated_frame)
                self._draw_counting_line(annotated_frame)
                self._draw_zones(annotated_frame)
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

                    in_or_out = "in" if self.counting_method == 'none' else "unknown"
                    dobj = DetectedObject(
                        class_name=cname,
                        object_id=tid,
                        detection_time=datetime.now(),
                        location=location,
                        image=frame.copy(),
                        inOrOut=in_or_out,
                        confidence=float(conf)
                    )
                    if self.counting_method == 'none':
                        dobj.post_event()
                except Exception as e:
                    print(f"Warning: Failed sending DetectedObject: {e}")

            # Line crossing counting
            if self.counting_method == 'line' and self.config.line_coords and i < len(xywhs):
                cx, cy = int(xywhs[i][0]), int(xywhs[i][1])
                w, h = int(xywhs[i][2]), int(xywhs[i][3])
                curr_point = (cx, cy)
                curr_side = self._point_side(curr_point, self.config.line_coords)

                if tid in self.track_history:
                    prev_point = self.track_history[tid]
                    prev_side = self.track_sides.get(tid)
                    direction = self._check_crossing(prev_point, curr_point, self.config.line_coords, prev_side, curr_side)

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
                            dobj.post_event()
                        except Exception as e:
                            print(f"Warning: Failed creating DetectedObject: {e}")

                self.track_history[tid] = curr_point
                if curr_side != 0:
                    self.track_sides[tid] = curr_side

            # Zone dwell counting
            if self.counting_method == 'zone' and self.zone_counter and i < len(xywhs):
                cx, cy = int(xywhs[i][0]), int(xywhs[i][1])
                w, h = int(xywhs[i][2]), int(xywhs[i][3])
                direction = self.zone_counter.update(tid, cname, (cx, cy))
                self.stats['zone_counts'] = self.zone_counter.get_counts()

                if direction:
                    print(f"Info: Object counted in {direction} zone: ID {tid} ({cname})")
                    try:
                        x = max(0, int(cx - w / 2))
                        y = max(0, int(cy - h / 2))
                        conf = confs[i] if i < len(confs) else 0.0

                        dobj = DetectedObject(
                            class_name=cname,
                            object_id=tid,
                            detection_time=datetime.now(),
                            location=[x, y, int(w), int(h)],
                            image=frame.copy(),
                            inOrOut=direction,
                            confidence=float(conf)
                        )
                        self.detected_events.append(dobj)
                        dobj.post_event()
                    except Exception as e:
                        print(f"Warning: Failed creating zone DetectedObject: {e}")

    def _point_side(self, point, line_coords):
        """
        Return the side of point relative to the directed line A->B:
        1 = left side, -1 = right side, 0 = on the line.
        """
        x1, y1, x2, y2 = line_coords
        px, py = point
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross > 0:
            return 1
        if cross < 0:
            return -1
        return 0

    def _segments_intersect(self, p1, p2, q1, q2):
        def orientation(a, b, c):
            value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if value > 0:
                return 1
            if value < 0:
                return -1
            return 0

        def on_segment(a, b, c):
            return (
                min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
                and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
            )

        o1 = orientation(p1, p2, q1)
        o2 = orientation(p1, p2, q2)
        o3 = orientation(q1, q2, p1)
        o4 = orientation(q1, q2, p2)

        if o1 != o2 and o3 != o4:
            return True
        if o1 == 0 and on_segment(p1, q1, p2):
            return True
        if o2 == 0 and on_segment(p1, q2, p2):
            return True
        if o3 == 0 and on_segment(q1, p1, q2):
            return True
        if o4 == 0 and on_segment(q1, p2, q2):
            return True

        return False

    def _check_crossing(self, p1, p2, line_coords, prev_side=None, curr_side=None):
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

        if prev_side is None:
            prev_side = self._point_side(p1, line_coords)
        if curr_side is None:
            curr_side = self._point_side(p2, line_coords)

        if prev_side == 0 or curr_side == 0 or prev_side == curr_side:
            return None

        if self._segments_intersect(p1, p2, l1, l2):
            if prev_side < 0 and curr_side > 0:
                return 'in'
            if prev_side > 0 and curr_side < 0:
                return 'out'

        return None

    def _draw_counting_line(self, frame):
        if self.counting_method != 'line' or not self.config.line_coords:
            return

        x1, y1, x2, y2 = self.config.line_coords
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "A", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, "B", (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def _draw_zones(self, frame):
        if self.counting_method != 'zone' or not self.zone_counter:
            return

        colors = {
            'in': (0, 180, 0),
            'out': (0, 0, 220),
        }
        labels = {
            'in': 'IN +1',
            'out': 'OUT -1',
        }
        for zone_name, zone in self.zone_counter.zones.items():
            x1, y1, x2, y2 = zone
            color = colors.get(zone_name, (255, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, labels.get(zone_name, zone_name.upper()), (x1, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

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
        if self.counting_method == 'line' and self.config.line_coords:
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

        if self.counting_method == 'zone' and self.zone_counter:
             y_offset += 10
             title = "Zone Counts:"
             (w, h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
             cv2.rectangle(frame, (10, y_offset - h - 5), (10 + w, y_offset + 5), (0, 0, 0), -1)
             cv2.putText(frame, title, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
             y_offset += 35

             for cname, counts in self.stats['zone_counts'].items():
                text = f"{cname}: +{counts['in']} | -{counts['out']} | Net {counts['net']}"

                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (10, y_offset - h - 5), (10 + w, y_offset + 5), (0, 0, 0), -1)
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                y_offset += 35
