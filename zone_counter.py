import time
from typing import Callable, Dict, List, Optional, Tuple


Zone = List[int]
Point = Tuple[int, int]


class ZoneCounter:
    """Counts tracked objects after they dwell inside named zones."""

    def __init__(
        self,
        zones: Dict[str, Zone],
        dwell_seconds: float = 3.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.zones = {
            name: self._normalize_zone(zone)
            for name, zone in (zones or {}).items()
            if name in {"in", "out"} and self._is_zone(zone)
        }
        self.dwell_seconds = max(0.0, float(dwell_seconds))
        self.clock = clock
        self.track_zone_state: Dict[int, Dict[str, Dict[str, object]]] = {}
        self.counts: Dict[str, Dict[str, int]] = {}

    def update(self, track_id: int, class_name: str, point: Point) -> Optional[str]:
        now = self.clock()
        state = self.track_zone_state.setdefault(track_id, {})

        for zone_name, zone in self.zones.items():
            zone_state = state.setdefault(zone_name, {"entered_at": None, "counted": False})
            if self._point_in_zone(point, zone):
                if zone_state["entered_at"] is None:
                    zone_state["entered_at"] = now
                    zone_state["counted"] = False
                    continue

                elapsed = now - float(zone_state["entered_at"])
                if elapsed >= self.dwell_seconds and not zone_state["counted"]:
                    zone_state["counted"] = True
                    self._increment(class_name, zone_name)
                    return zone_name
            else:
                zone_state["entered_at"] = None
                zone_state["counted"] = False

        return None

    def get_counts(self) -> Dict[str, Dict[str, int]]:
        return self.counts

    def _increment(self, class_name: str, zone_name: str) -> None:
        counts = self.counts.setdefault(class_name, {"in": 0, "out": 0, "net": 0})
        if zone_name == "in":
            counts["in"] += 1
            counts["net"] += 1
        elif zone_name == "out":
            counts["out"] += 1
            counts["net"] -= 1

    @staticmethod
    def _is_zone(zone) -> bool:
        return isinstance(zone, list) and len(zone) == 4

    @staticmethod
    def _normalize_zone(zone: Zone) -> Zone:
        x1, y1, x2, y2 = [int(v) for v in zone]
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    @staticmethod
    def _point_in_zone(point: Point, zone: Zone) -> bool:
        px, py = point
        x1, y1, x2, y2 = zone
        return x1 <= px <= x2 and y1 <= py <= y2
