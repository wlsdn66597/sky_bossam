from __future__ import annotations

import math
from .common import require
from .geometry import geometry_features, validate_geometry

FEATURE_ORDER = ["center_x", "center_y", "bbox_w", "bbox_h", "det_conf", "outside_ratio", "open_edge_distance", "rail_edge_distance", "velocity_x", "velocity_y", "approach_speed", "area_change_per_s", "track_age_s", "dt_s", "missing_age_s", "det_valid", "geometry_valid", "has_open_boundary", "has_rail_boundary", "brightness_mean"]
BINARY_INDICES = [15,16,17,18]


def iou(a,b):
    overlap = max(0., min(a[2],b[2])-max(a[0],b[0])) * max(0., min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-overlap
    return overlap/union if union > 0 else 0.


class FeatureExtractor:
    def __init__(self, geometry, max_gap=.5):
        validate_geometry(geometry)
        self.g = geometry
        self.max_gap = max_gap
        self.track_id = 0
        self.box = None
        self.last_seen = self.started = self.last_ts = None
        self.previous = None

    def step(self, timestamp, detections, brightness, geometry_valid=True):
        require(math.isfinite(timestamp) and (self.last_ts is None or timestamp > self.last_ts), "Timestamps must strictly increase")
        dt = timestamp-self.last_ts if self.last_ts is not None else None
        self.last_ts = timestamp
        valid = bool(self.g["calibration_valid"] and geometry_valid)
        for d in detections:
            x1,y1,x2,y2 = d["bbox"]
            require(0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 and 0 <= d["confidence"] <= 1, "Invalid detector box/confidence")
        chosen, replaced = None, False
        if self.box is not None and self.last_seen is not None and timestamp-self.last_seen <= self.max_gap:
            matches = [d for d in detections if iou(self.box, d["bbox"]) >= .1]
            if len(matches) == 1:
                chosen = matches[0]
        if chosen is None and len(detections) == 1:
            chosen = detections[0]
            replaced = True
        values = {k: None for k in FEATURE_ORDER}
        values.update(det_valid=float(chosen is not None), geometry_valid=float(valid), has_open_boundary=float(any(b["type"] == "open" for b in self.g["boundaries"])), has_rail_boundary=float(any(b["type"] == "rail" for b in self.g["boundaries"])), brightness_mean=brightness, dt_s=dt)
        boundary = None
        if chosen:
            if replaced:
                self.track_id += 1
                self.started = timestamp
                self.previous = None
            x1,y1,x2,y2 = chosen["bbox"]
            values.update(center_x=(x1+x2)/2, center_y=(y1+y2)/2, bbox_w=x2-x1, bbox_h=y2-y1, det_conf=chosen["confidence"], track_age_s=timestamp-self.started, missing_age_s=0.)
            if valid:
                geo = geometry_features(chosen["bbox"], self.g)
                values.update({k: geo[k] for k in ("outside_ratio", "open_edge_distance", "rail_edge_distance")})
                boundary = geo["open_boundary_id"]
            prev = self.previous
            if prev and dt is not None and dt <= self.max_gap:
                values["velocity_x"] = (values["center_x"]-prev["center_x"])/dt
                values["velocity_y"] = (values["center_y"]-prev["center_y"])/dt
                values["area_change_per_s"] = (values["bbox_w"]*values["bbox_h"]-prev["bbox_w"]*prev["bbox_h"])/dt
                if valid and boundary == prev.get("boundary_id") and prev["open_edge_distance"] is not None:
                    values["approach_speed"] = (prev["open_edge_distance"]-values["open_edge_distance"])/dt
            self.box = chosen["bbox"]
            self.last_seen = timestamp
            self.previous = dict(values, boundary_id=boundary)
        else:
            values["missing_age_s"] = timestamp-self.last_seen if self.last_seen is not None else None
            self.previous = None  # No interpolation through missing observations.
        return dict(schema_version=1, video_ts_s=timestamp, track_id=self.track_id or None, track_replaced=replaced, boundary_id=boundary, bbox=chosen["bbox"] if chosen else None, observed=chosen is not None, values=[values[k] for k in FEATURE_ORDER], masks=[int(values[k] is not None) for k in FEATURE_ORDER])


class R0:
    def __init__(self, config):
        self.c = config
        for name in ("near_distance_ratio", "outward_speed_ratio_per_s", "warning_hold_s", "clear_hold_s", "max_observation_gap_s", "alarm_merge_gap_s"):
            require(math.isfinite(config[name]) and config[name] >= 0, f"Invalid policy parameter {name}")
        self.candidate = self.clear_since = self.last_ts = None
        self.track = self.boundary = None
        self.active = False
        self.counter = 0

    def step(self, row):
        ts = row["video_ts_s"]
        require(self.last_ts is None or ts > self.last_ts, "R0 time must strictly increase")
        gap = self.last_ts is not None and ts-self.last_ts > self.c["max_observation_gap_s"]
        self.last_ts = ts
        v = dict(zip(FEATURE_ORDER, row["values"]))
        events = []
        switched = self.track is not None and row["track_id"] != self.track
        boundary_changed = bool(row["observed"] and self.boundary is not None and row.get("boundary_id") != self.boundary)
        if switched or boundary_changed:
            if self.active:
                events.append(self.event("risk_interrupted", row, ["track_or_boundary_changed"]))
            self.active = False
        if gap or switched or boundary_changed:
            self.candidate = self.clear_since = None
        self.track = row["track_id"]
        if row["observed"]:
            self.boundary = row.get("boundary_id")
        available = bool(v["det_valid"] and v["geometry_valid"])
        if not available:
            self.candidate = self.clear_since = None
            return dict(state="unavailable", active_warning=self.active, observation_valid=False, events=events)
        distance = v["open_edge_distance"]
        rail = v["rail_edge_distance"]
        near = distance is not None and distance <= self.c["near_distance_ratio"]
        rail_near = rail is not None and abs(rail) <= self.c["near_distance_ratio"]
        moving = v["approach_speed"] is not None and v["approach_speed"] >= self.c["outward_speed_ratio_per_s"]
        if near and moving:
            self.clear_since = None
            if self.candidate is None:
                self.candidate = ts
            if not self.active and ts-self.candidate + 1e-9 >= self.c["warning_hold_s"]:
                self.active = True
                events.append(self.event("risk_started", row, ["near_open_boundary", "moving_outward"]))
        else:
            self.candidate = None
            # A stopped infant near/outside an edge is not a clear observation.
            if self.active and distance is not None and distance > self.c["near_distance_ratio"]:
                if self.clear_since is None:
                    self.clear_since = ts
                if ts-self.clear_since + 1e-9 >= self.c["clear_hold_s"]:
                    self.active = False
                    events.append(self.event("risk_ended", row, ["observed_away_from_open_boundary"]))
            else:
                self.clear_since = None
        return dict(state="edge_departure_warning" if self.active else "edge_approach" if near or rail_near else "monitoring", active_warning=self.active, observation_valid=True, events=events)

    def event(self, kind, row, reasons):
        self.counter += 1
        return dict(schema_version=1, event_id=f"r0_{self.counter:06d}", event_type=kind, risk_type="edge_departure_warning", video_ts_s=row["video_ts_s"], track_id=row["track_id"], boundary_id=row.get("boundary_id"), reason_codes=reasons, observation_valid=bool(row["observed"]), risk_probability=None, policy_version=self.c["policy"])
