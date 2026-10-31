"""2D image geometry only. Convex bed polygons are checked explicitly."""
from __future__ import annotations

import math
from .common import require


def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def area(poly):
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(poly, poly[1:]+poly[:1]))) / 2 if len(poly) >= 3 else 0.0


def inside(point, poly):
    signs = [cross(a, b, point) for a, b in zip(poly, poly[1:] + poly[:1])]
    return all(x >= -1e-9 for x in signs) or all(x <= 1e-9 for x in signs)


def segment_distance(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    length2 = dx*dx+dy*dy
    require(length2 > 0, "Zero-length boundary")
    t = max(0., min(1., ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/length2))
    return math.hypot(p[0]-a[0]-t*dx, p[1]-a[1]-t*dy)


def clip_to_box(poly, box):
    out = list(poly)
    for axis, edge, sign in ((0, box[0], 1), (0, box[2], -1), (1, box[1], 1), (1, box[3], -1)):
        source, out = out, []
        if not source:
            break
        for a, b in zip(source[-1:] + source[:-1], source):
            ai, bi = sign*(a[axis]-edge) >= 0, sign*(b[axis]-edge) >= 0
            if ai != bi:
                t = (edge-a[axis])/(b[axis]-a[axis])
                out.append([a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])])
            if bi:
                out.append(b)
    return out


def validate_geometry(g):
    require(g["coordinate_system"] == "normalized_image_xy", "Geometry must use normalized_image_xy")
    p = g["bed_polygon"]
    require(len(p) >= 3, "At least 3 polygon vertices required")
    require(all(len(q) == 2 and all(math.isfinite(v) and 0 <= v <= 1 for v in q) for q in p), "Polygon vertices outside [0,1]")
    turns = [cross(p[i-1], p[i], p[(i+1) % len(p)]) for i in range(len(p))]
    require(area(p) > 1e-6 and (all(t > 1e-9 for t in turns) or all(t < -1e-9 for t in turns)), "Use a nondegenerate convex polygon in perimeter order")
    require(g.get("reference_width", 0) > 0 and g.get("reference_height", 0) > 0, "Calibrated frame dimensions required")
    require(isinstance(g.get("calibration_valid"), bool), "calibration_valid must be explicit")
    ids = set()
    for b in g["boundaries"]:
        require(b["id"] not in ids, "Duplicate boundary ID")
        ids.add(b["id"])
        require(b["type"] in ("open", "rail", "wall", "unknown"), "Unknown boundary type")
        require(len(b["points"]) == 2, "Boundary needs 2 endpoints")
        a, c = b["points"]
        require(a != c, "Zero-length boundary")
        require(any(segment_distance(a, x, y) < 1e-6 and segment_distance(c, x, y) < 1e-6 for x, y in zip(p, p[1:]+p[:1])), "Boundary must lie on one bed polygon edge")


def geometry_features(box, g):
    x1,y1,x2,y2 = box
    cx,cy = (x1+x2)/2, (y1+y2)/2
    poly = g["bed_polygon"]
    outside = 1 - area(clip_to_box(poly, box)) / ((x2-x1)*(y2-y1))
    # Convert normalized x/y to proportional pixels before measuring distances.
    ratio = g["reference_height"] / g["reference_width"]
    to_metric = lambda p: (p[0], p[1]*ratio)
    pp = [to_metric(p) for p in poly]
    diagonal = math.hypot(max(p[0] for p in pp)-min(p[0] for p in pp), max(p[1] for p in pp)-min(p[1] for p in pp))
    centroid = (sum(p[0] for p in poly)/len(poly), sum(p[1] for p in poly)/len(poly))
    result = dict(outside_ratio=max(0., min(1., outside)))
    for kind in ("open", "rail"):
        distances = []
        for boundary in g["boundaries"]:
            if boundary["type"] != kind:
                continue
            a,b = boundary["points"]
            sign = 1 if cross(a,b,centroid)*cross(a,b,(cx,cy)) >= 0 else -1
            distance = segment_distance(to_metric((cx,cy)),to_metric(a),to_metric(b))/diagonal
            distances.append((distance,boundary["id"],sign))
        dist,bid,sign = min(distances) if distances else (None,None,1)
        result[f"{kind}_edge_distance"] = sign*dist if dist is not None else None
        result[f"{kind}_boundary_id"] = bid
    return result
