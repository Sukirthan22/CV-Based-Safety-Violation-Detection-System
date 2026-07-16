import json
from pathlib import Path

import cv2
import numpy as np

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "zones.json"

AT_HEIGHT = "AT_HEIGHT"
EDGE = "EDGE"

_CONFIG = None


def point_in_zone(point, zone):
    return cv2.pointPolygonTest(
        np.array(zone, dtype=np.int32),
        point,
        False
    ) >= 0


def _load_config():
    global _CONFIG
    if _CONFIG is None:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                _CONFIG = json.load(fh)
        else:
            _CONFIG = {}
    return _CONFIG


def get_zones(camera_id, zone_type):
    """
    Returns the approved zones of one type configured for this camera, or None
    when the camera has none.

    Zones written by tools/propose_zones.py carry status "proposed" and are
    ignored until a human approves them (tools/draw_zones.py --review).
    Zones without a status field are treated as approved (hand-traced).
    """
    camera = _load_config().get(camera_id)
    if not camera:
        return None
    zones = [
        z
        for z in camera.get("zones", [])
        if z.get("type") == zone_type and z.get("status", "approved") == "approved"
    ]
    return zones or None


def get_at_height_zones(camera_id):
    """Elevated surfaces: standing here means the harness rule applies."""
    return get_zones(camera_id, AT_HEIGHT)


def get_edge_zones(camera_id):
    """Open edges of elevated surfaces: standing here means a fall risk."""
    return get_zones(camera_id, EDGE)


def zone_label(zone):
    """Human-readable name for messages, from the optional 'label' field."""
    return zone.get("label") or zone["name"].replace("_", " ")


def scale_polygon(polygon, frame_size):
    w, h = frame_size
    return [(int(x * w), int(y * h)) for x, y in polygon]


def find_zone(point, frame_size, camera_id, zone_type):
    """
    First zone of `zone_type` containing `point`, or None. Returns None both
    when the camera has no zones of that type and when the point is outside
    them all — callers that need to tell those apart should check get_zones().
    """
    zones = get_zones(camera_id, zone_type)
    if zones is None:
        return None
    for zone in zones:
        if point_in_zone(point, scale_polygon(zone["polygon"], frame_size)):
            return zone
    return None


def is_at_height(feet_point, frame_size, camera_id):
    """
    Polygon-based elevation test: True iff the feet point lies inside any
    AT_HEIGHT polygon configured for this camera. Polygons are stored in
    normalized [0,1] coordinates, so they work at any stream resolution.
    Returns None when the camera has no AT_HEIGHT config.
    """
    if get_at_height_zones(camera_id) is None:
        return None
    return find_zone(feet_point, frame_size, camera_id, AT_HEIGHT) is not None
