import json
from pathlib import Path

import cv2
import numpy as np

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "zones.json"

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


def get_at_height_zones(camera_id):
    """
    Returns the list of approved AT_HEIGHT zone dicts configured for this
    camera, or None when the camera has none (caller should fall back to the
    position heuristic).

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
        if z.get("type") == "AT_HEIGHT" and z.get("status", "approved") == "approved"
    ]
    return zones or None


def scale_polygon(polygon, frame_size):
    w, h = frame_size
    return [(int(x * w), int(y * h)) for x, y in polygon]


def is_at_height(feet_point, frame_size, camera_id):
    """
    Polygon-based elevation test: True iff the feet point lies inside any
    AT_HEIGHT polygon configured for this camera. Polygons are stored in
    normalized [0,1] coordinates, so they work at any stream resolution.
    Returns None when the camera has no AT_HEIGHT config.
    """
    zones = get_at_height_zones(camera_id)
    if zones is None:
        return None
    for zone in zones:
        if point_in_zone(feet_point, scale_polygon(zone["polygon"], frame_size)):
            return True
    return False
