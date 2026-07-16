from logic.zones import EDGE, find_zone, is_at_height, zone_label


def feet_point(person_box):
    """Bottom-center of the bbox: where the person contacts the surface."""
    x1, y1, x2, y2 = person_box
    return (int((x1 + x2) / 2), int(y2))


def _frame_dims(frame_shape):
    if isinstance(frame_shape, (tuple, list)):
        return frame_shape[0], frame_shape[1]
    return frame_shape, None


def is_person_at_height(person_box, frame_shape, camera_id=None, threshold=0.8):
    """
    Determines if a person is working at height.

    Preferred path: the camera's AT_HEIGHT polygons from config/zones.json —
    at height iff the feet point falls inside one (per-camera site knowledge).
    Fallback (no config for this camera, or only the frame height is known):
    bbox-bottom position heuristic.

    frame_shape is (h, w) as in numpy frame.shape, or a bare image height.
    """
    if person_box is None:
        return False

    h, w = _frame_dims(frame_shape)

    if camera_id is not None and w is not None:
        verdict = is_at_height(feet_point(person_box), (w, h), camera_id)
        if verdict is not None:
            return verdict

    y_bottom = person_box[3]
    return y_bottom / h < threshold


def get_person_edge_zone(person_box, frame_shape, camera_id=None):
    """
    Human-readable name of the EDGE zone the person is standing in, or None.

    EDGE zones are traced along the open (unprotected) sides of elevated
    surfaces — a polygon boundary is not itself a drop, so which side is open
    is site knowledge a person has to mark.
    """
    if person_box is None or camera_id is None:
        return None
    h, w = _frame_dims(frame_shape)
    if w is None:
        return None
    zone = find_zone(feet_point(person_box), (w, h), camera_id, EDGE)
    return zone_label(zone) if zone else None
