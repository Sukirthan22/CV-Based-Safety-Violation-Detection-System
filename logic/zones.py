import cv2
import numpy as np

SAFE_ZONE = [(50, 50), (600, 50), (600, 400), (50, 400)]
HIGH_RISK_ZONE = [(650, 50), (1200, 50), (1200, 400), (650, 400)]

def point_in_zone(point, zone):
    return cv2.pointPolygonTest(
        np.array(zone, dtype=np.int32),
        point,
        False
    ) >= 0
