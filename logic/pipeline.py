from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from logic.alerts import decide_alert_action
from logic.context import get_person_edge_zone, is_person_at_height
from logic.perception import detect_ppe
from logic.rules import NEAR_EDGE, evaluate_ppe_rules
from logic.zones import get_at_height_zones, get_edge_zones, scale_polygon


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model" / "best.pt"

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO(str(MODEL_PATH))
        print("MODEL CLASSES:", _MODEL.names)
    return _MODEL


def build_contextual_reason(violation, at_height):
    # NEAR_EDGE carries its zone as "NEAR_EDGE:<zone label>" so the log and the
    # dashboard can say which edge, not just that there was one.
    if violation.startswith(NEAR_EDGE):
        _, _, zone = violation.partition(":")
        return f"Standing near the open edge of {zone or 'an elevated surface'}"

    reasons = []
    if violation == "NO_HELMET":
        reasons.append("Helmet missing")
    if violation == "NO_HARNESS":
        reasons.append("Safety harness missing")
    if at_height:
        reasons.append("while working at height")
    return " ".join(reasons)


def _draw_zone_outlines(frame, zones, frame_size, color, label, ui):
    for zone_cfg in zones or []:
        pts = scale_polygon(zone_cfg["polygon"], frame_size)
        cv2.polylines(
            frame,
            [np.array(pts, dtype=np.int32)],
            isClosed=True,
            color=color,
            thickness=1,
        )
        zx, zy = pts[0]
        cv2.putText(
            frame,
            label,
            (zx + 4, zy + int(16 * ui)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75 * ui,
            color,
            1,
        )


def _event_id_for_person_violation(person_bbox, violation):
    x1, y1, x2, y2 = person_bbox
    qcx = int(((x1 + x2) / 2) // 20)
    qcy = int(((y1 + y2) / 2) // 20)
    return f"{violation}:{qcx}:{qcy}"


def process_frame(frame, camera_id="CAM_STREAM"):
    model = get_model()
    h, w, _ = frame.shape
    all_violations = []

    # Inference runs on the untouched frame; overlays are drawn after.
    persons = detect_ppe(frame, model)

    # Overlay text scales with frame width so portrait and landscape both stay readable.
    ui = max(0.4, min(0.7, w / 900))
    thick = 1 if ui < 0.55 else 2
    label_y = int(34 * ui) + 4

    _draw_zone_outlines(
        frame, get_at_height_zones(camera_id), (w, h), (0, 165, 255), "AT HEIGHT", ui
    )
    _draw_zone_outlines(
        frame, get_edge_zones(camera_id), (w, h), (255, 0, 255), "EDGE", ui
    )
    person_alerts = []
    for person in persons:
        at_height = is_person_at_height(person["bbox"], (h, w), camera_id)
        edge_zone = get_person_edge_zone(person["bbox"], (h, w), camera_id)
        violations = evaluate_ppe_rules(person, at_height, edge_zone)
        person_violations = []
        for sev, violation in violations:
            reason = build_contextual_reason(violation, at_height)
            event_id = _event_id_for_person_violation(person["bbox"], violation)
            person_violations.append((sev, violation, reason, event_id))
        all_violations.extend(person_violations)
        person_alerts.append(decide_alert_action(person_violations))

    alert = decide_alert_action(all_violations)

    for person, person_alert in zip(persons, person_alerts):
        x1, y1, x2, y2 = person["bbox"]
        color = (0, 255, 0)
        if person_alert == "WARNING":
            color = (0, 255, 255)
        elif person_alert == "CRITICAL":
            color = (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if alert != "INFO":
        unique_reasons = []
        for v in all_violations:
            # Drop the ":<zone>" suffix so the banner stays short; the zone is
            # named in the logged reason and on the dashboard.
            label = v[1].split(":")[0]
            if label not in unique_reasons:
                unique_reasons.append(label)
        banner = f"{alert}: {', '.join(unique_reasons)}"
        banner_color = (0, 255, 255) if alert == "WARNING" else (0, 0, 255)
        bscale = ui
        (tw, th), base = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, bscale, thick)
        max_tw = w - 20
        if tw > max_tw:
            bscale = bscale * max_tw / tw
            (tw, th), base = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, bscale, thick)
        bx, by = 10, h - 10
        cv2.rectangle(frame, (bx - 6, by - th - base - 6), (bx + tw + 6, by + 4), (0, 0, 0), -1)
        cv2.putText(
            frame,
            banner,
            (bx, by - base),
            cv2.FONT_HERSHEY_SIMPLEX,
            bscale,
            banner_color,
            thick,
        )

    legend = [("COMPLIANT", (0, 255, 0)), ("WARNING", (0, 255, 255)), ("CRITICAL", (0, 0, 255))]
    legend_scale = 0.85 * ui
    ly = label_y + int(28 * ui)
    for text, color in legend:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, legend_scale, thick)
        cv2.putText(
            frame,
            text,
            (w - tw - 10, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            legend_scale,
            color,
            thick,
        )
        ly += int(26 * ui)

    return frame, alert, all_violations
