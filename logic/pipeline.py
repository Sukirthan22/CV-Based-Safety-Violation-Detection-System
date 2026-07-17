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
        return f"Standing near {zone or 'an unprotected edge'}"

    reasons = []
    if violation == "NO_HELMET":
        reasons.append("Helmet missing")
    if violation == "NO_HARNESS":
        reasons.append("Safety harness missing")
    if at_height:
        reasons.append("while working at height")
    return " ".join(reasons)


def _draw_person_labels(frame, person_box, person_violations, ui):
    """Stack each person's own violation labels above their box."""
    if not person_violations:
        return
    x1, y1, x2, y2 = person_box
    h, w = frame.shape[:2]

    seen = []
    for sev, violation, _reason, _event_id in person_violations:
        text = violation.split(":")[0]
        if text not in [t for t, _ in seen]:
            seen.append((text, sev))

    scale = max(0.35, 0.5 * ui)
    (_, th), base = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    row = th + base + 4
    widths = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for t, _ in seen]

    # Keep the label block on screen: nudge left if it would overflow the right
    # edge, and drop it just inside the box when there is no room above.
    lx = max(0, min(x1, w - max(widths) - 8))
    top = y1 - row * len(seen) - 2
    if top < 0:
        top = min(y1 + 2, h - row * len(seen))

    for (text, sev), tw in zip(seen, widths):
        color = (0, 255, 255) if sev == "WARNING" else (0, 0, 255)
        cv2.rectangle(frame, (lx, top), (lx + tw + 6, top + row - 2), (0, 0, 0), -1)
        cv2.putText(
            frame,
            text,
            (lx + 3, top + row - base - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
        )
        top += row


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
    per_person = []
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
        per_person.append((person, person_violations, decide_alert_action(person_violations)))

    alert = decide_alert_action(all_violations)

    for person, person_violations, person_alert in per_person:
        x1, y1, x2, y2 = person["bbox"]
        color = (0, 255, 0)
        if person_alert == "WARNING":
            color = (0, 255, 255)
        elif person_alert == "CRITICAL":
            color = (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        _draw_person_labels(frame, person["bbox"], person_violations, ui)

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
