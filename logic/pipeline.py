from pathlib import Path

import cv2
from ultralytics import YOLO

from logic.alerts import decide_alert_action
from logic.context import get_person_zone, is_person_at_height
from logic.perception import detect_ppe
from logic.rules import evaluate_ppe_rules


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model" / "best.pt"

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO(str(MODEL_PATH))
        print("MODEL CLASSES:", _MODEL.names)
    return _MODEL


def build_contextual_reason(violation, zone, at_height):
    reasons = []
    if violation == "NO_HELMET":
        reasons.append("Helmet missing")
    if violation == "NO_HARNESS":
        reasons.append("Safety harness missing")
    if zone == "HIGH_RISK":
        reasons.append("in HIGH-RISK zone")
    if at_height:
        reasons.append("while working at height")
    return " ".join(reasons)


def _event_id_for_person_violation(person_bbox, zone, violation):
    x1, y1, x2, y2 = person_bbox
    qcx = int(((x1 + x2) / 2) // 20)
    qcy = int(((y1 + y2) / 2) // 20)
    return f"{violation}:{zone}:{qcx}:{qcy}"


def process_frame(frame):
    model = get_model()
    h, w, _ = frame.shape
    all_violations = []

    # Inference runs on the untouched frame; zone markings are drawn after.
    persons = detect_ppe(frame, model)

    # Overlay text scales with frame width so portrait and landscape both stay readable.
    ui = max(0.4, min(0.7, w / 900))
    thick = 1 if ui < 0.55 else 2
    label_y = int(34 * ui) + 4

    boundary_x = int(0.6 * w)
    cv2.line(frame, (boundary_x, 0), (boundary_x, h), (0, 0, 255), 2)
    cv2.putText(
        frame,
        "SAFE ZONE",
        (10, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        ui,
        (0, 255, 0),
        thick,
    )
    cv2.putText(
        frame,
        "HIGH RISK ZONE",
        (boundary_x + 10, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        ui,
        (0, 0, 255),
        thick,
    )

    person_alerts = []
    for person in persons:
        zone = get_person_zone(person, w)
        at_height = is_person_at_height(person["bbox"], h)
        violations = evaluate_ppe_rules(person, zone, at_height)
        person_violations = []
        for sev, violation in violations:
            reason = build_contextual_reason(violation, zone, at_height)
            event_id = _event_id_for_person_violation(person["bbox"], zone, violation)
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
            if v[1] not in unique_reasons:
                unique_reasons.append(v[1])
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

    legend = [("SAFE", (0, 255, 0)), ("WARNING", (0, 255, 255)), ("CRITICAL", (0, 0, 255))]
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
