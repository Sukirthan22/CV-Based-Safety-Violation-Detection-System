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

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (int(0.6 * w), h), (0, 255, 0), -1)
    cv2.rectangle(overlay, (int(0.6 * w), 0), (w, h), (0, 0, 255), -1)
    alpha = 0.15
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.putText(
        frame,
        "SAFE ZONE",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        "HIGH RISK ZONE",
        (int(0.6 * w) + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    persons = detect_ppe(frame, model)

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
        reasons = ", ".join([v[1] for v in all_violations])
        cv2.putText(
            frame,
            f"{alert}: {reasons}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3,
        )

    cv2.putText(frame, "SAFE", (w - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(
        frame,
        "WARNING",
        (w - 180, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "CRITICAL",
        (w - 180, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    return frame, alert, all_violations
