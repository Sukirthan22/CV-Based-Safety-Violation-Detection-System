import argparse
from datetime import datetime, timedelta

import cv2
from ultralytics import YOLO

from logic.alerts import decide_alert_action
from logic.context import get_person_zone, is_person_at_height
from logic.logger import log_violation
from logic.perception import detect_ppe
from logic.rules import evaluate_ppe_rules


EVENT_CONFIRM_FRAMES = 5
EVENT_COOLDOWN_SECONDS = 8
EVENT_FORGET_FRAMES = 45


parser = argparse.ArgumentParser(description="PPE Monitoring System")
parser.add_argument(
    "--mode",
    type=str,
    default="demo",
    choices=["demo", "video"],
    help="Run mode: demo (webcam) or video (file)",
)
parser.add_argument(
    "--video_path",
    type=str,
    default=None,
    help="Path to video file (used in video mode)",
)

args = parser.parse_args()
if args.mode == "demo":
    cap = cv2.VideoCapture(0)
    camera_id = "CAM_DEMO"
else:
    if args.video_path is None:
        print("Video path not provided")
        raise SystemExit(1)
    cap = cv2.VideoCapture(args.video_path)
    camera_id = "CAM_VIDEO"

model = YOLO("CVBASEDSMS\\CVBASEDSMS\\model\\best.pt")
print("MODEL CLASSES:", model.names)


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
    # Quantized center keeps event ids stable across nearby frames.
    qcx = int(((x1 + x2) / 2) // 20)
    qcy = int(((y1 + y2) / 2) // 20)
    return f"{violation}:{zone}:{qcx}:{qcy}"


def process_frame(frame):
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

    for person in persons:
        zone = get_person_zone(person, w)
        at_height = is_person_at_height(person["bbox"], h)
        violations = evaluate_ppe_rules(person, zone, at_height)
        for sev, violation in violations:
            reason = build_contextual_reason(violation, zone, at_height)
            event_id = _event_id_for_person_violation(person["bbox"], zone, violation)
            all_violations.append((sev, violation, reason, event_id))

    alert = decide_alert_action(all_violations)

    for person in persons:
        x1, y1, x2, y2 = person["bbox"]
        color = (0, 255, 0)
        if alert == "WARNING":
            color = (0, 255, 255)
        elif alert == "CRITICAL":
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


if __name__ == "__main__":
    cap = cv2.VideoCapture("CVBASEDSMS\\CVBASEDSMS\\videos\\test.mp4")
    event_state = {}
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, alert, all_violations = process_frame(frame)
        frame_index += 1
        now = datetime.now()

        current_event_ids = {v[3] for v in all_violations}

        for event_id in current_event_ids:
            state = event_state.get(
                event_id, {"count": 0, "last_seen_frame": -1, "last_logged_at": None}
            )
            if state["last_seen_frame"] == frame_index - 1:
                state["count"] += 1
            else:
                state["count"] = 1
            state["last_seen_frame"] = frame_index
            event_state[event_id] = state

        stale_ids = [
            event_id
            for event_id, state in event_state.items()
            if frame_index - state["last_seen_frame"] > EVENT_FORGET_FRAMES
        ]
        for event_id in stale_ids:
            del event_state[event_id]

        events_to_log = []
        for event_id in current_event_ids:
            state = event_state[event_id]
            cooldown_done = (
                state["last_logged_at"] is None
                or (now - state["last_logged_at"]) >= timedelta(seconds=EVENT_COOLDOWN_SECONDS)
            )
            if state["count"] >= EVENT_CONFIRM_FRAMES and cooldown_done:
                events_to_log.append(event_id)

        if events_to_log:
            violations_to_log = [v for v in all_violations if v[3] in events_to_log]
            if violations_to_log:
                log_violation(
                    camera_id=camera_id,
                    violations=violations_to_log,
                    severity=alert,
                )
                for event_id in events_to_log:
                    event_state[event_id]["last_logged_at"] = now
                print("LOGGED:", alert, [(v[1], v[3]) for v in violations_to_log])

        cv2.imshow("PPE Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

