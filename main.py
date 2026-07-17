import argparse
from datetime import datetime, timedelta
from pathlib import Path

import cv2

from logic.logger import log_violation
from logic.pipeline import process_frame


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
        video_candidate = Path("videos/test.mp4")
        if video_candidate.exists():
            args.video_path = str(video_candidate)
        else:
            print("Video path not provided")
            raise SystemExit(1)
    cap = cv2.VideoCapture(args.video_path)
    camera_id = "CAM_VIDEO"


if __name__ == "__main__":
    # cap is already initialized based on args
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

