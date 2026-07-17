import argparse
from datetime import datetime, timedelta
from pathlib import Path

import cv2

from logic.logger import log_violation_start, log_violation_end, close_all_open_violations
from logic.pipeline import process_frame
from logic.tracker import ViolationTracker
from logic.audio import AudioAlerter


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
    tracker = ViolationTracker(tolerance_seconds=1.5, confirm_seconds=1.5, forget_seconds=10.0, cooldown_seconds=10.0)
    alerter = AudioAlerter()

    # Clean up any dangling "Open" violations from previous crashes
    close_all_open_violations()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, alert, events_to_speak, events_started, events_ended = process_frame(frame, camera_id, tracker=tracker)

            if events_started:
                for event in events_started:
                    log_violation_start(
                        camera_id=camera_id,
                        person_id=event[4],
                        violation=event[1],
                        severity=event[0]
                    )
                tracker.mark_started(events_started)
                print("STARTED:", [(v[0], v[1], v[3]) for v in events_started])
                
            if events_ended:
                for event in events_ended:
                    log_violation_end(
                        camera_id=camera_id,
                        person_id=event[4],
                        violation=event[1]
                    )
                print("ENDED:", [(v[0], v[1], v[3]) for v in events_ended])

            if events_to_speak:
                alerter.process_events(events_to_speak)
                tracker.mark_spoken(events_to_speak)

            cv2.imshow("PPE Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        close_all_open_violations()
        cap.release()
        cv2.destroyAllWindows()

