import argparse
import json
import threading
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from logic.logger import log_violation
from logic.pipeline import process_frame


BASE_DIR = Path(__file__).resolve().parent

EVENT_CONFIRM_FRAMES = 15
EVENT_COOLDOWN_SECONDS = 60
EVENT_FORGET_FRAMES = 120


class StreamState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.alert = "INFO"
        self.updated_at = time.time()

    def set_frame(self, data):
        with self.lock:
            self.frame = data

    def get_frame(self):
        with self.lock:
            return self.frame

    def set_alert(self, alert):
        with self.lock:
            self.alert = alert
            self.updated_at = time.time()

    def get_status(self):
        with self.lock:
            return self.alert, self.updated_at


def frame_producer(source, fps, state, camera_id, mode):
    if mode == "rtsp":
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open RTSP stream: {source}")
    else:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {source}")

    frame_interval = 1.0 / max(fps, 1)
    event_state = {}
    frame_index = 0
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            if mode == "file":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            cap.release()
            time.sleep(1.0)
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            continue

        frame, alert, all_violations = process_frame(frame, camera_id)
        frame_index += 1
        now = datetime.now()

        current_event_ids = {v[3] for v in all_violations}
        for event_id in current_event_ids:
            state_item = event_state.get(
                event_id, {"count": 0, "last_seen_frame": -1, "last_logged_at": None}
            )
            if state_item["last_seen_frame"] == frame_index - 1:
                state_item["count"] += 1
            else:
                state_item["count"] = 1
            state_item["last_seen_frame"] = frame_index
            event_state[event_id] = state_item

        stale_ids = [
            event_id
            for event_id, state_item in event_state.items()
            if frame_index - state_item["last_seen_frame"] > EVENT_FORGET_FRAMES
        ]
        for event_id in stale_ids:
            del event_state[event_id]

        events_to_log = []
        for event_id in current_event_ids:
            state_item = event_state[event_id]
            cooldown_done = (
                state_item["last_logged_at"] is None
                or (now - state_item["last_logged_at"])
                >= timedelta(seconds=EVENT_COOLDOWN_SECONDS)
            )
            if state_item["count"] >= EVENT_CONFIRM_FRAMES and cooldown_done:
                events_to_log.append(event_id)

        if events_to_log:
            violations_to_log = [v for v in all_violations if v[3] in events_to_log]
            if violations_to_log:
                # Severity of the logged batch, not of the whole frame — a
                # WARNING event must not inherit CRITICAL from bystanders.
                severity = (
                    "CRITICAL"
                    if any(v[0] == "CRITICAL" for v in violations_to_log)
                    else violations_to_log[0][0]
                )
                log_violation(
                    camera_id=camera_id,
                    violations=violations_to_log,
                    severity=severity,
                )
                for event_id in events_to_log:
                    event_state[event_id]["last_logged_at"] = now

        state.set_alert(alert)
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            state.set_frame(encoded.tobytes())

        elapsed = time.time() - start
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/status":
            alert, updated_at = self.server.state.get_status()
            payload = json.dumps(
                {"alert": alert, "updated_at": updated_at}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path != "/stream":
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            while True:
                frame = self.server.state.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return


def main():
    parser = argparse.ArgumentParser(description="MJPEG stream server with detections")
    parser.add_argument("--mode", choices=["file", "rtsp"], default="file")
    parser.add_argument(
        "--video_path",
        type=str,
        default=str(BASE_DIR / "videos" / "test.mp4"),
        help="Path to video file (mode=file)",
    )
    parser.add_argument(
        "--rtsp_url",
        type=str,
        default="rtsp://localhost:8554/live",
        help="RTSP URL (mode=rtsp)",
    )
    parser.add_argument("--camera_id", type=str, default="CAM_STREAM")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    state = StreamState()
    source = args.video_path if args.mode == "file" else args.rtsp_url
    thread = threading.Thread(
        target=frame_producer,
        args=(source, args.fps, state, args.camera_id, args.mode),
        daemon=True,
    )
    thread.start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), StreamHandler)
    server.state = state
    print(f"Streaming on http://localhost:{args.port}/stream")
    server.serve_forever()


if __name__ == "__main__":
    main()
