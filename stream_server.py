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
from logic.tracker import ViolationTracker
from logic.audio import AudioAlerter


BASE_DIR = Path(__file__).resolve().parent


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
    tracker = ViolationTracker(tolerance_seconds=1.5, confirm_seconds=1.5, forget_seconds=10.0, cooldown_seconds=10.0)
    alerter = AudioAlerter()
    
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

        frame, alert, events_to_log = process_frame(frame, camera_id, tracker=tracker)

        if events_to_log:
            for event in events_to_log:
                log_violation(
                    camera_id=camera_id,
                    violations=[event],
                    severity=event[0],
                )
            tracker.mark_logged(events_to_log)
            alerter.process_events(events_to_log)

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
