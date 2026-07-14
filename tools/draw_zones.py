"""
Interactive zone editor: trace polygons over a video frame and save them to
config/zones.json as normalized coordinates for a camera.

Usage:
    python tools/draw_zones.py --video videos/test.mp4 --frame 85 --camera_id CAM_STREAM

Controls:
    left click   add a vertex to the current polygon
    c / Enter    close the current polygon (needs >= 3 points), then type its name in the terminal
    u            undo last vertex
    d            delete last completed polygon
    s            save all polygons to the config and exit
    q / Esc      quit without saving
"""

import argparse
import json
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config" / "zones.json"

ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)


def read_frame(video_path, frame_index):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Unable to open video: {video_path}")
    frame = None
    for _ in range(frame_index + 1):
        ret, frame = cap.read()
        if not ret:
            break
    cap.release()
    if frame is None:
        raise SystemExit(f"Could not read frame {frame_index} from {video_path}")
    return frame


def draw_state(base, polygons, current):
    canvas = base.copy()
    for poly in polygons:
        pts = poly["points"]
        for i, pt in enumerate(pts):
            cv2.line(canvas, pt, pts[(i + 1) % len(pts)], ORANGE, 2)
        cv2.putText(canvas, poly["name"], (pts[0][0] + 4, pts[0][1] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1)
    for i, pt in enumerate(current):
        cv2.circle(canvas, pt, 3, WHITE, -1)
        if i > 0:
            cv2.line(canvas, current[i - 1], pt, WHITE, 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=str, default=str(BASE_DIR / "videos" / "test.mp4"))
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--camera_id", type=str, default="CAM_STREAM")
    parser.add_argument("--type", type=str, default="AT_HEIGHT")
    args = parser.parse_args()

    base = read_frame(args.video, args.frame)
    h, w = base.shape[:2]

    polygons = []
    current = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))

    window = f"zones: {args.camera_id} (c=close, u=undo, d=delete, s=save, q=quit)"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    print(__doc__)

    while True:
        cv2.imshow(window, draw_state(base, polygons, current))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            print("Quit without saving.")
            break
        if key == ord("u") and current:
            current.pop()
        if key == ord("d") and polygons:
            removed = polygons.pop()
            print(f"Deleted polygon '{removed['name']}'.")
        if key in (ord("c"), 13):
            if len(current) < 3:
                print("A polygon needs at least 3 points.")
                continue
            name = input(f"Name for this {args.type} polygon: ").strip() or f"zone_{len(polygons) + 1}"
            polygons.append({"name": name, "points": list(current)})
            current.clear()
            print(f"Closed polygon '{name}'.")
        if key == ord("s"):
            if not polygons:
                print("Nothing to save.")
                continue
            config = {}
            if CONFIG_PATH.exists():
                config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            camera = config.setdefault(args.camera_id, {})
            zones = camera.setdefault("zones", [])
            for poly in polygons:
                zones.append({
                    "name": poly["name"],
                    "type": args.type,
                    "polygon": [[round(x / w, 4), round(y / h, 4)] for x, y in poly["points"]],
                })
            CONFIG_PATH.parent.mkdir(exist_ok=True)
            CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
            print(f"Saved {len(polygons)} polygon(s) to {CONFIG_PATH}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
