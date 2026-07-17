"""
Interactive zone editor: trace polygons over a video frame and save them to
config/zones.json as normalized coordinates for a camera.

Modes:
  (default)  manual tracing — click each vertex yourself
  --assist   one-click tracing — each left click runs SAM at that point and
             traces the surface outline for you (then close/name as usual)
  --review   approve or reject zones proposed by tools/propose_zones.py

Usage:
    python tools/draw_zones.py --video videos/test.mp4 --frame 85 --camera_id CAM_STREAM
    python tools/draw_zones.py --assist --video videos/demo.mp4 --frame 40 --camera_id CAM_DEMO
    python tools/draw_zones.py --review --camera_id CAM_DEMO --video videos/demo.mp4 --frame 40

Controls (editor):
    left click   add a vertex (manual) / trace the clicked surface (--assist)
    c / Enter    close the current polygon (needs >= 3 points), then type its name in the terminal
    u            undo last vertex (manual) / discard the current trace (--assist)
    d            delete last completed polygon
    s            save all polygons to the config and exit
    q / Esc      quit without saving

Controls (--review):
    a            approve the highlighted proposal (activates it)
    d            reject the highlighted proposal (removes it)
    q / Esc      quit without saving decisions
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config" / "zones.json"

ORANGE = (0, 165, 255)
GREEN = (0, 200, 0)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)


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


def load_config():
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


def save_config(config):
    CONFIG_PATH.parent.mkdir(exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def polygon_to_px(polygon, w, h):
    return np.array([[int(x * w), int(y * h)] for x, y in polygon], dtype=np.int32)


class SamTracer:
    """Lazy-loaded one-click surface tracer (used with --assist)."""

    def __init__(self, weights, pad_px):
        self.weights = weights
        self.pad_px = pad_px
        self.model = None

    def trace(self, frame, point):
        if self.model is None:
            print("Loading SAM (first run downloads weights)...")
            from ultralytics import SAM

            self.model = SAM(self.weights)
        result = self.model(frame, points=[list(point)], labels=[1], verbose=False)[0]
        if result.masks is None or len(result.masks.data) == 0:
            print("SAM returned no mask for that click.")
            return []
        h, w = frame.shape[:2]
        mask = (result.masks.data[0].cpu().numpy() > 0).astype(np.uint8) * 255
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if self.pad_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * self.pad_px + 1, 2 * self.pad_px + 1)
            )
            mask = cv2.dilate(mask, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        contour = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        return [tuple(pt) for pt in approx.reshape(-1, 2)]


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
    if len(current) > 2:
        cv2.line(canvas, current[-1], current[0], WHITE, 1)
    return canvas


def run_editor(args):
    base = read_frame(args.video, args.frame)
    h, w = base.shape[:2]

    tracer = SamTracer(args.sam_weights, args.pad) if args.assist else None
    polygons = []
    current = []

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if tracer is not None:
            print(f"Tracing surface at ({x},{y})...")
            current[:] = tracer.trace(base, (x, y))
            print(f"Traced {len(current)} points. c=accept, u=discard, or click elsewhere.")
        else:
            current.append((x, y))

    mode = "assist" if args.assist else "manual"
    window = f"zones[{mode}]: {args.camera_id} (c=close, u=undo, d=delete, s=save, q=quit)"
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
            if tracer is not None:
                current.clear()
            else:
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
            config = load_config()
            camera = config.setdefault(args.camera_id, {})
            zones = camera.setdefault("zones", [])
            for poly in polygons:
                zones.append({
                    "name": poly["name"],
                    "type": args.type,
                    "polygon": [[round(x / w, 4), round(y / h, 4)] for x, y in poly["points"]],
                })
            save_config(config)
            print(f"Saved {len(polygons)} polygon(s) to {CONFIG_PATH}")
            break

    cv2.destroyAllWindows()


def run_review(args):
    config = load_config()
    camera = config.get(args.camera_id, {})
    zones = camera.get("zones", [])
    proposed = [z for z in zones if z.get("status", "approved") == "proposed"]
    if not proposed:
        raise SystemExit(f"No proposed zones for {args.camera_id}. Run tools/propose_zones.py first.")

    base = read_frame(args.video, args.frame)
    h, w = base.shape[:2]
    window = f"review: {args.camera_id} (a=approve, d=reject, q=quit)"
    cv2.namedWindow(window)

    decisions = {}  # index in `proposed` -> "approved" | "rejected"
    idx = 0
    while idx < len(proposed):
        canvas = base.copy()
        for z in zones:
            if z.get("status", "approved") == "approved":
                cv2.polylines(canvas, [polygon_to_px(z["polygon"], w, h)], True, GREEN, 2)
        for j, z in enumerate(proposed):
            if decisions.get(j) == "rejected":
                continue
            color = YELLOW if j == idx else ORANGE
            thickness = 3 if j == idx else 1
            pts = polygon_to_px(z["polygon"], w, h)
            cv2.polylines(canvas, [pts], True, color, thickness)
            if j == idx:
                label = f"{z['name']} ({z.get('source', 'proposed')})  a=approve d=reject"
                cv2.putText(canvas, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            print("Quit without saving decisions.")
            cv2.destroyAllWindows()
            return
        if key == ord("a"):
            decisions[idx] = "approved"
            print(f"Approved '{proposed[idx]['name']}'.")
            idx += 1
        if key == ord("d"):
            decisions[idx] = "rejected"
            print(f"Rejected '{proposed[idx]['name']}'.")
            idx += 1
    cv2.destroyAllWindows()

    kept = []
    for z in zones:
        if z.get("status", "approved") != "proposed":
            kept.append(z)
            continue
        verdict = decisions.get(proposed.index(z))
        if verdict == "approved":
            z["status"] = "approved"
            kept.append(z)
        # rejected proposals are dropped
    camera["zones"] = kept
    save_config(config)
    approved = sum(1 for v in decisions.values() if v == "approved")
    print(f"Saved: {approved} approved, {len(decisions) - approved} rejected -> {CONFIG_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=str, default=str(BASE_DIR / "videos" / "test.mp4"))
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--camera_id", type=str, default="CAM_STREAM")
    parser.add_argument("--type", type=str, default="AT_HEIGHT")
    parser.add_argument("--assist", action="store_true",
                        help="One-click tracing: SAM outlines the surface you click")
    parser.add_argument("--review", action="store_true",
                        help="Approve/reject zones proposed by tools/propose_zones.py")
    parser.add_argument("--pad", type=int, default=10,
                        help="Pixels to pad assisted traces outward (default 10)")
    parser.add_argument("--sam_weights", type=str, default="sam2.1_b.pt")
    args = parser.parse_args()

    if args.review:
        run_review(args)
    else:
        run_editor(args)


if __name__ == "__main__":
    main()
