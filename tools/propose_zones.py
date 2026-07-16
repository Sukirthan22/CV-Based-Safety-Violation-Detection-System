"""
AI-assisted zone calibration: propose AT_HEIGHT polygons for a camera by
matching text prompts (config/zone_prompts.txt) against a video frame with
an open-vocabulary segmentation model (YOLOE), then write them to
config/zones.json with status "proposed".

Proposed zones are IGNORED by the runtime until a human approves them:

    python tools/propose_zones.py --video videos/demo.mp4 --frame 40 --camera_id CAM_DEMO
    python tools/draw_zones.py --review --camera_id CAM_DEMO --video videos/demo.mp4 --frame 40

A preview image of the proposals is saved next to the config for a quick look.
Model weights (~600 MB total) auto-download to the repo root on first run and
are gitignored.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config" / "zones.json"
PROMPTS_PATH = BASE_DIR / "config" / "zone_prompts.txt"

ORANGE = (0, 165, 255)


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


def read_prompts(path):
    if not Path(path).exists():
        raise SystemExit(f"Prompts file not found: {path}")
    prompts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    if not prompts:
        raise SystemExit(f"No prompts in {path} (all empty or commented out).")
    return prompts


def mask_to_polygon(poly_xy, frame_shape, pad_px):
    """Rasterize a mask outline, pad it, and simplify to a few corners."""
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_xy.astype(np.int32)], 255)
    if pad_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad_px + 1, 2 * pad_px + 1))
        mask = cv2.dilate(mask, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    if len(approx) < 3:
        return None
    return [[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in approx.reshape(-1, 2)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=str, default=str(BASE_DIR / "videos" / "demo.mp4"))
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--camera_id", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, default=str(PROMPTS_PATH))
    parser.add_argument("--prompts", type=str, default=None,
                        help="Comma-separated override of the prompts file (one-off experiments)")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Minimum detection confidence for a proposal (default 0.3)")
    parser.add_argument("--pad", type=int, default=10,
                        help="Pixels to pad each mask outward so detected feet points stay inside")
    parser.add_argument("--weights", type=str, default="yoloe-11s-seg.pt")
    parser.add_argument("--type", type=str, default="AT_HEIGHT")
    args = parser.parse_args()

    if args.prompts:
        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    else:
        prompts = read_prompts(args.prompts_file)

    frame = read_frame(args.video, args.frame)

    print(f"Prompts: {prompts}")
    print("Loading open-vocabulary model (first run downloads weights)...")
    from ultralytics import YOLOE

    model = YOLOE(args.weights)
    model.set_classes(prompts, model.get_text_pe(prompts))
    result = model.predict(frame, conf=args.conf, verbose=False)[0]

    proposals = []
    preview = frame.copy()
    if result.masks is not None:
        for i, box in enumerate(result.boxes):
            prompt = prompts[int(box.cls[0])]
            conf = float(box.conf[0])
            polygon = mask_to_polygon(result.masks.xy[i], frame.shape, args.pad)
            if polygon is None:
                continue
            proposals.append({
                "name": f"{prompt.replace(' ', '_')}_{i + 1}",
                "type": args.type,
                "polygon": polygon,
                "status": "proposed",
                "source": f"yoloe:{prompt} conf={conf:.2f} frame={args.frame}",
            })
            h, w = frame.shape[:2]
            pts = np.array([[int(x * w), int(y * h)] for x, y in polygon], dtype=np.int32)
            cv2.polylines(preview, [pts], True, ORANGE, 2)
            cv2.putText(preview, f"{prompt} {conf:.2f}", tuple(pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ORANGE, 2)

    if not proposals:
        raise SystemExit(
            f"No regions matched above conf={args.conf}. Try lowering --conf, editing "
            f"{args.prompts_file}, or trace manually: python tools/draw_zones.py --assist"
        )

    config = {}
    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    camera = config.setdefault(args.camera_id, {})
    zones = camera.setdefault("zones", [])
    # Re-running replaces earlier proposals instead of stacking duplicates.
    zones[:] = [z for z in zones if z.get("status", "approved") != "proposed"]
    zones.extend(proposals)
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    preview_path = BASE_DIR / "config" / f"proposed_{args.camera_id}.png"
    cv2.imwrite(str(preview_path), preview)

    print(f"\nWrote {len(proposals)} proposed zone(s) for {args.camera_id} to {CONFIG_PATH}")
    print(f"Preview image: {preview_path}")
    print("They are INACTIVE until approved. Review with:")
    print(f"  python tools/draw_zones.py --review --camera_id {args.camera_id} "
          f"--video {args.video} --frame {args.frame}")


if __name__ == "__main__":
    main()
