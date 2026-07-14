# CV-Based Safety Monitoring System (CVBSMS)

Real-time construction-site PPE compliance monitoring. A YOLO model detects
workers, helmets, and safety harnesses in a live video feed; a symbolic rules
layer converts raw detections into **context-aware safety violations**
(e.g. *no harness while working at height in a high-risk zone*), debounces
them over time, and surfaces them on a live web dashboard with severity-based
audio alerts.

## Architecture

```
 Camera / RTSP / video file
            │
            ▼
 ┌─────────────────────────┐
 │  stream_server.py       │   frame producer thread
 │                         │
 │  logic/perception.py    │   YOLO inference + PPE-to-person association
 │  logic/context.py       │   at-height check (per-camera polygons)
 │  logic/rules.py         │   violation rules (declarative safety policy)
 │  logic/alerts.py        │   severity escalation (INFO/WARNING/CRITICAL)
 │  temporal debouncing    │   confirm ≥5 frames, 8s cooldown per event
 │  logic/logger.py        │   append confirmed events → logs/violations.csv
 └───────┬─────────┬───────┘
         │         │
   MJPEG /stream   JSON /status        logs/violations.csv
         │         │                          │
         ▼         ▼                          ▼
 ┌──────────────────────────────────────────────────┐
 │  next-dashboard  (Next.js, http://localhost:3000)│
 │  live feed · alert badge · KPIs · incident table │
 │  audio siren/buzzer on critical/warning events   │
 └──────────────────────────────────────────────────┘
```

Design principle: the neural network only *perceives*; whether something is a
violation, and how severe it is, is decided by small auditable rule functions
in [`logic/`](logic/). Safety policy can be changed without retraining.

### Key components

| Path | Role |
|---|---|
| `stream_server.py` | Main entry point. Reads a video file or RTSP stream, runs the pipeline, serves MJPEG at `/stream` and alert state at `/status` (port 8000). |
| `logic/pipeline.py` | Per-frame orchestration: detect → contextualize → evaluate rules → annotate frame. |
| `logic/perception.py` | YOLO inference; associates helmets (head region) and harnesses (torso region) to each detected person. |
| `logic/rules.py` | Violation rules: no helmet → WARNING; no harness at height → CRITICAL. |
| `config/zones.json` | Per-camera AT_HEIGHT polygons (normalized coordinates) marking elevated work surfaces in the image. |
| `tools/draw_zones.py` | Interactive editor: trace zone polygons on a video frame by clicking, saves to `config/zones.json`. |
| `next-dashboard/` | Next.js dashboard: live feed, filters, KPIs, incident history from the CSV via `/api/violations`. |
| `dashboard.py` | Earlier Streamlit prototype of the dashboard (kept for reference). |
| `logs/violations.csv` | Confirmed violation events (`timestamp, camera_id, violations, severity`). |

### At-height detection

Elevation cannot be read from a single 2D frame, but fixed cameras allow
encoding it as **site knowledge**: elevated work surfaces (scaffolds, girders,
platforms) occupy fixed image regions, so each camera gets AT_HEIGHT polygons
in [`config/zones.json`](config/zones.json). A person is *at height* iff their
feet point (bbox bottom-center) falls inside one — tested with
`cv2.pointPolygonTest`. Polygons are stored normalized, so they survive
resolution changes; new cameras are onboarded by tracing polygons with
[`tools/draw_zones.py`](tools/draw_zones.py) — no code changes, no retraining.
Cameras without a polygon config fall back to a bbox-position heuristic.

### False-positive suppression

Raw per-frame detections are noisy, so events are debounced before logging:
a violation must persist for **5 consecutive frames** to be confirmed, an
identical event is not re-logged within an **8-second cooldown**, and event
state is forgotten after **45 frames** without re-detection.

## Setup

Requires Python 3.10+ and Node.js 18+.

```bash
# Python side
pip install -r requirements.txt

# Dashboard
cd next-dashboard
npm install
```

**Model weights and demo videos are not committed** (see `.gitignore`).
Place your trained YOLO weights at `model/best.pt` (classes: person, helmet,
harness — class names are matched case-insensitively, see
`logic/perception.py:_class_groups`) and a demo clip at `videos/test.mp4`.

## Running

1. **Start the detection + stream server** (port 8000):

   ```bash
   # Loop a video file (default: videos/test.mp4)
   python stream_server.py --mode file --video_path videos/test.mp4

   # Or ingest a live RTSP camera
   python stream_server.py --mode rtsp --rtsp_url rtsp://<host>:8554/live --camera_id CAM_SITE_1
   ```

2. **Start the dashboard** (port 3000):

   ```bash
   cd next-dashboard
   npm run dev
   ```

   Open http://localhost:3000 — the live annotated feed, current alert
   status, and incident history update automatically.

Alternatively, the Streamlit prototype runs standalone:
`streamlit run dashboard.py`.

## Current demo scoping

- At-height polygons are a 2D image-space test: a person standing on the
  ground directly in front of an elevated surface can be falsely flagged, and
  polygons must be redrawn if the camera moves. Workers on *moving* elevated
  platforms (e.g. a suspended segment) are covered only while the platform
  stays within its traced polygon.

## Roadmap

- ByteTrack track IDs (via `ultralytics` tracking) to replace the quantized
  bbox-center event identity.
- Additional polygon zone types (e.g. restricted/exclusion areas) using the
  same per-camera config mechanism as at-height.
- SQLite event store replacing the CSV log.
- Model evaluation harness (mAP, per-class precision/recall, confusion matrix).
