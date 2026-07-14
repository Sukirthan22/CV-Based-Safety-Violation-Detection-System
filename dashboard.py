from datetime import datetime, timedelta
from pathlib import Path
import time

import cv2
import pandas as pd
import streamlit as st

from logic.logger import log_violation
from logic.pipeline import process_frame


BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "logs" / "violations.csv"
VIDEO_PATH = BASE_DIR / "videos" / "test.mp4"
CAMERA_ID = "CAM_DASHBOARD"
EVENT_CONFIRM_FRAMES = 5
EVENT_COOLDOWN_SECONDS = 8
EVENT_FORGET_FRAMES = 45


st.set_page_config(page_title="KRUU Safety Monitor", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg: #f5f7fa;
  --surface: #ffffff;
  --surface-muted: #f9fafb;
  --border: #e5e7eb;
  --text: #0f172a;
  --muted: #64748b;
  --primary: #0b5fff;
  --success: #0f766e;
  --warning: #b45309;
  --critical: #b42318;
  --shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
}

html, body, [class*="css"] {
  font-family: "IBM Plex Sans", "Manrope", "Source Sans 3", sans-serif;
  color: var(--text);
}

.stApp {
  background: radial-gradient(1200px 600px at 10% -10%, #e6f0ff 0, #f5f7fa 35%, #f8fafc 100%);
}

[data-testid="stHeader"] {
  visibility: hidden;
}

.page {
  padding: 8px 6px 24px 6px;
}

.topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  box-shadow: var(--shadow);
}

.title {
  font-size: 22px;
  font-weight: 600;
  letter-spacing: 0.2px;
}

.subtitle {
  color: var(--muted);
  font-size: 12px;
  margin-top: 2px;
}

.badge {
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  border: 1px solid var(--border);
  background: var(--surface-muted);
  color: var(--muted);
}

.badge-critical {
  border-color: #fecaca;
  background: #fef2f2;
  color: var(--critical);
}

.badge-warning {
  border-color: #fed7aa;
  background: #fff7ed;
  color: var(--warning);
}

.badge-info {
  border-color: #cbd5e1;
  background: #f8fafc;
  color: var(--muted);
}

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  box-shadow: var(--shadow);
  padding: 14px 16px;
}

.kpi-title {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
}

.kpi-value {
  font-size: 26px;
  font-weight: 600;
  margin-top: 6px;
}

.section-title {
  font-size: 14px;
  font-weight: 600;
  margin: 12px 0 8px 0;
}

.table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.table th {
  text-align: left;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  padding: 10px 8px;
  border-bottom: 1px solid var(--border);
}

.table td {
  padding: 10px 8px;
  border-bottom: 1px solid var(--border);
}

.chip {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
}

.chip-warning {
  background: #fff7ed;
  color: var(--warning);
  border: 1px solid #fed7aa;
}

.chip-critical {
  background: #fef2f2;
  color: var(--critical);
  border: 1px solid #fecaca;
}

.chip-info {
  background: #f1f5f9;
  color: var(--muted);
  border: 1px solid #e2e8f0;
}
</style>
""",
    unsafe_allow_html=True,
)


def load_violations():
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "camera_id", "violations", "severity"])
    df = pd.read_csv(LOG_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["timestamp"])
    return df

def log_confirmed_events(all_violations, alert):
    if "event_state" not in st.session_state:
        st.session_state.event_state = {}
    st.session_state.frame_index = st.session_state.get("frame_index", 0) + 1

    event_state = st.session_state.event_state
    frame_index = st.session_state.frame_index
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
            or (now - state_item["last_logged_at"]) >= timedelta(seconds=EVENT_COOLDOWN_SECONDS)
        )
        if state_item["count"] >= EVENT_CONFIRM_FRAMES and cooldown_done:
            events_to_log.append(event_id)

    if not events_to_log:
        return

    violations_to_log = [v for v in all_violations if v[3] in events_to_log]
    if not violations_to_log:
        return

    log_violation(
        camera_id=CAMERA_ID,
        violations=violations_to_log,
        severity=alert,
    )
    for event_id in events_to_log:
        event_state[event_id]["last_logged_at"] = now


def badge_for_alert(alert):
    if alert == "CRITICAL":
        return '<span class="badge badge-critical">CRITICAL</span>'
    if alert == "WARNING":
        return '<span class="badge badge-warning">WARNING</span>'
    return '<span class="badge badge-info">INFO</span>'


def chip_for_severity(sev):
    s = str(sev).upper()
    if "CRITICAL" in s:
        return '<span class="chip chip-critical">CRITICAL</span>'
    if "WARNING" in s:
        return '<span class="chip chip-warning">WARNING</span>'
    return '<span class="chip chip-info">INFO</span>'


def kpi_card(title, value):
    st.markdown(
        f"""
<div class="card">
  <div class="kpi-title">{title}</div>
  <div class="kpi-value">{value}</div>
</div>
""",
        unsafe_allow_html=True,
    )


filters = st.columns([1.2, 1.2, 1.2, 2.4])
df_all = load_violations()

with filters[0]:
    camera_options = ["All"] + sorted(df_all["camera_id"].dropna().unique().tolist())
    camera_choice = st.selectbox("Camera", camera_options)
with filters[1]:
    severity_choice = st.selectbox("Severity", ["All", "CRITICAL", "WARNING", "INFO"])
with filters[2]:
    range_choice = st.selectbox("Time Range", ["Last 15 min", "Last 1 hour", "Last 24 hours", "All"])
with filters[3]:
    st.markdown(
        '<div class="subtitle" style="padding-top: 18px;">Filters update the metrics and incidents table</div>',
        unsafe_allow_html=True,
    )

df = df_all.copy()
now = pd.Timestamp.now()
if range_choice == "Last 15 min":
    df = df[df["timestamp"] >= now - pd.Timedelta(minutes=15)]
elif range_choice == "Last 1 hour":
    df = df[df["timestamp"] >= now - pd.Timedelta(hours=1)]
elif range_choice == "Last 24 hours":
    df = df[df["timestamp"] >= now - pd.Timedelta(hours=24)]

if camera_choice != "All":
    df = df[df["camera_id"] == camera_choice]
if severity_choice != "All":
    df = df[df["severity"].astype(str).str.upper().str.contains(severity_choice)]


if "cap" not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(str(VIDEO_PATH))
cap = st.session_state.cap

ret, frame = cap.read()
if not ret:
    st.warning("No frame received")
    st.stop()

frame, alert, all_violations = process_frame(frame)
log_confirmed_events(all_violations, alert)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

st.markdown('<div class="page">', unsafe_allow_html=True)

col_left, col_right = st.columns([3, 1.5])
with col_left:
    st.markdown(
        f"""
<div class="topbar">
  <div>
    <div class="title">Construction Safety Monitor</div>
    <div class="subtitle">Live PPE compliance and incident tracking</div>
  </div>
  <div>{badge_for_alert(alert)}</div>
</div>
""",
        unsafe_allow_html=True,
    )

with col_right:
    st.markdown(
        """
<div class="topbar">
  <div>
    <div class="kpi-title">System</div>
    <div class="kpi-value">Online</div>
  </div>
  <div>
    <div class="subtitle">Last refresh</div>
    <div class="kpi-title">{}</div>
  </div>
</div>
""".format(datetime.now().strftime("%H:%M:%S")),
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)


metrics = st.columns(4)
with metrics[0]:
    kpi_card("Incidents (filtered)", int(len(df)))
with metrics[1]:
    kpi_card("Critical", int(df["severity"].astype(str).str.contains("CRITICAL").sum()))
with metrics[2]:
    kpi_card("Warning", int(df["severity"].astype(str).str.contains("WARNING").sum()))
with metrics[3]:
    last_ts = df["timestamp"].max() if not df.empty else None
    kpi_card("Last Incident", last_ts.strftime("%H:%M:%S") if last_ts is not None else "None")


main_cols = st.columns([2.2, 1])
with main_cols[0]:
    st.markdown('<div class="section-title">Live Feed</div>', unsafe_allow_html=True)
    st.image(frame_rgb, channels="RGB", use_container_width=True)
    st.markdown(
        '<div class="subtitle">Green = Safe Zone, Red = High Risk Zone</div>',
        unsafe_allow_html=True,
    )
with main_cols[1]:
    st.markdown('<div class="section-title">Current Status</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="card">Alert: {badge_for_alert(alert)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title">Active Violations</div>', unsafe_allow_html=True)
    if alert != "INFO":
        rows = {}
        for violation in all_violations:
            sev, v, reason = violation[:3]
            key = (v, reason)
            rows[key] = rows.get(key, 0) + 1
        data = []
        for (v, reason), count in rows.items():
            data.append({"Violation": v, "Reason": reason, "Count": count})
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True, height=220)
    else:
        st.markdown('<div class="card">No active violations detected.</div>', unsafe_allow_html=True)


st.markdown('<div class="section-title">Incident Timeline</div>', unsafe_allow_html=True)
if not df.empty:
    by_minute = df.set_index("timestamp").resample("1min").size().rename("count").to_frame()
    st.line_chart(by_minute, height=160)
else:
    st.markdown('<div class="card">No incidents in selected range.</div>', unsafe_allow_html=True)


st.markdown('<div class="section-title">Recent Incidents</div>', unsafe_allow_html=True)
if not df.empty:
    recent = df.sort_values("timestamp", ascending=False).head(10)
    rows_html = []
    for _, row in recent.iterrows():
        rows_html.append(
            "<tr>"
            f"<td>{row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</td>"
            f"<td>{row.get('camera_id', '')}</td>"
            f"<td>{row.get('violations', '')}</td>"
            f"<td>{chip_for_severity(row.get('severity', 'INFO'))}</td>"
            "</tr>"
        )
    table_html = (
        '<div class="card"><table class="table">'
        "<thead><tr><th>Timestamp</th><th>Camera</th><th>Violation</th><th>Severity</th></tr></thead>"
        "<tbody>"
        + "".join(rows_html)
        + "</tbody></table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)
else:
    st.markdown('<div class="card">No incident history available.</div>', unsafe_allow_html=True)


time.sleep(0.03)
st.rerun()
