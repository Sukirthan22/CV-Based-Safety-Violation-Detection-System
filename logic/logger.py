import csv
import os
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = BASE_DIR / "logs" / "violations.csv"

# person_id, opening_timestamp, closing_timestamp, violation_detail, camera_source, severity
HEADERS = ["person_id", "opening_timestamp", "closing_timestamp", "violation_detail", "camera_source", "severity"]

def _ensure_csv():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        with LOG_FILE.open(mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)

def log_violation_start(camera_id, person_id, violation, severity, start_time=None):
    _ensure_csv()
    if start_time is None:
        start_time = datetime.now()
    
    start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    
    with LOG_FILE.open(mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            person_id,
            start_str,
            "Open",
            violation,
            camera_id,
            severity
        ])

def log_violation_end(camera_id, person_id, violation, end_time=None):
    _ensure_csv()
    if end_time is None:
        end_time = datetime.now()
    end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
    
    rows = []
    updated = False
    with LOG_FILE.open(mode="r", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        if headers:
            rows.append(headers)
        for row in reader:
            if len(row) == 6:
                r_pid, r_open, r_close, r_viol, r_cam, r_sev = row
                if r_pid == str(person_id) and r_viol == violation and r_close == "Open" and r_cam == camera_id:
                    row[2] = end_str
                    updated = True
            rows.append(row)
            
    if updated:
        # Write to a temp file and replace to prevent corruption
        temp_file = LOG_FILE.with_suffix('.csv.tmp')
        with temp_file.open(mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        temp_file.replace(LOG_FILE)

def close_all_open_violations():
    if not LOG_FILE.exists():
        return
        
    end_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    updated = False
    
    with LOG_FILE.open(mode="r", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        if headers:
            rows.append(headers)
        for row in reader:
            if len(row) == 6 and row[2] == "Open":
                row[2] = end_str
                updated = True
            rows.append(row)
            
    if updated:
        temp_file = LOG_FILE.with_suffix('.csv.tmp')
        with temp_file.open(mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        temp_file.replace(LOG_FILE)
