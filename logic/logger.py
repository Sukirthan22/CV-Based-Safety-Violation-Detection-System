import csv
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = BASE_DIR / "logs" / "violations.csv"

def log_violation(camera_id, violations, severity):
    """
    Logs safety violations to CSV file
    """

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_FILE.exists()

    with LOG_FILE.open(mode="a", newline="") as f:
        writer = csv.writer(f)

        # Write header once
        if not file_exists:
            writer.writerow([
                "timestamp",
                "camera_id",
                "violations",
                "severity"
            ])
        

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            camera_id,
            ", ".join([v[1] for v in violations]),
            severity
        ])
