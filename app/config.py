"""Central configuration for the Smart Attendance Management System."""

from pathlib import Path

# Paths (relative to project root — works on Windows and Linux)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_PATH = DATABASE_DIR / "attendance.db"
FACES_DIR = PROJECT_ROOT / "faces"

# Face recognition
RECOGNITION_TOLERANCE = 0.6
FRAME_SCALE = 0.25  # Downscale frames before processing for performance

# Attendance
LATE_ARRIVAL_CUTOFF = "09:30"  # HH:MM — configurable via settings
ATTENDANCE_COOLDOWN_SECONDS = 30  # Debounce between attendance marks per person
DEFAULT_STATUS = "Present"

# Liveness (blink detection)
BLINK_EAR_THRESHOLD = 0.21
BLINK_CONSECUTIVE_FRAMES = 2
LIVENESS_FRAMES_REQUIRED = 3

# Upload validation
MAX_UPLOAD_SIZE_MB = 5
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_BATCH_FILES = 50

# Analytics
ANOMALY_DROP_THRESHOLD = 0.5  # Flag if attendance rate drops >50% vs prior period
STREAK_LOOKBACK_DAYS = 90

# UI
APP_TITLE = "Smart Attendance AI"
APP_ICON = "📋"
