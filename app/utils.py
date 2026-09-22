"""Shared utility helpers."""

from __future__ import annotations

import io
import os
from datetime import datetime, time
from pathlib import Path
from typing import Optional

import streamlit as st

from app.config import ALLOWED_IMAGE_EXTENSIONS, MAX_UPLOAD_SIZE_MB


def ensure_directories(*paths: Path) -> None:
    """Create directories if they do not exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def get_admin_password() -> str:
    """Return admin password from secrets or environment."""
    try:
        if hasattr(st, "secrets") and "ADMIN_PASSWORD" in st.secrets:
            return str(st.secrets["ADMIN_PASSWORD"])
    except (FileNotFoundError, KeyError, RuntimeError):
        pass
    return os.environ.get("ADMIN_PASSWORD", "admin123")


def is_admin_mode() -> bool:
    """Check if current session is in admin mode."""
    return st.session_state.get("is_admin", False)


def parse_time_str(time_str: str) -> time:
    """Parse HH:MM string to time object."""
    parts = time_str.strip().split(":")
    return time(int(parts[0]), int(parts[1]))


def format_datetime(dt: Optional[datetime] = None) -> str:
    """Format datetime for display."""
    dt = dt or datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_date(dt: Optional[datetime] = None) -> str:
    """Format date as YYYY-MM-DD."""
    dt = dt or datetime.now()
    return dt.strftime("%Y-%m-%d")


def format_time(dt: Optional[datetime] = None) -> str:
    """Format time as HH:MM:SS."""
    dt = dt or datetime.now()
    return dt.strftime("%H:%M:%S")


def validate_image_upload(
    uploaded_file, max_mb: float = MAX_UPLOAD_SIZE_MB
) -> tuple[bool, str]:
    """Validate uploaded image file type and size."""
    if uploaded_file is None:
        return False, "No file provided."

    name = getattr(uploaded_file, "name", "") or ""
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return False, f"Invalid file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"

    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_mb:
        return False, f"File too large ({size_mb:.1f} MB). Max: {max_mb} MB."

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    return True, ""


def load_image_bytes(data: bytes):
    """Load PIL Image from bytes."""
    from PIL import Image

    return Image.open(io.BytesIO(data)).convert("RGB")


def parse_batch_filename(filename: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse batch filename like 'EMP001_John_Doe.jpg' or 'John_Doe_EMP001.png'.
    Returns (person_id, name) or (None, None) if unparseable.
    """
    import re
    from pathlib import Path

    stem = Path(filename).stem
    parts = stem.replace("-", "_").split("_")
    if len(parts) < 2:
        return None, None

    if re.match(r"^[A-Za-z0-9]+$", parts[0]) and len(parts[0]) <= 20:
        person_id = parts[0]
        name = " ".join(parts[1:]).replace("_", " ").title()
        return person_id, name

    person_id = parts[-1]
    name = " ".join(parts[:-1]).replace("_", " ").title()
    return person_id, name


def confidence_from_distance(distance: float, tolerance: float) -> float:
    """Convert face distance to a 0-100 confidence percentage."""
    if tolerance <= 0:
        return 0.0
    conf = max(0.0, min(100.0, (1.0 - distance / tolerance) * 100.0))
    return round(conf, 1)


def inject_responsive_css(dark_mode: bool = False) -> None:
    """Inject responsive layout CSS and theme variables."""

    if dark_mode:
        bg = "#0e1117"
        text = "#fafafa"
        card = "#1a1d24"
        input_bg = "#262b35"
        border = "#3a3f4b"
    else:
        bg = "#ffffff"
        text = "#262730"
        card = "#f0f2f6"
        input_bg = "#ffffff"
        border = "#d6d9e0"

    accent = "#4CAF50"

    st.markdown(
        f"""
        <style>

        /* Main Streamlit application */
        html, body, .stApp {{
            background-color: {bg} !important;
            color: {text} !important;
        }}

        [data-testid="stAppViewContainer"] {{
            background-color: {bg} !important;
        }}

        [data-testid="stMain"] {{
            background-color: {bg} !important;
            color: {text} !important;
        }}

        [data-testid="stHeader"] {{
            background-color: {bg} !important;
        }}

        /* Main content */
        .main {{
            background-color: {bg} !important;
            color: {text} !important;
        }}

        .main .block-container {{
            padding-top: 1rem;
            max-width: 1200px;
        }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {card} !important;
        }}

        [data-testid="stSidebar"] > div:first-child {{
            background-color: {card} !important;
        }}

        /* Text */
        h1, h2, h3, h4, h5, h6,
        p, label, span, div {{
            color: {text};
        }}

        /* Cards */
        .metric-card {{
            background: {card} !important;
            border-radius: 10px;
            padding: 1rem 1.25rem;
            border-left: 4px solid {accent};
            margin-bottom: 0.5rem;
        }}

        .metric-card h3,
        .metric-card p {{
            color: {text} !important;
        }}

        /* Inputs */
        input,
        textarea {{
            background-color: {input_bg} !important;
            color: {text} !important;
            border-color: {border} !important;
        }}

        [data-baseweb="select"] > div {{
            background-color: {input_bg} !important;
            color: {text} !important;
        }}

        /* Buttons */
        .stButton > button {{
            background-color: {card} !important;
            color: {text} !important;
            border: 1px solid {border} !important;
        }}

        .stButton > button:hover {{
            background-color: {input_bg} !important;
            color: {text} !important;
            border-color: {accent} !important;
        }}

        .stButton > button p {{
            color: {text} !important;
        }}

        /* Information box */
        .attendance-info {{
            background: {card} !important;
            color: {text} !important;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }}

        /* Success box */
        .attendance-success {{
            background: #d4edda;
            color: #155724 !important;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }}

        /* Mobile */
        @media (max-width: 768px) {{
            .main .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
            }}

            .metric-card p {{
                font-size: 1.35rem;
            }}

            [data-testid="stSidebar"] {{
                min-width: 200px;
            }}
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )