"""Tests for Smart Attendance Management System."""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def temp_db(monkeypatch):
    """Use a temporary database for each test."""
    with tempfile.TemporaryDirectory() as tmp:
        db_dir = Path(tmp) / "database"
        db_dir.mkdir()
        db_path = db_dir / "attendance.db"
        monkeypatch.setattr("app.database.DATABASE_DIR", db_dir)
        monkeypatch.setattr("app.database.DATABASE_PATH", db_path)
        from app.database import init_database

        init_database()
        yield db_path


@pytest.fixture
def sample_encoding():
    return np.random.rand(128).astype(np.float64)


class TestDatabase:
    def test_init_creates_tables(self, temp_db):
        from app.database import get_connection

        with get_connection() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            names = {t["name"] for t in tables}
            assert "people" in names
            assert "attendance" in names

    def test_add_person(self, temp_db, sample_encoding):
        from app.database import add_person, get_person_by_id

        ok, msg = add_person("EMP001", "Alice", sample_encoding)
        assert ok is True
        person = get_person_by_id("EMP001")
        assert person is not None
        assert person["name"] == "Alice"

    def test_duplicate_person_id(self, temp_db, sample_encoding):
        from app.database import add_person

        add_person("EMP001", "Alice", sample_encoding)
        ok, msg = add_person("EMP001", "Bob", sample_encoding)
        assert ok is False
        assert "already exists" in msg.lower()

    def test_encoding_roundtrip(self, temp_db, sample_encoding):
        from app.database import blob_to_encoding, encoding_to_blob

        blob = encoding_to_blob(sample_encoding)
        restored = blob_to_encoding(blob)
        assert restored is not None
        np.testing.assert_array_almost_equal(restored, sample_encoding)

    def test_corrupted_encoding(self):
        from app.database import blob_to_encoding

        assert blob_to_encoding(b"bad") is None

    def test_record_attendance(self, temp_db, sample_encoding):
        from app.database import add_person, get_attendance_records, record_attendance

        add_person("EMP001", "Alice", sample_encoding)
        ok, msg = record_attendance("EMP001", "Alice", 95.5)
        assert ok is True
        records = get_attendance_records()
        assert len(records) == 1
        assert records[0]["confidence_score"] == 95.5

    def test_duplicate_attendance_prevention(self, temp_db, sample_encoding):
        from app.database import add_person, record_attendance

        add_person("EMP001", "Alice", sample_encoding)
        record_attendance("EMP001", "Alice", 90.0)
        ok, msg = record_attendance("EMP001", "Alice", 90.0)
        assert ok is False

    def test_date_filtering(self, temp_db, sample_encoding):
        from app.database import add_person, get_attendance_records, record_attendance

        add_person("EMP001", "Alice", sample_encoding)
        today = datetime.now().strftime("%Y-%m-%d")
        record_attendance("EMP001", "Alice", 90.0, att_date=today)
        records = get_attendance_records(start_date=today, end_date=today)
        assert len(records) == 1
        records = get_attendance_records(start_date="2000-01-01", end_date="2000-01-02")
        assert len(records) == 0

    def test_delete_person(self, temp_db, sample_encoding):
        from app.database import add_person, delete_person, get_person_by_id, record_attendance

        add_person("EMP001", "Alice", sample_encoding)
        record_attendance("EMP001", "Alice", 90.0)
        ok, _ = delete_person("EMP001")
        assert ok is True
        assert get_person_by_id("EMP001") is None


class TestAttendance:
    def test_late_status(self, temp_db, sample_encoding, monkeypatch):
        from app.attendance import AttendanceManager
        from app.database import add_person, get_attendance_records, save_setting

        add_person("EMP001", "Alice", sample_encoding)
        save_setting("late_arrival_cutoff", "08:00")
        mgr = AttendanceManager()
        ok, msg = mgr.mark_attendance("EMP001", "Alice", 95.0, force=True)
        assert ok is True
        records = get_attendance_records()
        assert records[0]["status"] in ("Present", "Late")

    def test_cooldown(self, temp_db, sample_encoding):
        from app.attendance import AttendanceManager
        from app.database import add_person, record_attendance

        add_person("EMP001", "Alice", sample_encoding)
        record_attendance("EMP001", "Alice", 90.0)
        mgr = AttendanceManager()
        ok, msg = mgr.mark_attendance("EMP001", "Alice", 90.0)
        assert ok is False


class TestAnalytics:
    def test_summary_stats(self, temp_db, sample_encoding):
        from app.analytics import summary_stats
        from app.database import add_person, record_attendance

        add_person("EMP001", "Alice", sample_encoding)
        record_attendance("EMP001", "Alice", 90.0)
        stats = summary_stats()
        assert stats["total_people"] == 1
        assert stats["present_today"] == 1

    def test_weekly_trend(self, temp_db, sample_encoding):
        from app.analytics import weekly_trend
        from app.database import add_person, record_attendance

        add_person("EMP001", "Alice", sample_encoding)
        today = datetime.now().strftime("%Y-%m-%d")
        record_attendance("EMP001", "Alice", 90.0, att_date=today)
        trend = weekly_trend(7)
        assert not trend.empty
        assert trend["count"].sum() >= 1

    def test_compute_streaks(self, temp_db, sample_encoding):
        from app.analytics import compute_streaks
        from app.database import add_person, record_attendance

        add_person("EMP001", "Alice", sample_encoding)
        today = datetime.now().strftime("%Y-%m-%d")
        record_attendance("EMP001", "Alice", 90.0, att_date=today)
        streaks = compute_streaks()
        assert len(streaks) == 1
        assert streaks[0]["current_streak"] >= 1

    def test_detect_anomalies_empty(self, temp_db):
        from app.analytics import detect_anomalies

        assert detect_anomalies() == []


class TestReports:
    def test_csv_generation(self, temp_db, sample_encoding):
        from app.database import add_person, record_attendance
        from app.reports import generate_csv

        add_person("EMP001", "Alice", sample_encoding)
        record_attendance("EMP001", "Alice", 92.5)
        from app.database import get_attendance_records

        records = get_attendance_records()
        csv_bytes = generate_csv(records)
        text = csv_bytes.decode("utf-8")
        assert "person_id" in text
        assert "EMP001" in text
        assert "Alice" in text

    def test_csv_empty(self):
        from app.reports import generate_csv

        csv_bytes = generate_csv([])
        assert b"person_id" in csv_bytes

    def test_pdf_generation(self, temp_db, sample_encoding):
        from app.database import add_person, get_attendance_records, record_attendance
        from app.reports import generate_pdf

        add_person("EMP001", "Alice", sample_encoding)
        record_attendance("EMP001", "Alice", 92.5)
        records = get_attendance_records()
        pdf = generate_pdf(records, "2024-01-01", "2024-12-31")
        assert pdf[:4] == b"%PDF"


class TestLiveness:
    def test_eye_aspect_ratio(self):
        from app.liveness import eye_aspect_ratio

        # Flat eye (open)
        eye = [(0, 0), (1, 0), (2, 0), (3, 0), (2, 1), (1, 1)]
        ear = eye_aspect_ratio(eye)
        assert ear > 0

    def test_blink_detection(self):
        from app.liveness import LivenessTracker

        tracker = LivenessTracker(person_id="test")
        landmarks = {
            "left_eye": [(0, 5), (2, 4), (4, 4), (6, 5), (4, 6), (2, 6)],
            "right_eye": [(10, 5), (12, 4), (14, 4), (16, 5), (14, 6), (12, 6)],
        }
        for _ in range(5):
            tracker.update(landmarks)
        assert tracker.frames_processed == 5


class TestUtils:
    def test_confidence_from_distance(self):
        from app.utils import confidence_from_distance

        assert confidence_from_distance(0.0, 0.6) == 100.0
        assert confidence_from_distance(0.6, 0.6) == 0.0

    def test_parse_batch_filename(self):
        from app.utils import parse_batch_filename

        pid, name = parse_batch_filename("EMP001_John_Doe.jpg")
        assert pid == "EMP001"
        assert "John" in name


class TestConfig:
    def test_relative_paths(self):
        from app.config import DATABASE_PATH, FACES_DIR, PROJECT_ROOT

        assert DATABASE_PATH.name == "attendance.db"
        assert DATABASE_PATH.parent.name == "database"
        assert FACES_DIR.name == "faces"
        assert PROJECT_ROOT.exists()

    def test_no_hardcoded_user_paths(self):
        """Scan app modules for hardcoded Windows user paths."""
        app_dir = PROJECT_ROOT / "app"
        for py_file in app_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            assert "C:\\Users" not in content
            assert "C:/Users" not in content
