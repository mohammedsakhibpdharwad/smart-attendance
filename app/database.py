"""SQLite database layer — centralized SQL operations."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator, Optional

import numpy as np

from app.config import DATABASE_DIR, DATABASE_PATH
from app.utils import ensure_directories, format_date

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Raised when a database operation fails."""


def init_database() -> None:
    """Create database file and tables if they do not exist."""
    ensure_directories(DATABASE_DIR)
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                encoding BLOB NOT NULL,
                photo_path TEXT,
                registered_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Present',
                confidence_score REAL,
                UNIQUE(person_id, date)
            );

            CREATE INDEX IF NOT EXISTS idx_attendance_person_id ON attendance(person_id);
            CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);
            CREATE INDEX IF NOT EXISTS idx_people_person_id ON people(person_id);
            """
        )


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Yield a database connection with row factory."""
    ensure_directories(DATABASE_DIR)
    conn = None
    try:
        conn = sqlite3.connect(str(DATABASE_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        if conn:
            conn.rollback()
        logger.error("Database error: %s", exc)
        raise DatabaseError(str(exc)) from exc
    finally:
        if conn:
            conn.close()


def encoding_to_blob(encoding: np.ndarray) -> bytes:
    """Serialize numpy encoding to bytes."""
    return encoding.astype(np.float64).tobytes()


def blob_to_encoding(blob: bytes, length: int = 128) -> Optional[np.ndarray]:
    """Deserialize bytes to numpy encoding."""
    try:
        arr = np.frombuffer(blob, dtype=np.float64)
        if len(arr) != length:
            return None
        return arr
    except (ValueError, TypeError):
        return None


def add_person(
    person_id: str,
    name: str,
    encoding: np.ndarray,
    photo_path: Optional[str] = None,
) -> tuple[bool, str]:
    """Insert a new person. Returns (success, message)."""
    person_id = person_id.strip()
    name = name.strip()
    if not person_id or not name:
        return False, "Person ID and name are required."

    if get_person_by_id(person_id):
        return False, f"Person ID '{person_id}' already exists."

    try:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO people (person_id, name, encoding, photo_path, registered_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    person_id,
                    name,
                    encoding_to_blob(encoding),
                    photo_path,
                    datetime.now().isoformat(),
                ),
            )
        return True, f"Successfully registered {name} (ID: {person_id})."
    except DatabaseError as exc:
        return False, f"Failed to register person: {exc}"


def get_person_by_id(person_id: str) -> Optional[dict[str, Any]]:
    """Fetch a person by their string ID."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM people WHERE person_id = ?", (person_id.strip(),)
        ).fetchone()
        return dict(row) if row else None


def get_all_people() -> list[dict[str, Any]]:
    """Return all registered people."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM people ORDER BY name COLLATE NOCASE"
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_encodings() -> tuple[list[str], list[str], list[np.ndarray]]:
    """Load all encodings for recognition. Returns (ids, names, encodings)."""
    ids: list[str] = []
    names: list[str] = []
    encodings: list[np.ndarray] = []

    for person in get_all_people():
        enc = blob_to_encoding(person["encoding"])
        if enc is not None:
            ids.append(person["person_id"])
            names.append(person["name"])
            encodings.append(enc)
        else:
            logger.warning("Corrupted encoding for person %s", person["person_id"])

    return ids, names, encodings


def update_person(person_id: str, name: str) -> tuple[bool, str]:
    """Update a person's name."""
    name = name.strip()
    if not name:
        return False, "Name cannot be empty."
    if not get_person_by_id(person_id):
        return False, f"Person ID '{person_id}' not found."

    try:
        with get_connection() as conn:
            conn.execute(
                "UPDATE people SET name = ? WHERE person_id = ?",
                (name, person_id),
            )
        return True, f"Updated name to '{name}'."
    except DatabaseError as exc:
        return False, str(exc)


def delete_person(person_id: str) -> tuple[bool, str]:
    """Delete a person and their attendance records."""
    if not get_person_by_id(person_id):
        return False, f"Person ID '{person_id}' not found."

    try:
        with get_connection() as conn:
            conn.execute("DELETE FROM attendance WHERE person_id = ?", (person_id,))
            conn.execute("DELETE FROM people WHERE person_id = ?", (person_id,))
        return True, f"Deleted person '{person_id}' and their attendance records."
    except DatabaseError as exc:
        return False, str(exc)


def record_attendance(
    person_id: str,
    name: str,
    confidence_score: float,
    status: str = "Present",
    att_date: Optional[str] = None,
    att_time: Optional[str] = None,
) -> tuple[bool, str]:
    """Record attendance if not already marked today."""
    att_date = att_date or format_date()
    att_time = att_time or datetime.now().strftime("%H:%M:%S")

    if has_attendance_today(person_id, att_date):
        return False, f"Attendance already recorded for {name} on {att_date}."

    try:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO attendance (person_id, name, date, time, status, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (person_id, name, att_date, att_time, status, confidence_score),
            )
        return True, f"Attendance marked for {name} at {att_time}."
    except sqlite3.IntegrityError:
        return False, f"Attendance already recorded for {name} on {att_date}."
    except DatabaseError as exc:
        return False, f"Failed to record attendance: {exc}"


def has_attendance_today(person_id: str, att_date: Optional[str] = None) -> bool:
    """Check if attendance exists for person on given date."""
    att_date = att_date or format_date()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM attendance WHERE person_id = ? AND date = ?",
            (person_id, att_date),
        ).fetchone()
        return row is not None


def get_attendance_records(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    person_id: Optional[str] = None,
    search: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Fetch attendance records with optional filters."""
    query = "SELECT * FROM attendance WHERE 1=1"
    params: list[Any] = []

    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    if person_id and person_id != "All":
        query += " AND person_id = ?"
        params.append(person_id)
    if search:
        query += " AND (name LIKE ? OR person_id LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%"])

    query += " ORDER BY date DESC, time DESC"

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_today_attendance() -> list[dict[str, Any]]:
    """Return today's attendance records."""
    return get_attendance_records(start_date=format_date(), end_date=format_date())


def count_people() -> int:
    """Count registered people."""
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM people").fetchone()
        return int(row["cnt"]) if row else 0


def count_present_today() -> int:
    """Count people marked present today."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM attendance WHERE date = ?",
            (format_date(),),
        ).fetchone()
        return int(row["cnt"]) if row else 0


def get_settings() -> dict[str, str]:
    """Load app settings from settings table (create if needed)."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
        return {r["key"]: r["value"] for r in rows}


def save_setting(key: str, value: str) -> None:
    """Upsert a setting value."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO settings (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
