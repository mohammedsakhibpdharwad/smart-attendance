"""Attendance recording logic with cooldown and late detection."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

from app.config import ATTENDANCE_COOLDOWN_SECONDS, DEFAULT_STATUS, LATE_ARRIVAL_CUTOFF
from app.database import get_settings, record_attendance
from app.utils import parse_time_str

logger = logging.getLogger(__name__)


class AttendanceManager:
    """Handles attendance marking with debouncing and late-arrival detection."""

    def __init__(self) -> None:
        self._last_marked: dict[str, float] = {}

    def _get_cutoff_time(self) -> str:
        settings = get_settings()
        return settings.get("late_arrival_cutoff", LATE_ARRIVAL_CUTOFF)

    def _determine_status(self, att_time: Optional[str] = None) -> str:
        """Return Present or Late based on cutoff time."""
        att_time = att_time or datetime.now().strftime("%H:%M:%S")
        cutoff_str = self._get_cutoff_time()
        try:
            cutoff = parse_time_str(cutoff_str)
            current = parse_time_str(att_time[:5])
            if current > cutoff:
                return "Late"
        except (ValueError, IndexError):
            pass
        return DEFAULT_STATUS

    def can_mark(self, person_id: str) -> bool:
        """Check cooldown — prevent rapid duplicate marks."""
        last = self._last_marked.get(person_id, 0)
        return (time.time() - last) >= ATTENDANCE_COOLDOWN_SECONDS

    def mark_attendance(
        self,
        person_id: str,
        name: str,
        confidence_score: float,
        force: bool = False,
    ) -> tuple[bool, str]:
        """
        Attempt to mark attendance for a person.
        Returns (success, message).
        """
        if not force and not self.can_mark(person_id):
            return False, f"Cooldown active for {name}. Please wait."

        status = self._determine_status()
        success, message = record_attendance(
            person_id=person_id,
            name=name,
            confidence_score=confidence_score,
            status=status,
        )

        if success:
            self._last_marked[person_id] = time.time()
            if status == "Late":
                message += f" (Marked as Late — after {self._get_cutoff_time()})"

        return success, message

    def reset_cooldown(self, person_id: Optional[str] = None) -> None:
        if person_id:
            self._last_marked.pop(person_id, None)
        else:
            self._last_marked.clear()
