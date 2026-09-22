"""Liveness detection via Eye Aspect Ratio (EAR) blink detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.config import BLINK_CONSECUTIVE_FRAMES, BLINK_EAR_THRESHOLD, LIVENESS_FRAMES_REQUIRED

logger = logging.getLogger(__name__)

# dlib 68-point landmark indices for eyes
LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]


def eye_aspect_ratio(eye_points: list[tuple[int, int]]) -> float:
    """
    Compute Eye Aspect Ratio (EAR) for blink detection.
    EAR drops below threshold during a blink.
    """
    if len(eye_points) < 6:
        return 0.3

    pts = np.array(eye_points)
    # Vertical distances
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    # Horizontal distance
    h = np.linalg.norm(pts[0] - pts[3])
    if h == 0:
        return 0.3
    return (v1 + v2) / (2.0 * h)


@dataclass
class LivenessTracker:
    """
    Tracks blink events across frames for a recognized person.
    Requires at least one blink detected before confirming liveness.
    """

    person_id: str
    blink_counter: int = 0
    frame_counter: int = 0
    blink_detected: bool = False
    frames_processed: int = 0
    ear_history: list[float] = field(default_factory=list)

    def update(self, landmarks: dict) -> bool:
        """
        Process landmarks for one frame.
        Returns True if liveness is confirmed (blink detected).
        """
        self.frames_processed += 1

        left_eye = landmarks.get("left_eye", [])
        right_eye = landmarks.get("right_eye", [])

        if not left_eye or not right_eye:
            return self.blink_detected

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        self.ear_history.append(ear)

        if ear < BLINK_EAR_THRESHOLD:
            self.frame_counter += 1
        else:
            if self.frame_counter >= BLINK_CONSECUTIVE_FRAMES:
                self.blink_counter += 1
                self.blink_detected = True
            self.frame_counter = 0

        return self.blink_detected and self.frames_processed >= LIVENESS_FRAMES_REQUIRED

    @property
    def status_message(self) -> str:
        """Human-readable liveness status."""
        if self.blink_detected:
            return "Liveness verified (blink detected)"
        if self.frames_processed < LIVENESS_FRAMES_REQUIRED:
            return f"Please blink naturally... ({self.frames_processed}/{LIVENESS_FRAMES_REQUIRED} frames)"
        return "Blink to verify you are live (not a photo)"


class LivenessManager:
    """Manages per-person liveness trackers during live recognition."""

    def __init__(self) -> None:
        self._trackers: dict[str, LivenessTracker] = {}

    def get_tracker(self, person_id: str) -> LivenessTracker:
        if person_id not in self._trackers:
            self._trackers[person_id] = LivenessTracker(person_id=person_id)
        return self._trackers[person_id]

    def is_live(self, person_id: str) -> bool:
        tracker = self._trackers.get(person_id)
        return tracker.blink_detected if tracker else False

    def reset(self, person_id: Optional[str] = None) -> None:
        if person_id:
            self._trackers.pop(person_id, None)
        else:
            self._trackers.clear()

    def cleanup_stale(self, active_ids: set[str]) -> None:
        """Remove trackers for people no longer in frame."""
        stale = [pid for pid in self._trackers if pid not in active_ids]
        for pid in stale:
            del self._trackers[pid]

