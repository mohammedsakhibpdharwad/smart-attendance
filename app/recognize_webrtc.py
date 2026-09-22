"""Browser-based face recognition via streamlit-webrtc (deployment-safe)."""

from __future__ import annotations

import logging
import threading

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase

from app.attendance import AttendanceManager
from app.config import FRAME_SCALE, RECOGNITION_TOLERANCE
from app.database import get_all_encodings
from app.face_utils import (
    bgr_to_rgb,
    build_recognition_label,
    detect_faces,
    draw_face_box,
    get_face_encodings,
    get_face_landmarks,
    match_faces,
    resize_frame,
)
from app.liveness import LivenessManager
from app.utils import confidence_from_distance

logger = logging.getLogger(__name__)

# Module-level singletons for WebRTC thread
_lock = threading.Lock()
_attendance_mgr = AttendanceManager()
_liveness_mgr = LivenessManager()


@st.cache_data(ttl=300)
def load_known_faces() -> tuple[list[str], list[str], list[np.ndarray]]:
    """Cache known face encodings for the session."""
    return get_all_encodings()


class FaceRecognitionTransformer(VideoProcessorBase):
    """Video transformer for streamlit-webrtc live recognition."""

    def __init__(self) -> None:
        self.known_ids, self.known_names, self.known_encodings = load_known_faces()
        self.last_message: str = ""
        self.last_success: bool = False
        self.tolerance = RECOGNITION_TOLERANCE

    def reload_encodings(self) -> None:
        load_known_faces.clear()
        self.known_ids, self.known_names, self.known_encodings = load_known_faces()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if not self.known_encodings:
            cv2.putText(
                img, "No people registered", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        small = resize_frame(img, FRAME_SCALE)
        rgb_small = bgr_to_rgb(small)

        locations = detect_faces(rgb_small)
        encodings = get_face_encodings(rgb_small, locations) if locations else []
        landmarks_list = get_face_landmarks(rgb_small) if locations else []

        active_ids: set[str] = set()

        for i, (loc, encoding) in enumerate(zip(locations, encodings)):
            idx, distance = match_faces(encoding, self.known_encodings, self.tolerance)

            if idx >= 0:
                person_id = self.known_ids[idx]
                name = self.known_names[idx]
                active_ids.add(person_id)
                label = build_recognition_label(name, distance, self.tolerance)
                color = (0, 200, 0)

                # Liveness check via blink
                if i < len(landmarks_list):
                    tracker = _liveness_mgr.get_tracker(person_id)
                    is_live = tracker.update(landmarks_list[i])
                    if not is_live:
                        label += f" | {tracker.status_message[:30]}"
                        color = (0, 165, 255)  # Orange — waiting for blink
                    else:
                        conf = confidence_from_distance(distance, self.tolerance)
                        with _lock:
                            success, msg = _attendance_mgr.mark_attendance(
                                person_id, name, conf
                            )
                            self.last_message = msg
                            self.last_success = success
                        if success:
                            color = (0, 255, 0)
                            label += " ✓"
            else:
                label = build_recognition_label("Unknown", distance, self.tolerance)
                color = (0, 0, 255)

            draw_face_box(img, loc, label, color, scale=FRAME_SCALE)

        _liveness_mgr.cleanup_stale(active_ids)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
