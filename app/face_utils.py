"""Face detection, encoding, and recognition utilities."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import face_recognition
import numpy as np
from PIL import Image

from app.config import FRAME_SCALE, RECOGNITION_TOLERANCE
from app.utils import confidence_from_distance

logger = logging.getLogger(__name__)


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Convert PIL image to RGB numpy array."""
    return np.array(image.convert("RGB"))


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert BGR OpenCV frame to RGB."""
    if frame is None or frame.size == 0:
        return frame
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def resize_frame(frame: np.ndarray, scale: float = FRAME_SCALE) -> np.ndarray:
    """Downscale frame for faster processing."""
    if scale >= 1.0:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def detect_faces(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect face locations in RGB image. Returns (top, right, bottom, left) list."""
    try:
        return face_recognition.face_locations(image, model="hog")
    except Exception as exc:
        logger.error("Face detection failed: %s", exc)
        return []


def get_face_encodings(
    image: np.ndarray, locations: Optional[list] = None
) -> list[np.ndarray]:
    """Generate encodings for faces in image."""
    try:
        if locations is None:
            return face_recognition.face_encodings(image)
        return face_recognition.face_encodings(image, known_face_locations=locations)
    except Exception as exc:
        logger.error("Encoding generation failed: %s", exc)
        return []


def get_face_landmarks(image: np.ndarray) -> list[dict]:
    """Get facial landmarks for liveness detection."""
    try:
        return face_recognition.face_landmarks(image)
    except Exception as exc:
        logger.error("Landmark detection failed: %s", exc)
        return []


def extract_encoding_from_image(image: Image.Image) -> tuple[Optional[np.ndarray], str]:
    """
    Extract a single face encoding from a PIL image.
    Returns (encoding, error_message). encoding is None on failure.
    """
    rgb = pil_to_rgb_array(image)
    locations = detect_faces(rgb)

    if len(locations) == 0:
        return None, "No face detected. Please use a clear front-facing photo."
    if len(locations) > 1:
        return None, "Multiple faces detected. Please use a photo with exactly one face."

    encodings = get_face_encodings(rgb, locations)
    if not encodings:
        return None, "Could not generate face encoding. Try a different photo."

    return encodings[0], ""


def match_faces(
    face_encoding: np.ndarray,
    known_encodings: list[np.ndarray],
    tolerance: float = RECOGNITION_TOLERANCE,
) -> tuple[int, float]:
    """
    Find best match among known encodings.
    Returns (index, distance). index is -1 if no match.
    """
    if not known_encodings:
        return -1, 1.0

    try:
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        if best_dist <= tolerance:
            return best_idx, best_dist
        return -1, best_dist
    except Exception as exc:
        logger.error("Face matching failed: %s", exc)
        return -1, 1.0


def draw_face_box(
    frame: np.ndarray,
    location: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int] = (0, 255, 0),
    scale: float = 1.0,
) -> None:
    """Draw bounding box and label on BGR frame."""
    top, right, bottom, left = location
    if scale != 1.0:
        inv = 1.0 / scale
        top, right, bottom, left = int(top * inv), int(right * inv), int(bottom * inv), int(left * inv)

    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.rectangle(frame, (left, bottom - 28), (right, bottom), color, cv2.FILLED)
    cv2.putText(
        frame,
        label,
        (left + 6, bottom - 8),
        cv2.FONT_HERSHEY_DUPLEX,
        0.5,
        (255, 255, 255),
        1,
    )


def build_recognition_label(
    name: str, distance: float, tolerance: float = RECOGNITION_TOLERANCE
) -> str:
    """Build label with name and confidence score."""
    if name == "Unknown":
        return f"Unknown ({confidence_from_distance(distance, tolerance):.0f}%)"
    conf = confidence_from_distance(distance, tolerance)
    return f"{name} ({conf:.0f}%)"
