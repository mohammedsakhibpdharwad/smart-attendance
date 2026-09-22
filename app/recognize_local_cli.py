"""
LOCAL TESTING ONLY — Face recognition via OpenCV VideoCapture.

This script uses cv2.VideoCapture(0) and will NOT work on Streamlit Community Cloud
(servers have no physical webcam). Use the Streamlit dashboard with streamlit-webrtc
for deployment-safe browser-camera recognition.

Usage:
    python -m app.recognize_local_cli
"""

from __future__ import annotations

import logging
import sys
import time

import cv2

from app.config import FRAME_SCALE, RECOGNITION_TOLERANCE
from app.database import get_all_encodings, init_database
from app.face_utils import (
    bgr_to_rgb,
    build_recognition_label,
    detect_faces,
    draw_face_box,
    get_face_encodings,
    match_faces,
    resize_frame,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    print("=" * 60)
    print("LOCAL TESTING ONLY — OpenCV webcam recognition")
    print("This will NOT work on Streamlit Cloud.")
    print("Press 'q' to quit.")
    print("=" * 60)

    init_database()
    ids, names, encodings = get_all_encodings()

    if not encodings:
        print("No registered people. Register via the dashboard first.")
        return 1

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam (index 0).")
        return 1

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            small = resize_frame(frame, FRAME_SCALE)
            rgb = bgr_to_rgb(small)
            locations = detect_faces(rgb)
            face_encs = get_face_encodings(rgb, locations) if locations else []

            for loc, enc in zip(locations, face_encs):
                idx, dist = match_faces(enc, encodings, RECOGNITION_TOLERANCE)
                if idx >= 0:
                    label = build_recognition_label(names[idx], dist)
                    color = (0, 255, 0)
                else:
                    label = build_recognition_label("Unknown", dist)
                    color = (0, 0, 255)
                draw_face_box(frame, loc, label, color, scale=FRAME_SCALE)

            cv2.imshow("Local Face Recognition (TESTING ONLY)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(0.01)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
