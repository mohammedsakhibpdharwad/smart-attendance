"""Face registration logic for single and batch uploads."""

from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from app.config import FACES_DIR, MAX_BATCH_FILES
from app.database import add_person
from app.face_utils import extract_encoding_from_image
from app.utils import ensure_directories, load_image_bytes, parse_batch_filename, validate_image_upload

logger = logging.getLogger(__name__)


def save_face_photo(person_id: str, image: Image.Image) -> str:
    """Save face thumbnail and return relative path."""
    ensure_directories(FACES_DIR)
    safe_id = re.sub(r"[^\w\-]", "_", person_id)
    path = FACES_DIR / f"{safe_id}.jpg"
    image.save(path, "JPEG", quality=85)
    return str(path.relative_to(FACES_DIR.parent))


def register_single(
    person_id: str,
    name: str,
    image_data: bytes,
) -> tuple[bool, str]:
    """Register one person from image bytes."""
    try:
        image = load_image_bytes(image_data)
    except Exception:
        return False, "Invalid image file. Please upload a valid JPG or PNG."

    encoding, error = extract_encoding_from_image(image)
    if encoding is None:
        return False, error

    photo_path = save_face_photo(person_id, image)
    return add_person(person_id, name, encoding, photo_path)


def register_batch_files(files: list[Any]) -> list[dict[str, Any]]:
    """
    Register multiple people from uploaded files.
    Each file should be named ID_Name.ext
    Returns list of {filename, success, message}.
    """
    results = []
    if len(files) > MAX_BATCH_FILES:
        return [{"filename": "—", "success": False, "message": f"Max {MAX_BATCH_FILES} files allowed."}]

    for uploaded in files:
        filename = getattr(uploaded, "name", "unknown")
        valid, err = validate_image_upload(uploaded)
        if not valid:
            results.append({"filename": filename, "success": False, "message": err})
            continue

        person_id, name = parse_batch_filename(filename)
        if not person_id or not name:
            results.append({
                "filename": filename,
                "success": False,
                "message": "Name file as ID_Name.jpg (e.g. EMP001_John_Doe.jpg)",
            })
            continue

        data = uploaded.getvalue()
        success, message = register_single(person_id, name, data)
        results.append({"filename": filename, "success": success, "message": message})

    return results


def register_batch_zip(zip_bytes: bytes) -> list[dict[str, Any]]:
    """Register people from a ZIP of images."""
    results = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            members = [
                n for n in zf.namelist()
                if not n.endswith("/") and Path(n).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ]
            if len(members) > MAX_BATCH_FILES:
                return [{"filename": "—", "success": False, "message": f"Max {MAX_BATCH_FILES} files in ZIP."}]

            for name in members:
                person_id, display_name = parse_batch_filename(name)
                if not person_id or not display_name:
                    results.append({
                        "filename": name,
                        "success": False,
                        "message": "Name file as ID_Name.jpg inside ZIP",
                    })
                    continue
                data = zf.read(name)
                success, message = register_single(person_id, display_name, data)
                results.append({"filename": name, "success": success, "message": message})
    except zipfile.BadZipFile:
        results.append({"filename": "—", "success": False, "message": "Invalid ZIP file."})

    return results
