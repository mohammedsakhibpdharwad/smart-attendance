"""CSV and PDF report generation."""

from __future__ import annotations

import io
import logging
from datetime import datetime
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def generate_csv(records: list[dict[str, Any]]) -> bytes:
    """Generate CSV bytes from attendance records."""
    columns = ["person_id", "name", "date", "time", "status", "confidence_score"]
    if not records:
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(records)
        for col in columns:
            if col not in df.columns:
                df[col] = ""
        df = df[columns]
    return df.to_csv(index=False).encode("utf-8")


def generate_pdf(
    records: list[dict[str, Any]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> bytes:
    """Generate a one-page PDF attendance summary."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError("reportlab is required for PDF export") from exc

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    elements = []

    title = "Smart Attendance Report"
    elements.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    elements.append(Spacer(1, 12))

    date_range = f"{start_date or 'All'} to {end_date or 'All'}"
    elements.append(Paragraph(f"Date Range: {date_range}", styles["Normal"]))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    total = len(records)
    present = sum(1 for r in records if r.get("status") == "Present")
    late = sum(1 for r in records if r.get("status") == "Late")
    elements.append(Paragraph(f"<b>Summary:</b> {total} records | Present: {present} | Late: {late}", styles["Normal"]))
    elements.append(Spacer(1, 16))

    headers = ["ID", "Name", "Date", "Time", "Status", "Confidence"]
    data = [headers]
    for r in records[:100]:  # Limit rows for one-page layout
        conf = r.get("confidence_score", "")
        data.append([
            str(r.get("person_id", "")),
            str(r.get("name", ""))[:20],
            str(r.get("date", "")),
            str(r.get("time", "")),
            str(r.get("status", "")),
            f"{conf:.1f}%" if isinstance(conf, (int, float)) else str(conf),
        ])

    if len(records) > 100:
        data.append(["...", f"({len(records) - 100} more rows)", "", "", "", ""])

    table = Table(data, colWidths=[0.7 * inch, 1.3 * inch, 0.9 * inch, 0.8 * inch, 0.7 * inch, 0.8 * inch])
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4CAF50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ])
    )
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer.read()
