"""Analytics: trends, streaks, late arrivals, anomaly detection."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from app.config import ANOMALY_DROP_THRESHOLD, LATE_ARRIVAL_CUTOFF, STREAK_LOOKBACK_DAYS
from app.database import count_people, get_all_people, get_attendance_records, get_settings


def _get_cutoff() -> str:
    return get_settings().get("late_arrival_cutoff", LATE_ARRIVAL_CUTOFF)


def attendance_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert attendance records to DataFrame."""
    if not records:
        return pd.DataFrame(
            columns=["person_id", "name", "date", "time", "status", "confidence_score"]
        )
    return pd.DataFrame(records)


def weekly_trend(days: int = 7) -> pd.DataFrame:
    """Daily attendance counts for the last N days."""
    end = datetime.now().date()
    start = end - timedelta(days=days - 1)
    records = get_attendance_records(
        start_date=start.isoformat(),
        end_date=end.isoformat(),
    )
    df = attendance_to_dataframe(records)
    date_range = pd.date_range(start=start, end=end, freq="D")
    counts = df.groupby("date").size() if not df.empty else pd.Series(dtype=int)
    trend = pd.DataFrame({"date": date_range.strftime("%Y-%m-%d")})
    trend["count"] = trend["date"].map(counts).fillna(0).astype(int)
    return trend


def monthly_trend(months: int = 6) -> pd.DataFrame:
    """Monthly attendance totals."""
    end = datetime.now().date()
    start = end - timedelta(days=months * 31)
    records = get_attendance_records(start_date=start.isoformat(), end_date=end.isoformat())
    df = attendance_to_dataframe(records)
    if df.empty:
        return pd.DataFrame(columns=["month", "count"])
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    return df.groupby("month").size().reset_index(name="count")


def compute_streaks() -> list[dict[str, Any]]:
    """Compute consecutive-day attendance streak per person."""
    people = get_all_people()
    lookback_start = (datetime.now() - timedelta(days=STREAK_LOOKBACK_DAYS)).date().isoformat()
    records = get_attendance_records(start_date=lookback_start)
    df = attendance_to_dataframe(records)

    streaks = []
    today = datetime.now().date()

    for person in people:
        pid = person["person_id"]
        person_dates = set()
        if not df.empty:
            person_dates = set(df[df["person_id"] == pid]["date"].tolist())

        streak = 0
        check = today
        while check.isoformat() in person_dates:
            streak += 1
            check -= timedelta(days=1)

        streaks.append(
            {
                "person_id": pid,
                "name": person["name"],
                "current_streak": streak,
                "total_days": len(person_dates),
            }
        )

    streaks.sort(key=lambda x: x["current_streak"], reverse=True)
    return streaks


def late_arrivals(start_date: str | None = None, end_date: str | None = None) -> list[dict]:
    """Return records marked as Late or arriving after cutoff."""
    records = get_attendance_records(start_date=start_date, end_date=end_date)
    cutoff = _get_cutoff()
    late = []
    for r in records:
        if r.get("status") == "Late":
            late.append(r)
        elif r.get("time", "")[:5] > cutoff:
            late.append({**r, "status": "Late"})
    return late


def detect_anomalies(window_days: int = 14) -> list[dict[str, Any]]:
    """
    Flag people whose recent attendance rate dropped sharply vs prior period.
    Basic heuristic — not ML-based.
    """
    people = get_all_people()
    if not people:
        return []

    today = datetime.now().date()
    recent_start = (today - timedelta(days=window_days)).isoformat()
    prior_start = (today - timedelta(days=window_days * 2)).isoformat()
    prior_end = (today - timedelta(days=window_days + 1)).isoformat()

    recent = get_attendance_records(start_date=recent_start, end_date=today.isoformat())
    prior = get_attendance_records(start_date=prior_start, end_date=prior_end)

    recent_df = attendance_to_dataframe(recent)
    prior_df = attendance_to_dataframe(prior)

    anomalies = []
    for person in people:
        pid = person["person_id"]
        recent_count = len(recent_df[recent_df["person_id"] == pid]) if not recent_df.empty else 0
        prior_count = len(prior_df[prior_df["person_id"] == pid]) if not prior_df.empty else 0

        recent_rate = recent_count / window_days
        prior_rate = prior_count / window_days if prior_count > 0 else 0

        if prior_rate > 0 and recent_rate < prior_rate * (1 - ANOMALY_DROP_THRESHOLD):
            drop_pct = round((1 - recent_rate / prior_rate) * 100, 1)
            anomalies.append(
                {
                    "person_id": pid,
                    "name": person["name"],
                    "recent_rate": round(recent_rate, 2),
                    "prior_rate": round(prior_rate, 2),
                    "drop_percent": drop_pct,
                    "message": f"Attendance dropped {drop_pct}% vs prior {window_days} days",
                }
            )
        elif prior_rate >= 0.5 and recent_count == 0:
            anomalies.append(
                {
                    "person_id": pid,
                    "name": person["name"],
                    "recent_rate": 0,
                    "prior_rate": round(prior_rate, 2),
                    "drop_percent": 100.0,
                    "message": f"No attendance in last {window_days} days (was regular)",
                }
            )

    return anomalies


def summary_stats() -> dict[str, Any]:
    """Dashboard summary statistics."""
    total = count_people()
    today_records = get_attendance_records(
        start_date=datetime.now().date().isoformat(),
        end_date=datetime.now().date().isoformat(),
    )
    present = len(today_records)
    absent = max(0, total - present)
    pct = round((present / total * 100), 1) if total > 0 else 0.0
    return {
        "total_people": total,
        "present_today": present,
        "absent_today": absent,
        "attendance_percentage": pct,
    }
