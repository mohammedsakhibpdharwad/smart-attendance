"""Smart Attendance AI — main Streamlit dashboard."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.analytics import (  # noqa: E402
    compute_streaks,
    detect_anomalies,
    late_arrivals,
    monthly_trend,
    summary_stats,
    weekly_trend,
)
from app.config import APP_ICON, APP_TITLE, FACES_DIR, LATE_ARRIVAL_CUTOFF, PROJECT_ROOT  # noqa: E402
from app.database import (  # noqa: E402
    delete_person,
    get_all_people,
    get_attendance_records,
    get_today_attendance,
    init_database,
    save_setting,
    update_person,
)
from app.recognize_webrtc import FaceRecognitionTransformer  # noqa: E402
from app.register_face import register_batch_files, register_batch_zip, register_single  # noqa: E402
from app.reports import generate_csv, generate_pdf  # noqa: E402
from app.utils import (  # noqa: E402
    get_admin_password,
    inject_responsive_css,
    is_admin_mode,
    validate_image_upload,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_session_state() -> None:
    defaults = {
        "is_admin": False,
        "dark_mode": False,
        "page": "Dashboard",
        "attendance_msg": "",
        "attendance_ok": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><h3>{label}</h3><p>{value}</p></div>',
        unsafe_allow_html=True,
    )


def page_dashboard() -> None:
    st.header("Dashboard")
    stats = summary_stats()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Registered People", str(stats["total_people"]))
    with c2:
        render_metric_card("Present Today", str(stats["present_today"]))
    with c3:
        render_metric_card("Absent Today", str(stats["absent_today"]))
    with c4:
        render_metric_card("Attendance %", f"{stats['attendance_percentage']}%")

    st.subheader("Today's Attendance")
    today = get_today_attendance()
    if today:
        st.dataframe(pd.DataFrame(today), use_container_width=True, hide_index=True)
    else:
        st.info("No attendance recorded yet today.")

    st.subheader("Weekly Trend")
    trend = weekly_trend(7)
    if not trend.empty:
        st.line_chart(trend.set_index("date")["count"])
    else:
        st.info("No attendance data for trend chart.")


def page_live_attendance() -> None:
    st.header("Live Attendance")
    st.caption("Uses your browser camera via WebRTC — works locally and on Streamlit Cloud.")

    people = get_all_people()
    if not people:
        st.warning("No people registered yet. Go to **Register Person** first.")
        return

    st.info(
        "**Liveness check:** Please blink naturally when recognized. "
        "A printed photo cannot pass this check (basic anti-spoofing — not enterprise-grade)."
    )

    try:
        from streamlit_webrtc import WebRtcMode, webrtc_streamer

        ctx = webrtc_streamer(
            key="attendance-webrtc",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=FaceRecognitionTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.state.playing:
            processor = ctx.video_processor
            if processor and processor.last_message:
                if processor.last_success:
                    st.success(processor.last_message)
                else:
                    st.warning(processor.last_message)
        elif ctx.state.connection_state == "failed":
            st.error(
                "WebRTC connection failed. Check camera permissions, try a different browser, "
                "or ensure your network allows WebRTC (some corporate networks block it)."
            )
    except ImportError:
        st.error("streamlit-webrtc is not installed. Run: pip install streamlit-webrtc av")
    except Exception as exc:
        logger.exception("WebRTC error")
        st.error(f"Camera error: {exc}. Ensure you granted camera permission in your browser.")


def page_register() -> None:
    if not is_admin_mode():
        st.warning("Admin mode required to register people.")
        return

    st.header("Register Person")
    tab1, tab2 = st.tabs(["Single Registration", "Batch Registration"])

    with tab1:
        st.subheader("Register via Camera or Upload")
        person_id = st.text_input("Person ID", key="reg_id", placeholder="e.g. EMP001")
        name = st.text_input("Full Name", key="reg_name", placeholder="e.g. John Doe")

        col1, col2 = st.columns(2)
        with col1:
            camera_photo = st.camera_input("Take a photo")
        with col2:
            uploaded = st.file_uploader("Or upload image", type=["jpg", "jpeg", "png", "webp"])

        if st.button("Register", type="primary", key="reg_btn"):
            if not person_id or not name:
                st.error("Person ID and name are required.")
            elif camera_photo is None and uploaded is None:
                st.error("Please capture or upload a face photo.")
            else:
                if camera_photo:
                    data = camera_photo.getvalue()
                else:
                    valid, err = validate_image_upload(uploaded)
                    if not valid:
                        st.error(err)
                        return
                    data = uploaded.getvalue()

                success, msg = register_single(person_id.strip(), name.strip(), data)
                if success:
                    st.success(msg)
                    from app.recognize_webrtc import load_known_faces
                    load_known_faces.clear()
                else:
                    st.error(msg)

    with tab2:
        st.subheader("Batch Register")
        st.caption("Name files as `ID_Name.jpg` (e.g. `EMP001_John_Doe.jpg`)")
        batch_files = st.file_uploader(
            "Upload multiple images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="batch_files",
        )
        zip_file = st.file_uploader("Or upload a ZIP of images", type=["zip"], key="batch_zip")

        if st.button("Process Batch", type="primary", key="batch_btn"):
            results = []
            if batch_files:
                results.extend(register_batch_files(batch_files))
            if zip_file:
                results.extend(register_batch_zip(zip_file.getvalue()))
            if not results:
                st.error("Upload files or a ZIP to process.")
            else:
                for r in results:
                    if r["success"]:
                        st.success(f"{r['filename']}: {r['message']}")
                    else:
                        st.error(f"{r['filename']}: {r['message']}")
                from app.recognize_webrtc import load_known_faces
                load_known_faces.clear()


def page_people() -> None:
    st.header("People Management")
    people = get_all_people()

    if not people:
        st.info("No registered people yet.")
        return

    for person in people:
        with st.expander(f"{person['name']} ({person['person_id']})"):
            col1, col2 = st.columns([1, 2])
            with col1:
                photo = person.get("photo_path")
                if photo:
                    full = PROJECT_ROOT / photo
                    if full.exists():
                        st.image(str(full), width=120)
            with col2:
                st.write(f"**Registered:** {person['registered_at'][:10]}")
                if is_admin_mode():
                    new_name = st.text_input(
                        "Edit name", value=person["name"], key=f"edit_{person['person_id']}"
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Save", key=f"save_{person['person_id']}"):
                            ok, msg = update_person(person["person_id"], new_name)
                            st.success(msg) if ok else st.error(msg)
                            if ok:
                                st.rerun()
                    with c2:
                        if st.button("Delete", key=f"del_{person['person_id']}", type="secondary"):
                            st.session_state[f"confirm_del_{person['person_id']}"] = True
                    if st.session_state.get(f"confirm_del_{person['person_id']}"):
                        st.warning("Are you sure? This deletes all attendance records too.")
                        if st.button("Confirm Delete", key=f"confirm_{person['person_id']}"):
                            ok, msg = delete_person(person["person_id"])
                            st.success(msg) if ok else st.error(msg)
                            from app.recognize_webrtc import load_known_faces
                            load_known_faces.clear()
                            st.rerun()
                else:
                    st.caption("Viewer mode — editing disabled.")


def page_history() -> None:
    st.header("Attendance History")
    people = get_all_people()
    person_options = ["All"] + [p["person_id"] for p in people]

    col1, col2, col3 = st.columns(3)
    with col1:
        start = st.date_input("From", value=datetime.now().date() - timedelta(days=30))
    with col2:
        end = st.date_input("To", value=datetime.now().date())
    with col3:
        person_filter = st.selectbox("Person", person_options)

    search = st.text_input("Search by name or ID")

    records = get_attendance_records(
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        person_id=person_filter,
        search=search or None,
    )

    if records:
        df = pd.DataFrame(records)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No attendance records found for the selected filters.")


def page_analytics() -> None:
    st.header("Analytics")

    tab1, tab2, tab3, tab4 = st.tabs(["Trends", "Streaks", "Late Arrivals", "Anomalies"])

    with tab1:
        st.subheader("Weekly Attendance Trend")
        wt = weekly_trend(14)
        if not wt.empty:
            st.line_chart(wt.set_index("date")["count"])
        st.subheader("Monthly Trend")
        mt = monthly_trend(6)
        if not mt.empty:
            st.bar_chart(mt.set_index("month")["count"])
        else:
            st.info("Not enough data for monthly trend.")

    with tab2:
        streaks = compute_streaks()
        if streaks:
            st.dataframe(pd.DataFrame(streaks), use_container_width=True, hide_index=True)
        else:
            st.info("No streak data available.")

    with tab3:
        cutoff = st.session_state.get("late_cutoff", LATE_ARRIVAL_CUTOFF)
        st.caption(f"Late cutoff: {cutoff}")
        late = late_arrivals()
        if late:
            st.dataframe(pd.DataFrame(late), use_container_width=True, hide_index=True)
        else:
            st.info("No late arrivals detected.")

    with tab4:
        anomalies = detect_anomalies()
        if anomalies:
            for a in anomalies:
                st.warning(f"**{a['name']}** ({a['person_id']}): {a['message']}")
        else:
            st.success("No attendance anomalies detected.")


def page_reports() -> None:
    st.header("Reports & Export")
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("From", value=datetime.now().date() - timedelta(days=30), key="rep_start")
    with col2:
        end = st.date_input("To", value=datetime.now().date(), key="rep_end")

    records = get_attendance_records(start_date=start.isoformat(), end_date=end.isoformat())
    st.write(f"**{len(records)}** records in selected range")

    col1, col2 = st.columns(2)
    with col1:
        try:
            csv_data = generate_csv(records)
            st.download_button(
                "Download CSV",
                csv_data,
                file_name=f"attendance_{start}_{end}.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"CSV export failed: {exc}")

    with col2:
        try:
            pdf_data = generate_pdf(records, start.isoformat(), end.isoformat())
            st.download_button(
                "Download PDF",
                pdf_data,
                file_name=f"attendance_{start}_{end}.pdf",
                mime="application/pdf",
            )
        except Exception as exc:
            st.error(f"PDF export failed: {exc}")


def page_settings() -> None:
    st.header("Settings")

    st.subheader("Theme")
    dark = st.toggle("Dark Mode", value=st.session_state.dark_mode)
    if dark != st.session_state.dark_mode:
        st.session_state.dark_mode = dark
        st.rerun()

    st.subheader("Late Arrival Cutoff")
    cutoff = st.text_input("Cutoff time (HH:MM)", value=LATE_ARRIVAL_CUTOFF)
    if st.button("Save Cutoff"):
        save_setting("late_arrival_cutoff", cutoff)
        st.session_state.late_cutoff = cutoff
        st.success(f"Late cutoff set to {cutoff}")

    st.subheader("Role Mode")
    if is_admin_mode():
        st.success("You are in **Admin** mode.")
        if st.button("Switch to Viewer"):
            st.session_state.is_admin = False
            st.rerun()
    else:
        st.info("You are in **Viewer** mode (read-only for management actions).")
        pwd = st.text_input("Admin Password", type="password")
        if st.button("Switch to Admin"):
            if pwd == get_admin_password():
                st.session_state.is_admin = True
                st.success("Admin mode enabled.")
                st.rerun()
            else:
                st.error("Incorrect password.")


PAGES = {
    "Dashboard": page_dashboard,
    "Live Attendance": page_live_attendance,
    "Register Person": page_register,
    "People Management": page_people,
    "Attendance History": page_history,
    "Analytics": page_analytics,
    "Reports": page_reports,
    "Settings": page_settings,
}


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide", initial_sidebar_state="expanded")

    try:
        init_database()
    except Exception as exc:
        st.error(f"Database initialization failed: {exc}")
        return

    init_session_state()
    inject_responsive_css(st.session_state.dark_mode)

    with st.sidebar:
        st.title(f"{APP_ICON} Smart Attendance")
        role = "Admin" if is_admin_mode() else "Viewer"
        st.caption(f"Mode: **{role}**")

        page = st.radio(
            "Navigation",
            list(PAGES.keys()),
            key="nav_page",
        )

        st.session_state.page = page

        st.divider()
        st.caption("AI-Powered Face Recognition Attendance")

    try:
        PAGES[page]()
    except Exception as exc:
        logger.exception("Page error")
        st.error(f"Something went wrong: {exc}")


if __name__ == "__main__":
    main()
