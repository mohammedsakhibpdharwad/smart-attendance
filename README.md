# AI-Powered Smart Attendance Management System

An AI-powered attendance system that uses **browser-based camera streaming** (WebRTC) and **facial recognition** to identify registered people and automatically record attendance. Built for **local development** and **Streamlit Community Cloud** deployment.

## Features

### Core
- Register people with face encodings (camera snapshot or file upload)
- Live face recognition via **streamlit-webrtc** (works on Streamlit Cloud — no server webcam needed)
- Automatic attendance marking with duplicate prevention (once per day)
- Attendance history with date/person filters
- CSV and PDF report export
- People management (view, edit, delete)

### Standout Differentiators
- **Liveness / anti-spoofing:** Blink detection using Eye Aspect Ratio (EAR) before marking attendance — a printed photo cannot easily pass
- **Confidence scores:** Match confidence displayed next to each recognized name in live view
- **Smart analytics:** Weekly/monthly trends, late-arrival detection, attendance streaks, anomaly flags
- **Batch registration:** Upload multiple labeled images or a ZIP file
- **Dark / light theme toggle**
- **Role-aware views:** Admin vs viewer mode (password from secrets)
- **Mobile-responsive layout**

## Tech Stack

| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| Live camera | streamlit-webrtc + av |
| Face recognition | face_recognition, OpenCV (headless) |
| Database | SQLite |
| Analytics | pandas |
| PDF reports | reportlab |

## Architecture

```
smart-attendance/
├── app/
│   ├── dashboard.py           # Main Streamlit app
│   ├── recognize_webrtc.py    # Browser-camera recognition (deployment-safe)
│   ├── recognize_local_cli.py # LOCAL TESTING ONLY — OpenCV webcam
│   ├── register_face.py       # Single + batch registration
│   ├── database.py            # SQLite layer
│   ├── face_utils.py          # Detection, encoding, matching
│   ├── liveness.py            # Blink-based liveness check
│   ├── attendance.py          # Attendance logic + cooldown
│   ├── analytics.py           # Trends, streaks, anomalies
│   ├── reports.py             # CSV + PDF export
│   ├── config.py              # Centralized configuration
│   └── utils.py               # Shared helpers
├── faces/                     # Registered face thumbnails
├── database/                  # SQLite DB (auto-created)
├── tests/
├── .streamlit/config.toml
├── requirements.txt
└── packages.txt               # System deps for Streamlit Cloud
```

## Local Installation

### Prerequisites
- Python 3.9+
- Webcam (for live attendance testing in browser)
- On Windows, `face_recognition` may require [CMake](https://cmake.org/) and Visual C++ build tools

### Setup

```bash
git clone <your-repo-url>
cd smart-attendance
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
pip install pytest   # for running tests
```

### Environment / Secrets

Copy `.env.example` to `.env` for local reference, or create `.streamlit/secrets.toml`:

```toml
ADMIN_PASSWORD = "your-secure-password"
```

Default admin password (if no secret set): `admin123`

### Run Locally

```bash
streamlit run app/dashboard.py
```

Open `http://localhost:8501` in your browser.

### Optional: Local CLI Testing (webcam via OpenCV)

**LOCAL TESTING ONLY — does not work on Streamlit Cloud:**

```bash
python -m app.recognize_local_cli
```

## How It Works

### Registration
1. Enter Person ID and name
2. Capture via browser camera (`st.camera_input`) or upload a photo
3. System detects exactly one face, generates a 128-d encoding, stores in SQLite

### Live Attendance (WebRTC)
1. Navigate to **Live Attendance**
2. Allow camera permission in browser
3. Your camera stream is sent to the app via WebRTC
4. Faces are detected, matched against stored encodings
5. **Liveness check:** You must blink naturally (EAR-based detection)
6. On match + liveness pass → attendance recorded (once per day, with cooldown)

### Liveness Check — Honest Limitations

This uses **basic blink detection** (Eye Aspect Ratio across consecutive frames). It helps prevent simple photo spoofing but is **not enterprise-grade biometric security**. It can be fooled by video replay on another screen, sophisticated masks, or deepfakes. For high-security environments, use dedicated liveness SDKs and hardware.

## Database

Auto-created at `database/attendance.db` on first run.

**Tables:**
- `people`: person_id, name, encoding (blob), photo_path, registered_at
- `attendance`: person_id, name, date, time, status, confidence_score (unique per person/date)

## Export Reports

1. Go to **Reports**
2. Select date range
3. Download **CSV** or **PDF**

## Admin vs Viewer Mode

- **Viewer:** Can view attendance, analytics, reports; cannot register/delete/edit people
- **Admin:** Full access — toggle via Settings with admin password

## Deploy to Streamlit Community Cloud

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Smart Attendance AI"
git remote add origin https://github.com/YOUR_USERNAME/smart-attendance.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Select your repository
5. Set **Main file path:** `app/dashboard.py`
6. Click **Advanced settings** → add secrets:

```toml
ADMIN_PASSWORD = "your-production-password"
```

7. Click **Deploy**

Streamlit Cloud will automatically:
- Install Python packages from `requirements.txt`
- Install system packages from `packages.txt` (cmake, libgl1, etc. for dlib/face_recognition)

### Step 3: Test After Deploy

1. Open your deployed app URL
2. Switch to Admin mode (Settings)
3. Register a person via camera or upload
4. Go to **Live Attendance** — allow camera permission
5. Verify recognition, liveness blink, and attendance marking

## Troubleshooting

| Issue | Solution |
|-------|----------|
| WebRTC won't connect | Try Chrome/Firefox; check corporate firewall; allow camera permission |
| Camera permission denied | Click the camera icon in browser address bar and allow access |
| face_recognition build fails locally | Install CMake and build tools; on Cloud, `packages.txt` handles this |
| No faces detected | Use good lighting, face the camera directly, one person per registration photo |
| Duplicate attendance | By design — one record per person per day |
| Admin password not working | Set `ADMIN_PASSWORD` in Streamlit Secrets or `.streamlit/secrets.toml` |

## Running Tests

```bash
pytest tests/ -v
```

## Known Limitations

- Liveness check is basic blink detection, not production biometric security
- face_recognition uses HOG model (CPU-friendly but less accurate than CNN)
- WebRTC may fail on restrictive networks
- Batch registration requires `ID_Name.jpg` filename convention
- Camera-dependent live recognition must be tested manually in browser

## Future Enhancements

- CNN face detection model option
- Multi-factor liveness (head turn + blink)
- Email/Slack notifications for attendance
- HR system integrations
- GPU acceleration for larger deployments

## License

MIT (or your chosen license)
