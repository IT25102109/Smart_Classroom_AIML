# SmartClassroom AI — Intelligent Attendance & Attention Monitoring

A full-stack Computer Vision and IoT system designed to automate classroom occupancy tracking and detect prolonged student eye closure (fatigue/drowsiness) in real time. The platform features an AI computer vision engine, a persistent Flask API with SQLite storage, a modern command-center web dashboard, and an ESP32-S3 microcontroller hardware display node.

---

## 🌟 Key Features

* **Multi-Face Computer Vision Engine**:
  * Powered by **MediaPipe Face Mesh (468 landmarks)**.
  * **Adaptive Eye Aspect Ratio (EAR)**: Calibrates an individualized baseline per student to eliminate false alerts from different facial structures and seating distances.
  * **Head Pose & Gaze Pitch Compensation**: Distinguishes between reading/looking down at notes vs. actual drowsiness using 3D facial geometry (`solvePnP`).
  * **Blink Debouncing**: Filters out natural eye blinks ($100\text{ms}$–$400\text{ms}$) and only flags sustained micro-sleep closures ($\ge 1.5\text{s}$).
  * **Robust Multi-Person Tracking**: Combines centroid Euclidean distance with Bounding Box IOU overlap and exponential moving average (EMA) smoothing to prevent ID swapping and box jitter.
* **Modern Web Dashboard**:
  * Real-time telemetry (Active Occupancy, Drowsiness Count, Engagement Score %, Hardware Status).
  * Interactive **Chart.js** historical trend graphs (Attendance Trajectory & Fatigue Events).
  * Web Audio API alarm chime and interactive Alert Banner with Acknowledge/Dismiss controls.
  * Configurable alert thresholds ($\ge 1, 2, 3, 5$ students).
* **Permanent Data Storage & Export**:
  * SQLite persistence (`classroom.db`) restoring the latest state across server restarts.
  * One-click timestamped CSV export (`/export`) for attendance and attention analytics.
* **ESP32-S3 IoT Hardware Integration**:
  * Displays live student occupancy, drowsiness alerts, and connection status on a 1.3" I2C SH1106 OLED.
  * Inverted visual fatigue alert header.
  * Dynamic IP configuration via ESP32 Non-Volatile Storage (Preferences) & Serial console.
  * Automatic non-blocking Wi-Fi reconnection handling.

---

## 📐 System Architecture

```text
[ Logitech USB Webcam ] 
         │
         ▼
[ smart_classroom_ai.py ] ──(Async HTTP POST /update)──▶ [ Flask Server :5000 ] ──▶ [ SQLite DB ]
  - MediaPipe Face Mesh                                            ▲
  - Adaptive EAR & Pitch                                           │
  - Multi-Person IOU Tracking                    ┌─────────────────┴─────────────────┐
                                                 │                                   │
                                      (GET /classroom, /history)           (GET /classroom)
                                                 │                                   │
                                                 ▼                                   ▼
                                        [ Web Dashboard ]                  [ ESP32-S3 + OLED ]
                                        (index.html)                       (classroom_Arduino.ino)
```

---

## 📁 Repository Structure

```text
Smart_Classroom_AIML/
│
├── smart_classroom_ai.py      # Core Computer Vision pipeline (MediaPipe + Tracking + API publisher)
├── server.py                  # Flask REST API, SQLite persistence, and CSV export
├── evaluate_ai.py             # Model accuracy benchmark and confusion matrix evaluation script
├── webcam_test.py             # Hardware camera test & interactive index switcher (0 vs 1)
├── face_detection_test.py     # Standalone face detection & majority-vote smoothing test
├── drowsiness_test.py         # Standalone single-person EAR calculation test
│
├── templates/
│   └── index.html             # Command Center Dashboard with Chart.js & Web Audio alerts
│
├── classroom_Arduino/
│   └── classroom_Arduino.ino  # ESP32-S3 firmware with SH1106 OLED & NVS configuration
│
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Standard exclusions (.venv, databases, logs)
└── README.md                  # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
* **Python**: 3.10, 3.11, or 3.12
* **Hardware**:
  * USB Webcam (e.g., Logitech C270) or laptop built-in webcam.
  * (Optional) ESP32-S3 microcontroller with an SH1106 I2C 128x64 OLED display.

### 2. Environment Setup
Clone the repository and install dependencies in a virtual environment:

```bash
git clone https://github.com/IT25102109/Smart_Classroom_AIML.git
cd Smart_Classroom_AIML

# Create and activate virtual environment
python -m venv .venv

# Windows:
.\.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Launching the System

#### Step 1: Start the Backend Server
```bash
python server.py
```
* The server will run on `http://0.0.0.0:5000`.
* Access the web dashboard at: **[http://localhost:5000/dashboard](http://localhost:5000/dashboard)**.

#### Step 2: Start the AI Vision Engine
In a separate terminal (with the virtual environment activated):
```bash
# Uses default Camera Index 0 (Logitech USB Webcam):
python smart_classroom_ai.py

# Or specify an alternate camera index (e.g., Index 1 for built-in laptop camera):
python smart_classroom_ai.py 1
```

---

## 📟 ESP32-S3 Hardware Client Setup

1. Open `classroom_Arduino/classroom_Arduino.ino` in the **Arduino IDE**.
2. Install required libraries via Library Manager:
   * `ArduinoJson` (v7.x)
   * `Adafruit GFX Library`
   * `Adafruit SH110X`
3. Configure your Wi-Fi SSID and Password in the sketch.
4. Upload to your **ESP32S3 Dev Module** (Pins: `SDA=8`, `SCL=9`).
5. Open Serial Monitor (115200 baud). You can update the laptop's IP address dynamically without reflashing:
   ```text
   SET_IP 192.168.1.50
   ```

---

## 📊 Benchmarking & Model Evaluation

To evaluate model precision, recall, and false positive rates under varying lighting and seating scenarios, run:
```bash
python evaluate_ai.py
```
* **Controls**:
  * Press <kbd>O</kbd>: Mark Awake / Attentive
  * Press <kbd>C</kbd>: Mark Closed Eyes / Drowsy
  * Press <kbd>R</kbd>: Mark Reading / Looking Down
  * Press <kbd>Q</kbd>: Conclude and generate `evaluation_metrics.json`

---

## 📄 API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check & system status |
| `GET` | `/dashboard` | Interactive Web Dashboard UI |
| `GET` | `/classroom` | Latest classroom occupancy, drowsiness, and liveness metadata |
| `POST` | `/update` | Ingests live telemetry from AI engine (`{"occupancy": int, "drowsy_count": int}`) |
| `GET` | `/history` | Historical measurements from SQLite (`?limit=50&start=...&end=...`) |
| `GET` | `/export` | Download complete timestamped CSV report (`?start=...&end=...`) |

---

## ⚖️ Disclaimer
*This project is an experimental academic prototype for classroom attendance and focus analytics. Eye-closure estimations are algorithmic proxies and do not constitute medical or diagnostic evaluations of fatigue.*
