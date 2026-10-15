import requests
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import threading
from collections import deque, Counter

import sys

# ---------------- CONFIGURATION ----------------

# Camera Index:
# 0 = Logitech C270 USB Webcam (External)
# 1 = Integrated Laptop Camera (Built-in)
CAMERA_INDEX = 0

# Allow overriding via command-line argument: python smart_classroom_ai.py 0 or 1
if len(sys.argv) > 1 and sys.argv[1].isdigit():
    CAMERA_INDEX = int(sys.argv[1])

MAX_FACES = 10

# Baseline drowsiness timing
CLOSED_DURATION = 1.5           # Seconds of sustained eye-closure to trigger alert
TRACK_TIMEOUT = 1.2             # Seconds before dropping an unseen face track
CONFIRMATION_FRAMES = 3         # Frames required to confirm a new student track

# API Publishing
API_URL = "http://127.0.0.1:5000/update"
API_TIMEOUT = 1
PUBLISH_INTERVAL = 1.0          # Seconds between HTTP updates to prevent network lag

# -----------------------------------------------

print(f"[CAMERA] Attempting to open Camera Index {CAMERA_INDEX}...")
camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not camera.isOpened():
    print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
    print("Tip: If index 0 fails, try running: python smart_classroom_ai.py 1")
    raise SystemExit

mp_face_mesh = mp.solutions.face_mesh

# Facial Landmark Indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# 3D generic facial model points for Head Pose (SolvePnP)
HEAD_POSE_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (landmark 1)
    (0.0, -330.0, -65.0),        # Chin (landmark 152)
    (-225.0, 170.0, -135.0),     # Left eye outer corner (landmark 33)
    (225.0, 170.0, -135.0),      # Right eye outer corner (landmark 263)
    (-150.0, -150.0, -125.0),    # Left mouth corner (landmark 61)
    (150.0, -150.0, -125.0)      # Right mouth corner (landmark 291)
], dtype=np.float64)

tracks = {}
next_track_id = 1
recent_counts = deque(maxlen=10)
display_count = 0


def calculate_ear(landmarks, eye_indices, width, height):
    """Calculates the Eye Aspect Ratio (EAR) for a single eye."""
    points = np.array([
        [landmarks[i].x * width, landmarks[i].y * height]
        for i in eye_indices
    ], dtype=np.float64)

    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal < 1.0:
        return 0.0, points

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear, points


def estimate_head_pitch(landmarks, width, height):
    """Estimates the vertical pitch angle (in degrees) of the student's head."""
    image_points = np.array([
        [landmarks[1].x * width, landmarks[1].y * height],      # Nose tip
        [landmarks[152].x * width, landmarks[152].y * height],  # Chin
        [landmarks[33].x * width, landmarks[33].y * height],    # Left eye
        [landmarks[263].x * width, landmarks[263].y * height],  # Right eye
        [landmarks[61].x * width, landmarks[61].y * height],    # Left mouth
        [landmarks[291].x * width, landmarks[291].y * height]   # Right mouth
    ], dtype=np.float64)

    focal_length = width
    center = (width / 2.0, height / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        HEAD_POSE_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch = angles[0]
    return float(pitch)


def get_face_bbox_and_center(landmarks, width, height):
    """Calculates smoothed bounding box and centroid."""
    xs = [p.x * width for p in landmarks]
    ys = [p.y * height for p in landmarks]

    x1, x2 = max(0, min(xs)), min(width, max(xs))
    y1, y2 = max(0, min(ys)), min(height, max(ys))

    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    box = (int(x1), int(y1), int(x2), int(y2))
    return center, box


def calculate_iou(boxA, boxB):
    """Computes Intersection over Union between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union_area = float(boxA_area + boxB_area - inter_area)
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def match_faces_robust(detections, tracks, max_distance):
    """
    Robust multi-person tracker using both centroid distance and bounding-box IOU
    to eliminate track swapping and ID confusion.
    """
    if not detections or not tracks:
        return {}

    candidates = []

    for track_id, track in tracks.items():
        for index, det in enumerate(detections):
            dist = math.dist(track["center"], det["center"])
            iou = calculate_iou(track["box"], det["box"])

            # Cost combines normalized distance and IOU overlap
            cost = (dist / max_distance) * 0.6 + (1.0 - iou) * 0.4

            if dist < max_distance or iou > 0.25:
                candidates.append((cost, track_id, index))

    candidates.sort(key=lambda x: x[0])

    matched_tracks = set()
    matched_detections = set()
    matches = {}

    for cost, track_id, index in candidates:
        if track_id in matched_tracks or index in matched_detections:
            continue
        matches[index] = track_id
        matched_tracks.add(track_id)
        matched_detections.add(index)

    return matches


def _send_payload_worker(url, payload, timeout):
    """Background worker sending HTTP telemetry without blocking video loop."""
    try:
        requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException:
        pass


def publish_classroom_data(occupancy, drowsy_count):
    payload = {
        "occupancy": int(occupancy),
        "drowsy_count": int(drowsy_count)
    }
    thread = threading.Thread(
        target=_send_payload_worker,
        args=(API_URL, payload, API_TIMEOUT),
        daemon=True
    )
    thread.start()


# ---------------- MAIN VISION PIPELINE ----------------

with mp_face_mesh.FaceMesh(
    max_num_faces=MAX_FACES,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    print("==================================================")
    print(" Smart Classroom AI (Phase 7 - Advanced AI Engine)")
    print(" - Individual Adaptive EAR Baseline")
    print(" - Head Pitch / Nodding & Gaze Compensation")
    print(" - Blink Filtering & Debounced Micro-Sleep Timing")
    print(" - Robust IOU Multi-Face Tracking")
    print(" Press 'Q' in the camera window to stop.")
    print("==================================================")

    last_publish_time = 0.0

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame.")
            break

        now = time.time()
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        detections = []

        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                landmarks = face.landmark

                left_ear, left_pts = calculate_ear(landmarks, LEFT_EYE, width, height)
                right_ear, right_pts = calculate_ear(landmarks, RIGHT_EYE, width, height)
                ear = (left_ear + right_ear) / 2.0

                pitch = estimate_head_pitch(landmarks, width, height)
                center, box = get_face_bbox_and_center(landmarks, width, height)

                detections.append({
                    "center": center,
                    "box": box,
                    "ear": ear,
                    "pitch": pitch,
                    "eye_points": np.vstack((left_pts, right_pts)).astype(int)
                })

        # Match detections with existing tracks
        max_dist = 0.18 * math.hypot(width, height)
        matches = match_faces_robust(detections, tracks, max_dist)

        seen_track_ids = set()
        possible_drowsiness_count = 0
        confirmed_student_count = 0

        for index, det in enumerate(detections):
            if index in matches:
                track_id = matches[index]
            else:
                track_id = next_track_id
                next_track_id += 1

                tracks[track_id] = {
                    "center": det["center"],
                    "box": det["box"],
                    "last_seen": now,
                    "seen_frames": 0,
                    "closed_since": None,
                    "last_open_time": now,
                    "prolonged": False,
                    "baseline_ear": 0.28,      # Default open-eye baseline
                    "ear_history": deque(maxlen=25),
                    "status_label": "Calibrating"
                }

            track = tracks[track_id]
            track["last_seen"] = now
            track["seen_frames"] += 1
            seen_track_ids.add(track_id)

            # Smooth bounding box with Exponential Moving Average (EMA)
            alpha = 0.7
            old_box = track["box"]
            new_box = det["box"]
            smooth_box = (
                int(alpha * new_box[0] + (1 - alpha) * old_box[0]),
                int(alpha * new_box[1] + (1 - alpha) * old_box[1]),
                int(alpha * new_box[2] + (1 - alpha) * old_box[2]),
                int(alpha * new_box[3] + (1 - alpha) * old_box[3]),
            )
            track["box"] = smooth_box
            track["center"] = det["center"]

            ear = det["ear"]
            pitch = det["pitch"]

            # Maintain adaptive baseline EAR for this individual
            if ear > 0.20:
                track["ear_history"].append(ear)
                if len(track["ear_history"]) >= 5:
                    track["baseline_ear"] = float(np.median(track["ear_history"]))

            # Adaptive threshold tailored to individual eye geometry & distance
            base = track["baseline_ear"]
            effective_ear_threshold = min(0.22, max(0.16, base * 0.72))

            # Head Pose Compensations:
            # When looking down reading notes, eyelids naturally lower.
            # We compensate so studying students are not falsely flagged.
            is_looking_down = (pitch > 20.0)
            if is_looking_down:
                effective_ear_threshold *= 0.88  # Require firmer closure if head is tilted down reading

            # Determine instantaneous closure state
            is_eyes_closed = (ear < effective_ear_threshold)

            # Blink Debounce & Prolonged Timing:
            # Normal human blinks last 0.1s - 0.4s.
            if is_eyes_closed:
                if track["closed_since"] is None:
                    track["closed_since"] = now

                duration = now - track["closed_since"]

                # If eyes remain closed for >= 1.5 seconds, flag as prolonged closure
                if duration >= CLOSED_DURATION:
                    track["prolonged"] = True
                    track["status_label"] = f"Fatigue ({duration:.1f}s)"
                else:
                    track["prolonged"] = False
                    track["status_label"] = "Blinking" if duration < 0.4 else "Eyes Closing"
            else:
                # Brief tolerance (0.3s): Prevents single-frame tracking glitches from resetting micro-sleep
                if now - track["last_open_time"] > 0.30:
                    track["closed_since"] = None
                    track["prolonged"] = False

                track["last_open_time"] = now

                if is_looking_down:
                    track["status_label"] = "Reading/Looking Down"
                else:
                    track["status_label"] = "Attentive"

            # Count only confirmed tracks (detected for >= CONFIRMATION_FRAMES)
            if track["seen_frames"] >= CONFIRMATION_FRAMES:
                confirmed_student_count += 1
                if track["prolonged"]:
                    possible_drowsiness_count += 1

            # Determine visualization colors
            if track["prolonged"]:
                color = (0, 0, 255)       # Red: Alert
            elif is_eyes_closed:
                color = (0, 165, 255)     # Orange: Blink / Closing
            elif is_looking_down:
                color = (255, 200, 0)     # Cyan/Yellow: Looking down reading
            else:
                color = (0, 255, 0)       # Green: Attentive

            # Draw smooth bounding box
            x1, y1, x2, y2 = track["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw eye contour points for visual clarity
            for pt in det["eye_points"]:
                cv2.circle(frame, (pt[0], pt[1]), 1, color, -1)

            # Draw status label above face
            label_text = f"#{track_id}: {track['status_label']}"
            cv2.putText(
                frame, label_text, (x1, max(22, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2
            )

            # Draw telemetry below face (EAR, Base, Pitch)
            telemetry_text = f"EAR:{ear:.2f} (Thresh:{effective_ear_threshold:.2f}) Pitch:{pitch:+.0f}"
            cv2.putText(
                frame, telemetry_text, (x1, min(height - 10, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1
            )

        # Cleanup lost tracks
        for track_id, track in list(tracks.items()):
            if track_id not in seen_track_ids:
                if now - track["last_seen"] > TRACK_TIMEOUT:
                    del tracks[track_id]
                else:
                    track["closed_since"] = None
                    track["prolonged"] = False

        # Temporal Majority Smoothing for Occupancy
        recent_counts.append(confirmed_student_count)
        if len(recent_counts) == recent_counts.maxlen:
            common_count, freq = Counter(recent_counts).most_common(1)[0]
            if freq >= 6:
                display_count = common_count

        # On-screen HUD Statistics
        cv2.rectangle(frame, (10, 10), (320, 95), (20, 20, 30), -1)
        cv2.rectangle(frame, (10, 10), (320, 95), (80, 80, 100), 1)

        cv2.putText(
            frame, f"Occupancy : {display_count} Students",
            (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )

        drowsy_color = (0, 0, 255) if possible_drowsiness_count > 0 else (0, 255, 120)
        cv2.putText(
            frame, f"Drowsiness: {possible_drowsiness_count} Alert(s)",
            (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, drowsy_color, 2
        )

        cv2.putText(
            frame, f"Engine    : MediaPipe + Adaptive EAR",
            (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1
        )

        # Throttled Background API Publishing
        if now - last_publish_time >= PUBLISH_INTERVAL:
            publish_classroom_data(display_count, possible_drowsiness_count)
            last_publish_time = now

        cv2.imshow("SmartClassroom AI - Enhanced Vision Engine", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
print("Smart Classroom AI stopped cleanly.")
