"""
SmartClassroom AI — Accuracy Evaluation & Benchmark Tool (Phase 7)

Evaluates the AI estimation pipeline:
- Measures Confusion Matrix: TP, TN, FP, FN
- Computes Precision, Recall, Specificity, Accuracy, FPR, FNR
- Tests reading gaze vs false drowsiness
- Generates evaluation_report.json for demonstration & documentation

Controls during benchmark session:
  [O] Key : Set Ground Truth to 'AWAKE / OPEN'
  [C] Key : Set Ground Truth to 'PROLONGED CLOSED / DROWSY'
  [R] Key : Set Ground Truth to 'READING / LOOKING DOWN'
  [Q] Key : Stop session and generate accuracy report
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import json
import math

import sys

CAMERA_INDEX = 0  # 0: Logitech USB Webcam, 1: Integrated Laptop Camera
if len(sys.argv) > 1 and sys.argv[1].isdigit():
    CAMERA_INDEX = int(sys.argv[1])

CLOSED_DURATION = 1.5

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not camera.isOpened():
    print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
    raise SystemExit

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

HEAD_POSE_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

def calculate_ear(landmarks, eye_indices, width, height):
    points = np.array([[landmarks[i].x * width, landmarks[i].y * height] for i in eye_indices], dtype=np.float64)
    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])
    if horizontal < 1.0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def estimate_pitch(landmarks, width, height):
    image_points = np.array([
        [landmarks[1].x * width, landmarks[1].y * height],
        [landmarks[152].x * width, landmarks[152].y * height],
        [landmarks[33].x * width, landmarks[33].y * height],
        [landmarks[263].x * width, landmarks[263].y * height],
        [landmarks[61].x * width, landmarks[61].y * height],
        [landmarks[291].x * width, landmarks[291].y * height]
    ], dtype=np.float64)

    camera_matrix = np.array([[width, 0, width / 2.0], [0, width, height / 2.0], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    success, rvec, _ = cv2.solvePnP(HEAD_POSE_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return float(angles[0])

# Metrics counters
tp, tn, fp, fn = 0, 0, 0, 0
total_frames = 0
ground_truth = "AWAKE"  # AWAKE, DROWSY, READING

eyes_closed_since = None
baseline_ear = 0.28
ear_history = []

print("=======================================================")
print(" SmartClassroom AI — Model Evaluation Benchmark Tool")
print(" Keys to label Ground Truth live:")
print("   [O] : Mark AWAKE / ATTENTIVE (Negative)")
print("   [C] : Mark PROLONGED EYE CLOSURE / DROWSY (Positive)")
print("   [R] : Mark READING / LOOKING DOWN (Negative)")
print("   [Q] : Conclude session and output accuracy statistics")
print("=======================================================")

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    start_time = time.time()

    while True:
        success, frame = camera.read()
        if not success:
            break

        now = time.time()
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        ai_prediction_drowsy = False
        ear = 0.0
        pitch = 0.0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            ear_l = calculate_ear(landmarks, LEFT_EYE, width, height)
            ear_r = calculate_ear(landmarks, RIGHT_EYE, width, height)
            ear = (ear_l + ear_r) / 2.0
            pitch = estimate_pitch(landmarks, width, height)

            # Adaptive baseline
            if ear > 0.20:
                ear_history.append(ear)
                if len(ear_history) > 30:
                    ear_history.pop(0)
                baseline_ear = float(np.median(ear_history))

            threshold = min(0.22, max(0.16, baseline_ear * 0.72))
            if pitch > 20.0:  # looking down compensation
                threshold *= 0.88

            if ear < threshold:
                if eyes_closed_since is None:
                    eyes_closed_since = now
                if now - eyes_closed_since >= CLOSED_DURATION:
                    ai_prediction_drowsy = True
            else:
                eyes_closed_since = None

        # Compare ground truth with AI prediction
        gt_is_drowsy = (ground_truth == "DROWSY")

        if gt_is_drowsy and ai_prediction_drowsy:
            tp += 1
        elif not gt_is_drowsy and not ai_prediction_drowsy:
            tn += 1
        elif not gt_is_drowsy and ai_prediction_drowsy:
            fp += 1
        elif gt_is_drowsy and not ai_prediction_drowsy:
            fn += 1

        total_frames += 1

        # Draw UI overlay
        gt_color = (0, 0, 255) if ground_truth == "DROWSY" else ((255, 200, 0) if ground_truth == "READING" else (0, 255, 0))
        cv2.putText(frame, f"Ground Truth: {ground_truth} [Press O / C / R]", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gt_color, 2)
        
        pred_text = "DROWSY" if ai_prediction_drowsy else "AWAKE / ATTENTIVE"
        pred_color = (0, 0, 255) if ai_prediction_drowsy else (0, 255, 0)
        cv2.putText(frame, f"AI Predicted: {pred_text}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)

        cv2.putText(frame, f"EAR: {ear:.3f} | Base: {baseline_ear:.3f} | Pitch: {pitch:+.1f} deg", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn} (Frames: {total_frames})", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("SmartClassroom AI - Model Evaluation", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):
            ground_truth = "AWAKE"
        elif key == ord('c'):
            ground_truth = "DROWSY"
        elif key == ord('r'):
            ground_truth = "READING"

camera.release()
cv2.destroyAllWindows()

# Calculate statistics
duration_sec = round(time.time() - start_time, 1)
accuracy = (tp + tn) / total_frames if total_frames > 0 else 0.0
precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

report = {
    "session_duration_seconds": duration_sec,
    "total_evaluated_frames": total_frames,
    "confusion_matrix": {
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn
    },
    "metrics": {
        "accuracy_pct": round(accuracy * 100, 2),
        "precision_pct": round(precision * 100, 2),
        "recall_pct": round(recall * 100, 2),
        "f1_score": round(f1_score, 3),
        "false_positive_rate_pct": round(fpr * 100, 2),
        "false_negative_rate_pct": round(fnr * 100, 2)
    }
}

with open("evaluation_metrics.json", "w") as f:
    json.dump(report, f, indent=4)

print("\n" + "=" * 50)
print("           EVALUATION BENCHMARK RESULTS           ")
print("=" * 50)
print(f"Total Frames Evaluated : {total_frames} ({duration_sec}s)")
print(f"True Positives  (TP)   : {tp}")
print(f"True Negatives  (TN)   : {tn}")
print(f"False Positives (FP)   : {fp} (False alarms)")
print(f"False Negatives (FN)   : {fn} (Missed closures)")
print("-" * 50)
print(f"Model Accuracy         : {report['metrics']['accuracy_pct']}%")
print(f"Precision              : {report['metrics']['precision_pct']}%")
print(f"Recall (Sensitivity)   : {report['metrics']['recall_pct']}%")
print(f"F1 Score               : {report['metrics']['f1_score']}")
print(f"False Positive Rate    : {report['metrics']['false_positive_rate_pct']}%")
print(f"False Negative Rate    : {report['metrics']['false_negative_rate_pct']}%")
print("=" * 50)
print("Saved metrics to 'evaluation_metrics.json'")

