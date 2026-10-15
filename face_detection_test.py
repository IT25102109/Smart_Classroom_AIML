
import cv2
import mediapipe as mp
from collections import deque, Counter

# Open the Logitech USB webcam (Index 0)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Keep the most recent 10 face counts
recent_counts = deque(maxlen=10)
display_count = 0

with mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
) as face_detection:

    print("Smoothed occupancy counter started. Press Q to exit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        current_count = 0

        if results.detections:
            current_count = len(results.detections)

            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

        recent_counts.append(current_count)

        # Update only after collecting 10 frames
        if len(recent_counts) == recent_counts.maxlen:
            most_common_count, frequency = Counter(
                recent_counts
            ).most_common(1)[0]

            # Require the same count in at least 7 of 10 frames
            if frequency >= 7:
                display_count = most_common_count

        cv2.putText(
            frame,
            f"Occupancy: {display_count}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Smart Classroom - Stable Occupancy", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
