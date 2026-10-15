
import cv2
import mediapipe as mp
import numpy as np
import time

# Open the Logitech USB webcam (Index 0)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

mp_face_mesh = mp.solutions.face_mesh

# Landmark indexes for the left and right eyes
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(landmarks, eye_indices, width, height):
    points = []

    for index in eye_indices:
        landmark = landmarks[index]
        points.append([
            int(landmark.x * width),
            int(landmark.y * height)
        ])

    points = np.array(points, dtype=np.float64)

    # Vertical distances
    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])

    # Horizontal distance
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal == 0:
        return 0.0, points.astype(int)

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear, points.astype(int)


# Initial experimental threshold; it may need calibration
EAR_THRESHOLD = 0.21
CLOSED_DURATION = 1.5

eyes_closed_since = None
status = "Monitoring"

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    print("Drowsiness test started. Press Q to exit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame.")
            break

        height, width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb_frame)

        status = "Monitoring"
        ear_text = "EAR: --"

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_ear, left_points = calculate_ear(
                landmarks, LEFT_EYE, width, height
            )

            right_ear, right_points = calculate_ear(
                landmarks, RIGHT_EYE, width, height
            )

            average_ear = (left_ear + right_ear) / 2.0
            ear_text = f"EAR: {average_ear:.3f}"

            # Draw eye landmarks
            for point in np.vstack((left_points, right_points)):
                cv2.circle(
                    frame, tuple(point), 2, (0, 255, 0), -1
                )

            if average_ear < EAR_THRESHOLD:
                if eyes_closed_since is None:
                    eyes_closed_since = time.time()

                closed_time = time.time() - eyes_closed_since

                if closed_time >= CLOSED_DURATION:
                    status = "Possible prolonged eye closure"
                else:
                    status = "Eyes possibly closed"
            else:
                eyes_closed_since = None
                status = "Eyes open"

        else:
            # Reset when no face is detected
            eyes_closed_since = None
            status = "No face detected"

        cv2.putText(
            frame, ear_text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 255, 255), 2
        )

        cv2.putText(
            frame, status, (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 255) if "Possible prolonged" in status
            else (0, 255, 0), 2
        )

        cv2.imshow("Smart Classroom - Eye Closure Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
