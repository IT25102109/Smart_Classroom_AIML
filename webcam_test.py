import cv2
import sys

def test_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return None
    return cap

current_index = 0
if len(sys.argv) > 1 and sys.argv[1].isdigit():
    current_index = int(sys.argv[1])

print("==================================================")
print(" SmartClassroom AI — Camera Hardware Diagnostics")
print(" Connected Cameras Detected on System:")
print("   - Index 0 : External Logitech C270 HD WebCam")
print("   - Index 1 : Integrated Laptop Camera")
print(" Controls:")
print("   Press 'S' : Switch between Camera 0 and Camera 1")
print("   Press 'Q' : Exit diagnostics")
print("==================================================")

cap = test_camera(current_index)
if cap is None:
    print(f"Could not open Camera {current_index}, trying alternate index...")
    current_index = 1 if current_index == 0 else 0
    cap = test_camera(current_index)

if cap is None:
    print("Error: No working camera found.")
    raise SystemExit

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed.")
        break

    h, w = frame.shape[:2]
    cam_name = "Logitech USB Webcam (Index 0)" if current_index == 0 else "Integrated Laptop Camera (Index 1)"

    # Draw HUD
    cv2.rectangle(frame, (10, 10), (450, 75), (0, 0, 0), -1)
    cv2.putText(frame, f"Active: {cam_name}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(frame, f"Res: {w}x{h} | Press 'S' to switch, 'Q' to quit", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow("SmartClassroom AI - Camera Diagnostic Tool", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        # Switch camera index
        next_index = 1 if current_index == 0 else 0
        print(f"Switching to Camera Index {next_index}...")
        cap.release()
        new_cap = test_camera(next_index)
        if new_cap is not None:
            cap = new_cap
            current_index = next_index
        else:
            print(f"Failed to open Camera Index {next_index}, reverting back.")
            cap = test_camera(current_index)

cap.release()
cv2.destroyAllWindows()
