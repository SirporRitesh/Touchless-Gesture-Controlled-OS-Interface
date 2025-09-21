import cv2
import time
from hand_tracking import HandTracker
from gestures import is_fist, is_palm

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    window_name = 'Webcam'
    cv2.namedWindow(window_name)
    last_log_time = 0
    log_message = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        frame, results = tracker.find_hands(frame)

        # Default: no hand detected
        if not (results and results.multi_hand_landmarks):
            log_message = "No hand detected. Place your hand in front of the camera."
        else:
            hand_landmarks = results.multi_hand_landmarks[0].landmark
            current_time = time.time()
            # Only update log if 1 second has passed
            if current_time - last_log_time > 1:
                if is_fist(hand_landmarks):
                    log_message = "Closed fist = scroll down"
                    last_log_time = current_time
                elif is_palm(hand_landmarks):
                    log_message = "Open palm = scroll up"
                    last_log_time = current_time

        # Overlay log message
        if log_message:
            cv2.putText(
                frame,
                log_message,
                (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        # If window is closed by user (not ESC), show message
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("To close the webcam, please press the 'Esc' key.")
            break

        if key == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()