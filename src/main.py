import cv2
import time
from hand_tracking import HandTracker
from gestures import get_scroll_direction, PalmTimer
from actions import scroll_up, scroll_down

# Scroll cooldown settings
SCROLL_DELAY = 0.2  # 200 milliseconds

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    palm_timer = PalmTimer(timeout_seconds=5)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    window_name = 'Webcam'
    cv2.namedWindow(window_name)
    last_scroll_time = 0
    log_message = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        frame, hand_data = tracker.find_hands(frame)

        if not hand_data:
            log_message = "No hands detected. Place your hands in front of the camera."
            palm_timer.reset()
        else:
            current_time = time.time()
            
            # Process the first detected hand
            hand = hand_data[0]
            landmarks = hand['landmarks']
            label = hand['label']
            
            # Check for sustained open palm (5 seconds = exit)
            if palm_timer.update(landmarks):
                log_message = "Open palm held for 5 seconds. Exiting..."
                cv2.putText(frame, log_message, (10, frame.shape[0] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow(window_name, frame)
                cv2.waitKey(2000)  # Show message for 2 seconds
                break
            
            # Show countdown if palm is being held
            elapsed = palm_timer.get_elapsed_time()
            if elapsed > 0:
                remaining = 5 - elapsed
                log_message = f"Hold palm to exit: {remaining:.1f}s remaining"
            else:
                # Check for scroll gesture (with cooldown)
                if current_time - last_scroll_time > SCROLL_DELAY:
                    scroll_direction = get_scroll_direction(landmarks)
                    if scroll_direction == 'up':
                        scroll_up()
                        log_message = f"{label} hand: Peace Sign - Scroll Up"
                        last_scroll_time = current_time
                    elif scroll_direction == 'down':
                        scroll_down()
                        log_message = f"{label} hand: Peace Sign - Scroll Down"
                        last_scroll_time = current_time
                    else:
                        log_message = f"{label} hand detected"

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

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("To close the webcam, please press the 'Esc' key.")
            break

        if key == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()