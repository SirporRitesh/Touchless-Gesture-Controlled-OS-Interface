import cv2
import time
import math
from hand_tracking import HandTracker
from gestures import get_scroll_direction, PalmTimer, is_pinch
from actions import scroll_up, scroll_down

# optional OS click library
try:
    import pyautogui as pag
except Exception:
    pag = None

# Scroll cooldown settings
SCROLL_DELAY = 0.2  # 200 milliseconds
PINCH_THRESHOLD = 0.05
PINCH_VIS_RADIUS = 18
PINCH_COOLDOWN = 0.5  # seconds between clicks per hand

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

    # Per-hand state
    last_pinch_state = {}   # e.g. {'Left': False, 'Right': False}
    last_click_time = {}    # e.g. {'Left': 0.0, 'Right': 0.0}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        frame, hand_data = tracker.find_hands(frame)
        h, w, _ = frame.shape

        if not hand_data:
            log_message = "No hands detected. Place your hands in front of the camera."
            palm_timer.reset()
        else:
            current_time = time.time()

            # handle pinch for every detected hand (Left/Right)
            for hand in hand_data:
                landmarks = hand['landmarks']
                label = hand.get('label', 'Unknown')  # expected "Left" or "Right"

                # initialize state entries
                if label not in last_pinch_state:
                    last_pinch_state[label] = False
                if label not in last_click_time:
                    last_click_time[label] = 0.0

                pinched, index_lm = is_pinch(landmarks, threshold=PINCH_THRESHOLD)

                # If we have an index landmark, compute frame coordinates for feedback
                if index_lm:
                    cx = int(index_lm.x * w)
                    cy = int(index_lm.y * h)
                else:
                    cx, cy = None, None

                # trigger on rising edge of pinch and respect cooldown
                if pinched and not last_pinch_state[label]:
                    if (current_time - last_click_time[label]) >= PINCH_COOLDOWN:
                        last_click_time[label] = current_time
                        # Visual feedback
                        if cx is not None and cy is not None:
                            cv2.circle(frame, (cx, cy), PINCH_VIS_RADIUS, (0, 0, 255), -1)
                        button = 'left' if label.lower().startswith('left') else 'right'
                        log_message = f"{label} hand: Pinch -> {button} click"

                        # perform OS click if possible
                        if pag:
                            try:
                                # IMPORTANT: click at the current OS cursor position instead of
                                # mapping from camera landmarks. This prevents the cursor from
                                # jumping/moving while you hold the pinch. We still use the
                                # rising-edge + cooldown logic above so each pinch triggers one click.
                                sx, sy = pag.position()
                                pag.click(x=sx, y=sy, button=button)
                            except Exception as e:
                                print("Warning: OS click failed:", e)
                        else:
                            # fallback: indicate simulated click
                            print(f"Simulated {button} click at normalized ({index_lm.x:.3f}, {index_lm.y:.3f}) for {label} hand.")

                # update last pinch state for edge detection
                last_pinch_state[label] = bool(pinched)

            # Continue existing palm/scroll behavior using first hand
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
                        if not log_message:
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