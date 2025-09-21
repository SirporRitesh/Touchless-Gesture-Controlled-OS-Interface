import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandTracker:
    def __init__(self, max_num_hands=2):
        self.hands = mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    def find_hands(self, frame):
        # Flip the frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        hand_data = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get hand label (Left or Right)
                hand_label = handedness.classification[0].label
                
                # Store hand data
                hand_data.append({
                    'landmarks': hand_landmarks.landmark,
                    'label': hand_label
                })
                
                # Draw hand label on the frame
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                h, w, _ = frame.shape
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, hand_label, (cx-30, cy-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return frame, hand_data