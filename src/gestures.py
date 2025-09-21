import mediapipe as mp
import time
import math

mp_hands = mp.solutions.hands

def is_fist(landmarks):
    """
    Returns True if the hand is a closed fist.
    Placeholder: checks if all finger tips are below their respective MCP joints.
    """
    if not landmarks:
        return False
    # Indexes for finger tips and MCPs
    tips = [8, 12, 16, 20]
    mcps = [5, 9, 13, 17]
    for tip, mcp in zip(tips, mcps):
        if landmarks[tip].y < landmarks[mcp].y:
            return False
    return True

def is_palm(landmarks):
    """
    Returns True if the hand is an open palm.
    Placeholder: checks if all finger tips are above their respective MCP joints.
    """
    if not landmarks:
        return False
    tips = [8, 12, 16, 20]
    mcps = [5, 9, 13, 17]
    for tip, mcp in zip(tips, mcps):
        if landmarks[tip].y > landmarks[mcp].y:
            return False
    return True

def is_peace_sign(landmarks):
    """
    Returns True if the hand is making a peace sign (index and middle up, others down).
    Works for both upward and downward peace signs.
    """
    if not landmarks:
        return False
    
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]
    
    # Calculate distances to see if fingers are extended
    def distance(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    # Index and middle should be extended (tip far from palm)
    index_extended = distance(index_tip, index_pip) > 0.04
    middle_extended = distance(middle_tip, middle_pip) > 0.04
    
    # Ring and pinky should be folded (tip close to mcp)
    ring_folded = distance(ring_tip, ring_mcp) < 0.06
    pinky_folded = distance(pinky_tip, pinky_mcp) < 0.06
    
    return index_extended and middle_extended and ring_folded and pinky_folded

def is_open_palm(landmarks):
    """
    Returns True if the hand is an open palm (all fingers extended).
    """
    if not landmarks:
        return False
    
    # Check if all finger tips are above their respective MCP joints
    tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    mcps = [5, 9, 13, 17]   # Corresponding MCP joints
    
    for tip, mcp in zip(tips, mcps):
        if landmarks[tip].y > landmarks[mcp].y:  # Finger tip below MCP = folded
            return False
    
    # Also check thumb
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    
    # For thumb, check horizontal distance (x-axis) as it moves sideways
    if abs(thumb_tip.x - thumb_mcp.x) < 0.04:  # Thumb not extended
        return False
    
    return True

def get_scroll_direction(landmarks):
    """
    Returns 'up', 'down', or None based on peace sign orientation.
    """
    if not is_peace_sign(landmarks):
        return None
    
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    
    # Calculate average finger position
    avg_finger_y = (index_tip.y + middle_tip.y) / 2
    
    # Check orientation relative to wrist with some tolerance
    if avg_finger_y < wrist.y - 0.05:  # Fingers significantly above wrist
        return 'up'
    elif avg_finger_y > wrist.y + 0.05:  # Fingers significantly below wrist
        return 'down'
    else:
        return None  # Neutral position

class PalmTimer:
    """
    Tracks how long an open palm gesture has been sustained.
    """
    def __init__(self, timeout_seconds=5):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
        self.is_timing = False
    
    def update(self, landmarks):
        """
        Update the timer based on current hand state.
        Returns True if palm has been open for the timeout duration.
        """
        if is_open_palm(landmarks):
            if not self.is_timing:
                # Start timing
                self.start_time = time.time()
                self.is_timing = True
            else:
                # Check if timeout reached
                elapsed = time.time() - self.start_time
                if elapsed >= self.timeout_seconds:
                    return True
        else:
            # Reset timer if palm is not open
            self.reset()
        
        return False
    
    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.is_timing = False
    
    def get_elapsed_time(self):
        """Get elapsed time since palm was first detected."""
        if self.is_timing and self.start_time:
            return time.time() - self.start_time
        return 0

def is_pinch(landmarks, threshold=0.05):
    """
    Detects a pinch between thumb tip and index tip.
    Returns (True, index_landmark) when pinch detected; (False, None) otherwise.
    """
    if not landmarks:
        return False, None

    thumb = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    dist = math.hypot(thumb.x - index.x, thumb.y - index.y)
    return (dist < threshold), index