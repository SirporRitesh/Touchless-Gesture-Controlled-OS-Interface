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