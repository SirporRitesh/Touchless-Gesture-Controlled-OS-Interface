import pyautogui

def scroll_up():
    """Scroll up in the active window."""
    pyautogui.scroll(100)

def scroll_down():
    """Scroll down in the active window."""
    pyautogui.scroll(-100)

def left_click():
    """Perform a left mouse click."""
    pyautogui.click()

def move_cursor(x, y):
    """Move cursor to specified coordinates."""
    pyautogui.moveTo(x, y)