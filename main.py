import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Get screen width and height
screen_width, screen_height = pyautogui.size()

# Capture from the webcam
cap = cv2.VideoCapture(0)

# Moving average window size
SMOOTHING_WINDOW_SIZE = 5

# Queues to store recent positions
x_positions = deque(maxlen=SMOOTHING_WINDOW_SIZE)
y_positions = deque(maxlen=SMOOTHING_WINDOW_SIZE)

def get_smoothed_position(x, y):
    x_positions.append(x)
    y_positions.append(y)
    return int(np.mean(x_positions)), int(np.mean(y_positions))

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get landmark coordinates
            landmarks = hand_landmarks.landmark

            # Index finger tip
            index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x = int(index_finger_tip.x * screen_width)
            y = int(index_finger_tip.y * screen_height)

            # Smooth the cursor position
            smooth_x, smooth_y = get_smoothed_position(x, y)

            # Move the mouse
            pyautogui.moveTo(smooth_x, smooth_y)

            # Check for clicks
            # Thumb tip and index finger tip coordinates
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            thumb_tip_coords = np.array([thumb_tip.x, thumb_tip.y])
            index_tip_coords = np.array([index_finger_tip.x, index_finger_tip.y])

            # Distance between thumb and index finger
            distance = np.linalg.norm(thumb_tip_coords - index_tip_coords)

            # Left click
            if distance < 0.05:
                pyautogui.click()

            # Draw landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Air Mouse", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
