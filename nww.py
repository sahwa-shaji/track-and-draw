import numpy as np
import sys
import time
import os

try:
    import cv2
    import mediapipe as mp
    import pygame
    from collections import deque
except Exception as e:
    print("Import error:", e)
    sys.exit(1)

# === Pygame Setup ===
pygame.init()
screen_width, screen_height = 640, 480
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Button Triggered Video")
clock = pygame.time.Clock()

# === MediaPipe Hands Setup ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

trace_points = deque(maxlen=150)
shape_mode = False
colors = [(255, 255, 0), (0, 255, 0), (255, 0, 0)]
color_index = 0
draw_color = colors[color_index]
pinky_triggered = False

# === Video + Button Setup ===
button_size = 100
cooldown_duration = 5  # seconds
video_file = "vedio1.mp4"  # ✅ Ensure this file is in the same folder

buttons = [
    {"rect": pygame.Rect(50, 50, button_size, button_size), "last_trigger": 0, "label": "Button 1"},
    {"rect": pygame.Rect(490, 330, button_size, button_size), "last_trigger": 0, "label": "Button 2"},
]

font = pygame.font.SysFont(None, 24)

# === Camera Setup ===
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

running = True
while running:
    ret, frame = cap.read()
    if not ret:
        print("Camera not available.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    index_finger_pos = None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark
        index_x = int(lm[8].x * screen_width)
        index_y = int(lm[8].y * screen_height)
        index_finger_pos = (index_x, index_y)
        trace_points.append(index_finger_pos)

        # Pinky toggles
        pinky_x = int(lm[20].x * screen_width)
        if pinky_x < 100 and not pinky_triggered:
            shape_mode = not shape_mode
            print("Shape mode:", shape_mode)
            pinky_triggered = True
        elif pinky_x > screen_width - 100 and not pinky_triggered:
            color_index = (color_index + 1) % len(colors)
            draw_color = colors[color_index]
            print("Color changed")
            pinky_triggered = True
        elif 100 < pinky_x < screen_width - 100:
            pinky_triggered = False

        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # === Draw buttons and check for collisions ===
    current_time = time.time()
    for btn in buttons:
        pygame.draw.rect(screen, (255, 100, 100), btn["rect"])  # Red buttons
        label_surface = font.render(btn["label"], True, (0, 0, 0))
        screen.blit(label_surface, (btn["rect"].x + 10, btn["rect"].y + 10))

        if index_finger_pos and btn["rect"].collidepoint(index_finger_pos):
            if current_time - btn["last_trigger"] > cooldown_duration:
                btn["last_trigger"] = current_time
                print(f">>> {btn['label']} triggered!")
                try:
                    os.startfile(video_file)
                except Exception as e:
                    print(f"Could not play video: {e}")

    # === Display camera feed ===
    frame_surface = pygame.surfarray.make_surface(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).swapaxes(0, 1))
    screen.blit(frame_surface, (0, 0))

    # Draw finger trace
    if index_finger_pos:
        pygame.draw.circle(screen, draw_color, index_finger_pos, 10)

    if shape_mode and len(trace_points) > 10:
        points_np = np.array(trace_points, dtype=np.int32)
        (x, y), radius = cv2.minEnclosingCircle(points_np)
        pygame.draw.circle(screen, draw_color, (int(x), int(y)), int(radius), 2)
    elif len(trace_points) > 1:
        pygame.draw.lines(screen, draw_color, False, list(trace_points), 5)

    # === Event handling ===
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            running = False

    pygame.display.flip()
    clock.tick(30)

cap.release()
pygame.quit()


