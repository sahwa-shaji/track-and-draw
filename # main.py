# main.py
import sys
try:
    import cv2
    import mediapipe as mp
    import pygame
    from collections import deque
except Exception as e:
    print("Import error:", e)
    print("Make sure you installed: pip install opencv-python pygame mediapipe")
    sys.exit(1)

# Pygame init
pygame.init()
screen_width, screen_height = 640, 480
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Index Finger Tracking")
clock = pygame.time.Clock()

# MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

trace_points = deque(maxlen=150)

# Open camera (use CAP_DSHOW on Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

running = True
while running:
    ret, frame = cap.read()
    if not ret:
        print("Can't read from camera. Close other apps using camera or try different index.")
        break

    frame = cv2.flip(frame, 1)  # mirror
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    index_finger_pos = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark[8]  # index finger tip
        x = int(lm.x * screen_width)
        y = int(lm.y * screen_height)
        index_finger_pos = (x, y)
        trace_points.append(index_finger_pos)
        # Optional OpenCV drawing (for debugging)
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Convert to Pygame surface and draw
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    screen.blit(frame_surface, (0, 0))

    if len(trace_points) > 1:
        pygame.draw.lines(screen, (255, 255, 0), False, list(trace_points), 5)

    if index_finger_pos:
        pygame.draw.circle(screen, (255, 255, 0), index_finger_pos, 10)

    # Events: close window or press 'q'
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    pygame.display.flip()
    clock.tick(30)

cap.release()
pygame.quit()
