'''
ASL Translator Data Collector

Collects hand landmark coordinates from your webcam for each letter/number.

Controls:
  SPACE - capture current landmark sample
  N     - skip to next letter
  B     - go back to previous letter
  Q     - quit and save progress
'''

import cv2
import mediapipe as mp
import numpy as np
import csv
import os

LABELS = list('abcdefghijklmnopqrstuvwxyz') + [str(i) for i in range(10)]
SAMPLES_PER_CLASS = 100
OUTPUT_PATH = "data/landmarks.csv"

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def normalize_landmarks(raw):
    reshaped = raw.reshape(21, 3)
    reshaped = reshaped - reshaped[0]
    scale = np.abs(reshaped).max()
    if scale > 0:
        reshaped /= scale
    return reshaped.flatten()


# Count already-collected samples per label
counts = {label: 0 for label in LABELS}
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, newline='') as f:
        for row in csv.reader(f):
            if row and row[0] in counts:
                counts[row[0]] += 1

# Start from first incomplete label
current_idx = 0
while current_idx < len(LABELS) and counts[LABELS[current_idx]] >= SAMPLES_PER_CLASS:
    current_idx += 1

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

flash_frames = 0

print("Starting data collection. Press SPACE to capture, N to skip, B to go back, Q to quit.")

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        if current_idx >= len(LABELS):
            print("All classes collected!")
            break

        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        landmarks = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
                raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                landmarks = normalize_landmarks(raw)

        current_label = LABELS[current_idx]
        n_collected = counts[current_label]
        progress = int((n_collected / SAMPLES_PER_CLASS) * (w - 40))

        # Flash green on capture
        if flash_frames > 0:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 8)
            flash_frames -= 1

        # UI
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"Sign: {current_label.upper()}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        total_done = sum(counts.values())
        total = len(LABELS) * SAMPLES_PER_CLASS
        cv2.putText(frame, f"{n_collected}/{SAMPLES_PER_CLASS}  |  total: {total_done}/{total}",
                    (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Progress bar
        cv2.rectangle(frame, (20, h - 30), (w - 20, h - 15), (60, 60, 60), -1)
        if progress > 0:
            cv2.rectangle(frame, (20, h - 30), (20 + progress, h - 15), (0, 255, 100), -1)

        if landmarks is None:
            cv2.putText(frame, "No hand detected", (w // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("ASL Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('n'):
            current_idx = min(current_idx + 1, len(LABELS) - 1)
        elif key == ord('b'):
            current_idx = max(current_idx - 1, 0)
        elif key == ord(' ') and landmarks is not None:
            with open(OUTPUT_PATH, 'a', newline='') as f:
                csv.writer(f).writerow([current_label] + landmarks.tolist())
            counts[current_label] += 1
            flash_frames = 3
            if counts[current_label] >= SAMPLES_PER_CLASS:
                current_idx += 1
                while current_idx < len(LABELS) and counts[LABELS[current_idx]] >= SAMPLES_PER_CLASS:
                    current_idx += 1

cap.release()
cv2.destroyAllWindows()
print(f"Done. Total samples collected: {sum(counts.values())}")
