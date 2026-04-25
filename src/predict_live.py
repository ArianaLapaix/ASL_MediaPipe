import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

MODEL_PATH = "../models/asl_model.pkl"

LETTER_CONFIRM_FRAMES = 20  # frames a letter must be held to register (~0.7s)
SPACE_AFTER_SECONDS = 1.5   # no hand this long → add a space
IDLE_CLEAR_SECONDS = 5.0    # no hand this long → clear sentence

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)
model = saved["model"]
scaler = saved["scaler"]
labels = saved["labels"]


def normalize_landmarks(raw):
    reshaped = raw.reshape(21, 3)
    reshaped = reshaped - reshaped[0]
    scale = np.abs(reshaped).max()
    if scale > 0:
        reshaped /= scale
    return reshaped.flatten()


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

sentence = ""
current_letter = None
letter_frame_count = 0
last_confirmed_letter = None
last_hand_time = time.time()
space_added = False

print("Starting live prediction... Press 'q' to quit, 'c' to clear.")

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        idle_seconds = now - last_hand_time

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        pred_letter = None
        confidence = 0.0

        if results.multi_hand_landmarks:
            last_hand_time = now
            space_added = False
            idle_seconds = 0.0

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
                raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                features = scaler.transform([normalize_landmarks(raw)])
                pred_letter = model.predict(features)[0]
                confidence = model.predict_proba(features).max()

            # Count consecutive frames of the same letter
            if pred_letter == current_letter:
                letter_frame_count += 1
            else:
                current_letter = pred_letter
                letter_frame_count = 1
                last_confirmed_letter = None

            # Confirm letter after holding long enough
            if letter_frame_count >= LETTER_CONFIRM_FRAMES and pred_letter != last_confirmed_letter:
                sentence += pred_letter.upper()
                last_confirmed_letter = pred_letter

        else:
            current_letter = None
            letter_frame_count = 0
            last_confirmed_letter = None

            # Add space after short idle
            if idle_seconds >= SPACE_AFTER_SECONDS and not space_added and sentence and not sentence.endswith(" "):
                sentence += " "
                space_added = True

            # Clear after long idle
            if idle_seconds >= IDLE_CLEAR_SECONDS and sentence:
                sentence = ""
                space_added = False

        # ── Draw UI ──────────────────────────────────────────────

        # Current letter + hold progress bar (top left)
        if pred_letter:
            label_text = f"{pred_letter.upper()}  {confidence * 100:.0f}%"
            cv2.putText(frame, label_text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
            bar_w = int((letter_frame_count / LETTER_CONFIRM_FRAMES) * 150)
            bar_w = min(bar_w, 150)
            cv2.rectangle(frame, (20, 60), (170, 75), (60, 60, 60), -1)
            cv2.rectangle(frame, (20, 60), (20 + bar_w, 75), (0, 255, 100), -1)
        else:
            # Idle countdown bar (top left)
            if idle_seconds < IDLE_CLEAR_SECONDS:
                remaining = max(0.0, IDLE_CLEAR_SECONDS - idle_seconds)
                bar_w = int((remaining / IDLE_CLEAR_SECONDS) * 150)
                cv2.rectangle(frame, (20, 60), (170, 75), (60, 60, 60), -1)
                cv2.rectangle(frame, (20, 60), (20 + bar_w, 75), (0, 100, 255), -1)
                cv2.putText(frame, f"clear in {remaining:.1f}s", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

        # Sentence display (bottom, word-wrapped if needed)
        display = sentence if sentence else ""
        box_h = 70
        cv2.rectangle(frame, (0, h - box_h), (w, h), (0, 0, 0), -1)
        if display:
            font_scale = 1.4
            text_size = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            # Scale down if text is too wide
            if text_size[0] > w - 40:
                font_scale *= (w - 40) / text_size[0]
            text_size = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = max(20, (w - text_size[0]) // 2)
            cv2.putText(frame, display, (text_x, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

        cv2.imshow("ASL Translator", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            sentence = ""

cap.release()
cv2.destroyAllWindows()
