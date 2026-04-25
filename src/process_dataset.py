'''
ASL Dataset Processor

Runs MediaPipe on every image in the alphabet and digits datasets,
extracts hand landmarks, and saves them to data/dataset_landmarks.csv.

Run overnight — checkpoints progress so it can resume if interrupted.

Usage:
    python process_dataset.py
'''

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

ALPHABET_TRAIN  = "../data/ASL_Datasets/ASL_Alphabet_Dataset/asl_alphabet_train"
ALPHABET_TEST   = "../data/ASL_Datasets/ASL_Alphabet_Dataset/asl_alphabet_test"
DIGITS_DIR      = "../data/ASL_Datasets/American Sign Language Digits Dataset"
OUTPUT_PATH     = "../data/dataset_landmarks.csv"
CHECKPOINT_PATH = "../data/.process_checkpoint"
RESIZE          = 640       # resize long edge before MediaPipe (speeds things up)
SKIP_LABELS     = {"del", "nothing", "space"}

mp_hands = mp.solutions.hands


def normalize_landmarks(raw):
    reshaped = raw.reshape(21, 3)
    reshaped = reshaped - reshaped[0]
    scale = np.abs(reshaped).max()
    if scale > 0:
        reshaped /= scale
    return reshaped.flatten()


def resize_for_mediapipe(img):
    h, w = img.shape[:2]
    if max(h, w) <= RESIZE:
        return img
    scale = RESIZE / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def collect_image_paths():
    paths = []  # list of (image_path, label)

    # Alphabet train + test
    for root_dir in [ALPHABET_TRAIN, ALPHABET_TEST]:
        if not os.path.isdir(root_dir):
            continue
        for label_folder in sorted(os.listdir(root_dir)):
            if label_folder.lower() in SKIP_LABELS:
                continue
            label = label_folder.lower()
            folder = os.path.join(root_dir, label_folder)
            if not os.path.isdir(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append((os.path.join(folder, fname), label))

    # Digits — each digit folder has an "Input Images - Sign N" subfolder
    if os.path.isdir(DIGITS_DIR):
        for digit in sorted(os.listdir(DIGITS_DIR)):
            digit_folder = os.path.join(DIGITS_DIR, digit)
            if not os.path.isdir(digit_folder):
                continue
            for subfolder in os.listdir(digit_folder):
                if subfolder.startswith("Input Images"):
                    img_folder = os.path.join(digit_folder, subfolder)
                    for fname in sorted(os.listdir(img_folder)):
                        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                            paths.append((os.path.join(img_folder, fname), digit))

    return paths


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return int(f.read().strip())
    return 0


def save_checkpoint(n):
    with open(CHECKPOINT_PATH, "w") as f:
        f.write(str(n))


def main():
    print("Scanning dataset folders...")
    all_paths = collect_image_paths()
    total = len(all_paths)
    print(f"Found {total:,} images total.")

    start_from = load_checkpoint()
    if start_from > 0:
        print(f"Resuming from image {start_from:,} / {total:,}")

    detected = 0
    failed = 0
    start_time = time.time()

    write_mode = "a" if start_from > 0 else "w"

    with open(OUTPUT_PATH, write_mode, newline='') as out_f:
        writer = csv.writer(out_f)

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        ) as hands:

            for i, (img_path, label) in enumerate(all_paths):
                if i < start_from:
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    failed += 1
                    save_checkpoint(i + 1)
                    continue

                img = resize_for_mediapipe(img)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    lm = results.multi_hand_landmarks[0]
                    raw = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
                    features = normalize_landmarks(raw)
                    writer.writerow([label] + features.tolist())
                    out_f.flush()
                    detected += 1
                else:
                    failed += 1

                save_checkpoint(i + 1)

                # Progress every 500 images
                if (i + 1) % 500 == 0 or (i + 1) == total:
                    elapsed = time.time() - start_time
                    done = i + 1 - start_from
                    rate = done / elapsed if elapsed > 0 else 0
                    remaining = (total - i - 1) / rate if rate > 0 else 0
                    h, m, s = int(remaining // 3600), int((remaining % 3600) // 60), int(remaining % 60)
                    print(
                        f"[{i+1:>7,}/{total:,}]  "
                        f"detected: {detected:,}  "
                        f"skipped: {failed:,}  "
                        f"rate: {rate:.1f} img/s  "
                        f"ETA: {h:02d}:{m:02d}:{s:02d}"
                    )

    # Clear checkpoint on clean finish
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    print(f"\nDone! {detected:,} landmarks saved to {OUTPUT_PATH}")
    print(f"{failed:,} images had no hand detected and were skipped.")


if __name__ == "__main__":
    main()
