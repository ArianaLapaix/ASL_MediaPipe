# ASL Fingerspelling Translator

A real-time American Sign Language fingerspelling translator that uses MediaPipe to detect hand landmarks and an MLP classifier to translate signs into text. Point your webcam at your hand, fingerspell A–Z or 0–9, and watch the sentence build on screen.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-yellow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-purple)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## Demo

> Point your webcam at your hand and fingerspell — the predicted letter appears at the bottom of the screen. Hold a sign for ~0.7s to confirm it. Remove your hand for 1.5s to add a space, or 5s to clear the sentence.

## How It Works

1. **Hand Detection** — MediaPipe Hands detects 21 3D hand landmarks in real-time from the webcam feed.
2. **Feature Extraction** — The 21 landmarks (63 values: x, y, z per point) are normalized relative to the wrist and scaled, producing a compact, position- and scale-invariant feature vector.
3. **Classification** — A Multi-Layer Perceptron (MLP) trained on 100k+ landmark samples classifies the hand pose into one of 36 classes (A–Z, 0–9).
4. **Sentence Building** — Confirmed letters accumulate into a sentence displayed at the bottom of the screen, with automatic spacing and clearing on idle.

## Project Structure

```
ASL_MediaPipe/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── hand_detection.py       # MediaPipe landmark detection (exploration/dev)
│   ├── data_collector.py       # Step 1: Collect webcam landmark training data
│   ├── process_dataset.py      # Step 1 (alt): Extract landmarks from image datasets
│   ├── trainer.py              # Step 2: Train the MLP classifier
│   ├── evaluate.py             # Step 3: Generate accuracy plots
│   └── predict_live.py         # Step 4: Real-time prediction
├── models/                     # Saved model (git-ignored)
├── data/                       # Training data (git-ignored)
└── plots/                      # Evaluation plots (git-ignored)
```

## Setup

### Prerequisites

- Python 3.10+
- Webcam

### Installation

```bash
# Clone the repository
git clone https://github.com/arianalapaix/ASL_MediaPipe.git
cd ASL_MediaPipe

# Create and activate a Conda environment (recommended)
conda create -n asl python=3.10 -y
conda activate asl

# Install dependencies
python -m pip install -r requirements.txt
```

## Training Your Own Model

The model and training data are not included in this repo (too large). You have two options to train your own:

### Option A — Use your webcam (recommended for personal use)

Collect landmark data directly from your hand:

```bash
python src/data_collector.py
```

- The screen shows which letter to sign
- Press `SPACE` to capture a sample (100 per letter)
- Press `N` to skip, `B` to go back, `Q` to save and quit
- Progress is saved automatically — you can quit and resume anytime

Then train:

```bash
python src/trainer.py
```

### Option B — Use an image dataset

If you have an ASL image dataset, run MediaPipe across all images overnight to extract landmarks:

```bash
python src/process_dataset.py
```

This checkpoints progress and can be resumed if interrupted. Once complete, run `trainer.py` as above.

**Compatible dataset structure:**
```
data/ASL_Datasets/
├── ASL_Alphabet_Dataset/
│   ├── asl_alphabet_train/{A-Z}/   # image files
│   └── asl_alphabet_test/{A-Z}/
└── American Sign Language Digits Dataset/
    └── {0-9}/
        └── Input Images - Sign {N}/  # image files
```

Suggested datasets:
- [ASL Alphabet (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- [ASL Digits (Kaggle)](https://www.kaggle.com/datasets/rayeed045/american-sign-language-digit-dataset)

## Running the Translator

```bash
python src/predict_live.py
```

| Action | Result |
|---|---|
| Hold a sign for ~0.7s | Letter is confirmed and added |
| Remove hand for 1.5s | Space added |
| Remove hand for 5s | Sentence clears |
| Press `C` | Clear sentence instantly |
| Press `Q` | Quit |

## Evaluating the Model

Generate accuracy plots (confusion matrix, per-class accuracy, loss curve, precision/recall/F1):

```bash
python src/evaluate.py
```

Plots are saved to the `plots/` folder.

## Tech Stack

| Component | Tool |
|---|---|
| Hand Detection | MediaPipe Hands 0.10.21 |
| Video Capture | OpenCV 4.10 |
| Classifier | scikit-learn MLP (256 → 128 hidden layers) |
| Evaluation | matplotlib, seaborn |
| Language | Python 3.10 |

> **Note:** This project uses `mediapipe==0.10.21` because the `mp.solutions.hands` API was removed in later versions (0.10.31+). The underlying hand landmark model is unchanged.

## Roadmap

- [x] Live hand landmark detection from webcam
- [x] Webcam-based training data collection
- [x] Image dataset landmark extraction
- [x] MLP classifier for A–Z and 0–9
- [x] Real-time live prediction overlay
- [x] Sentence building with auto-space and auto-clear
- [x] Model evaluation plots
- [ ] Migrate to `mediapipe.tasks.vision.HandLandmarker` API
- [ ] Word suggestions / autocomplete
- [ ] Text-to-speech output

## License

MIT
