# ASL Fingerspelling Translator

This project is a real-time fingerspelling translator that uses MediaPipe to detect hand landmarks and translate them into text. Point your webcam at your hand and start fingerspelling A-Z or 0-9 and see the live translation

![Python](https://img.shields.io/badge/Python-3.10-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-yellow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-purple)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)

## How It Works

1. **Hand Detection:** - MediaPipe Hands detects 21 3D hand landmarks in real-time.
2. **Feature Extraction:** - The landmarks are normalized and converted into a feature vector. Each coordinate is extracted and normalized with respect to the wrist.
3. **Classification:** - The feature vector is passed through a pre-trained machine learning model (e.g., SVM, Random Forest, or Neural Network) to classify the sign into an ASL letter (A-Z) or number (0-9).
4. **Display:** - The predicted letter or number is displayed on the screen overlaying the webcam feed.

## Project Structure

```
ASL_MediaPipe/
│
├── README.md           # Project documentation
├── requirements.txt    # Project dependencies
├── .gitignore          # Git ignore file
├── src/
│   ├── hand_detection.py           # Step 1: Live MediaPipe hand landmark detection
│   ├── data_collector.py           # Step 2: Collect training data
│   ├── trainer.py                  # Step 3: Train classifier
│   └── predict_live.py             # Step 4: Predict real time on live video feed
├── models/
│   └── .gitkeep                   # Trained machine learning model
└── data/
│   └── .gitkeep                   # Training data
└── .env/
    └── .gitkeep               # Demogifs/screenshots/whatever
```
## Setup

### Prerequisites

- Python 3.10
- Webcam

### Installation

```bash
# Clone the repository
git clone https://github.com/arianalapaix/ASL_MediaPipe.git
cd ASL_MediaPipe 

# Create and Activate a Conda Environment (Recommended)
conda create -n asl python=3.10 -y
conda activate asl 

# Install Dependencies
python -m pip install -r requirements.txt

# OR Install Dependencies using pip (Alternative)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Quick Start

```bash
# Run the application
python src/predict_live.py
```
* Press 's' to print current landmark coordinates to the console
* Press 'q' to quit

## Development Roadmap

- [x] Live hand landmark detection from webcam with MediaPipe
- [ ] Data collection / loading (Dataset with landmark coordinated for A-Z, 0-9) for ASL fingerspelling 
- [ ] Model training for ASL fingerspelling
- [ ] Live  prediction of ASL fingerspelling overlayed on the webcam feed
- [ ] UI for ASL fingerspelling 
- [ ] Sentence building (buffer consecutive predictions into words)

# Tech Stack
| Component | Tool |
|---|---|
| Hand Detection | MediaPipe Hands (0.10.21) |
| Video Capture | OpenCV |
| Classifier | scikit-learn / PyTorch (TBD) |
| Language | Python 3.10 |

 > **Note:** This project uses `mediapipe==0.10.21` because the legacy `mp.solutions.hands` API was removed in later versions (0.10.31+). The underlying hand landmark model is unchanged. A future update may migrate to the `mediapipe.tasks.vision.HandLandmarker` API.

 ## License

 MIT

