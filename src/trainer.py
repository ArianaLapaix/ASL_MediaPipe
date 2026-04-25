import os
import cv2
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = "../data/asl_dataset"
MODEL_PATH = "../models/asl_model.pkl"
CACHE_PATH = "../data/features_cache.npz"
IMG_SIZE = 64

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.flatten()

def load_dataset():
    X, y = [], []
    labels = sorted([l for l in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, l))])

    for label in labels:
        folder = os.path.join(DATA_DIR, label)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"Processing '{label}': {len(files)} images")
        for fname in files:
            features = load_image(os.path.join(folder, fname))
            if features is not None:
                X.append(features)
                y.append(label)

    return np.array(X), np.array(y)

def train():
    if os.path.exists(CACHE_PATH):
        print("Loading cached features...")
        cache = np.load(CACHE_PATH, allow_pickle=True)
        X, y = cache["X"], cache["y"]
    else:
        print("Processing images (this only happens once)...")
        X, y = load_dataset()
        np.savez(CACHE_PATH, X=X, y=y)
        print("Features cached.")

    print(f"\nTotal samples: {len(X)} across {len(set(y))} classes")

    if len(X) == 0:
        print("No images loaded. Check your DATA_DIR path.")
        return

    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training MLP...")
    model = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=50, verbose=True, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\nTest accuracy: {acc * 100:.1f}%")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "labels": sorted(set(y))}, f)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
