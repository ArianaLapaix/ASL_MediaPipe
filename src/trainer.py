import csv
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

DATA_PATH = "../data/landmarks.csv"
MODEL_PATH = "../models/asl_model.pkl"


def load_dataset():
    X, y = [], []
    with open(DATA_PATH, newline='') as f:
        for row in csv.reader(f):
            if len(row) == 64:
                y.append(row[0])
                X.append([float(v) for v in row[1:]])
    return np.array(X), np.array(y)


def train():
    print("Loading landmark data...")
    X, y = load_dataset()
    labels = sorted(set(y))
    print(f"Total samples: {len(X)} across {len(labels)} classes")

    if len(X) == 0:
        print("No data found. Run data_collector.py first.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training MLP...")
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=100,
        verbose=True,
        random_state=42,
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\nTest accuracy: {acc * 100:.1f}%")
    print(classification_report(y_test, model.predict(X_test), target_names=labels))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "labels": labels}, f)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
