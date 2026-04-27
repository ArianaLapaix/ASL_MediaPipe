import csv
import os
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_PATHS = [
     "data/dataset_landmarks.csv",  # processed from image datasets
    # "data/landmarks.csv",          # collected from webcam
]
MODEL_PATH = "models/asl_model.pkl"

MAX_ITER       = 100
N_ITER_NO_CHANGE = 15   # early stopping patience
EVAL_SAMPLE    = 5000   # training samples used to compute train metrics per iteration


def load_dataset():
    X, y = [], []
    for path in DATA_PATHS:
        if not os.path.exists(path):
            print(f"  (skipping {path} — not found)")
            continue
        print(f"  loading {path}...")
        with open(path, newline='') as f:
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
    X_test  = scaler.transform(X_test)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    # Hold out 10% of training data as validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train_enc, test_size=0.1, random_state=42, stratify=y_train_enc
    )

    # Fixed sample for computing training metrics each iteration (for speed)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_tr), min(EVAL_SAMPLE, len(X_tr)), replace=False)
    X_sample, y_sample = X_tr[sample_idx], y_tr[sample_idx]

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=1,
        warm_start=True,
        random_state=42,
        verbose=False,
    )

    train_losses, train_accs = [], []
    val_losses,   val_accs   = [], []
    best_val_loss = np.inf
    no_improve    = 0

    print(f"\nTraining MLP (up to {MAX_ITER} iterations)...\n")
    print(f"{'Iter':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'Train Acc':>10}  {'Val Acc':>10}")
    print("-" * 55)

    for i in range(MAX_ITER):
        model.fit(X_tr, y_tr)

        tl = log_loss(y_sample, model.predict_proba(X_sample))
        ta = accuracy_score(y_sample, model.predict(X_sample)) * 100
        vl = log_loss(y_val, model.predict_proba(X_val))
        va = accuracy_score(y_val, model.predict(X_val)) * 100

        train_losses.append(tl)
        train_accs.append(ta)
        val_losses.append(vl)
        val_accs.append(va)

        print(f"{i+1:>5}  {tl:>10.4f}  {vl:>10.4f}  {ta:>9.1f}%  {va:>9.1f}%")

        if vl < best_val_loss - 1e-4:
            best_val_loss = vl
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= N_ITER_NO_CHANGE:
                print(f"\nEarly stopping at iteration {i+1}.")
                break

    acc = accuracy_score(y_test_enc, model.predict(X_test))
    print(f"\nTest accuracy: {acc * 100:.1f}%")
    print(classification_report(y_test_enc, model.predict(X_test), target_names=le.classes_))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model":        model,
            "scaler":       scaler,
            "le":           le,
            "labels":       labels,
            "X_test":       X_test,
            "y_test":       y_test_enc,
            "train_losses": train_losses,
            "train_accs":   train_accs,
            "val_losses":   val_losses,
            "val_accs":     val_accs,
        }, f)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
