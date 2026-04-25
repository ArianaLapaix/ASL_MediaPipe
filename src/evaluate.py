'''
ASL Model Evaluation

Generates four plots saved to plots/:
  1. confusion_matrix.png   — which letters get confused with each other
  2. per_class_accuracy.png — accuracy per letter/digit
  3. loss_curve.png         — training loss over iterations
  4. precision_recall_f1.png — precision, recall, and F1 per class
'''

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)

MODEL_PATH  = "../models/asl_model.pkl"
PLOTS_DIR   = "../plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

print("Loading model...")
with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)

model  = saved["model"]
scaler = saved["scaler"]
labels = saved["labels"]

if "X_test" in saved:
    X_test = saved["X_test"]
    y_test = saved["y_test"]
else:
    # Retrain split not saved — reload dataset and reproduce the same split
    import csv
    from sklearn.model_selection import train_test_split
    DATA_PATHS = ["../data/dataset_landmarks.csv", "../data/landmarks.csv"]
    X_all, y_all = [], []
    for path in DATA_PATHS:
        if not os.path.exists(path):
            continue
        with open(path, newline='') as f:
            for row in csv.reader(f):
                if len(row) == 64:
                    y_all.append(row[0])
                    X_all.append([float(v) for v in row[1:]])
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    _, X_test, _, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
    X_test = scaler.transform(X_test)
    print("(Reproduced test split from dataset)")

y_pred = model.predict(X_test)
overall_acc = accuracy_score(y_test, y_pred)
print(f"Overall test accuracy: {overall_acc * 100:.1f}%")

display_labels = [l.upper() for l in labels]

# ── 1. Confusion Matrix ───────────────────────────────────────────────────────
print("Plotting confusion matrix...")
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, ax = plt.subplots(figsize=(18, 15))
sns.heatmap(
    cm_pct, annot=True, fmt=".0f", cmap="Blues",
    xticklabels=display_labels, yticklabels=display_labels,
    linewidths=0.4, linecolor="lightgrey",
    cbar_kws={"label": "% of true class"},
    ax=ax,
)
ax.set_xlabel("Predicted", fontsize=13, labelpad=10)
ax.set_ylabel("Actual", fontsize=13, labelpad=10)
ax.set_title(f"Confusion Matrix  (overall accuracy: {overall_acc*100:.1f}%)", fontsize=15, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# ── 2. Per-class Accuracy ─────────────────────────────────────────────────────
print("Plotting per-class accuracy...")
per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
sorted_idx = np.argsort(per_class_acc)
sorted_labels = [display_labels[i] for i in sorted_idx]
sorted_acc    = per_class_acc[sorted_idx]

colors = ["#d9534f" if a < 70 else "#f0ad4e" if a < 90 else "#5cb85c" for a in sorted_acc]

fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.barh(sorted_labels, sorted_acc, color=colors, edgecolor="white", height=0.7)
ax.axvline(overall_acc * 100, color="steelblue", linestyle="--", linewidth=1.5, label=f"Overall avg ({overall_acc*100:.1f}%)")
ax.set_xlabel("Accuracy (%)", fontsize=12)
ax.set_title("Per-class Accuracy", fontsize=14)
ax.set_xlim(0, 105)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.legend(fontsize=11)
for bar, val in zip(bars, sorted_acc):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "per_class_accuracy.png"), dpi=150)
plt.close()

# ── 3. Training Loss Curve ────────────────────────────────────────────────────
print("Plotting loss curve...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(model.loss_curve_, color="steelblue", linewidth=2)
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Training Loss Curve", fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "loss_curve.png"), dpi=150)
plt.close()

# ── 4. Precision / Recall / F1 per class ─────────────────────────────────────
print("Plotting precision / recall / F1...")
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=labels)

x = np.arange(len(labels))
width = 0.28

fig, ax = plt.subplots(figsize=(18, 6))
ax.bar(x - width, precision * 100, width, label="Precision", color="#5bc0de", edgecolor="white")
ax.bar(x,         recall    * 100, width, label="Recall",    color="#5cb85c", edgecolor="white")
ax.bar(x + width, f1        * 100, width, label="F1 Score",  color="#f0ad4e", edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(display_labels, fontsize=10)
ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("Precision, Recall, and F1 Score per Class", fontsize=14)
ax.set_ylim(0, 110)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "precision_recall_f1.png"), dpi=150)
plt.close()

print(f"\nAll plots saved to {os.path.abspath(PLOTS_DIR)}/")
