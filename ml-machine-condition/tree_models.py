"""
Machine condition classification: NORMAL / FAULT (binary classification)
Part 2 of the group project: Decision Tree, Random Forest and AdaBoost.
Uses the same dataset, split and evaluation as svm_naive_bayes.py.
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report,
                             ConfusionMatrixDisplay)

from machine_data import load_data, FEATURES, LABELS

FOCUS = "FAULT"  # positive class for the one-vs-rest style metrics
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 1. Data ----------
df = load_data()
X = df[FEATURES]
y = df["Condition"]
print("Class distribution:\n", y.value_counts())

# ---------- 2. Split (same as svm_naive_bayes.py) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
print(f"\nTraining records: {len(X_train)} | Testing records: {len(X_test)}")

# ---------- 3. Models ----------
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
}

# ---------- 4. Train + evaluate ----------
results = []
for name, model in models.items():
    print("\n" + "=" * 60 + f"\n{name.upper()}\n" + "=" * 60)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Binary: macro average treats NORMAL and FAULT equally.
    results.append({
        "Algorithm": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1-score": f1_score(y_test, y_pred, average="macro", zero_division=0),
    })
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    f_true, f_pred = (y_test == FOCUS), (y_pred == FOCUS)
    print(f"[{FOCUS} focus] precision={precision_score(f_true, f_pred, zero_division=0):.3f} "
          f"recall={recall_score(f_true, f_pred, zero_division=0):.3f} "
          f"f1={f1_score(f_true, f_pred, zero_division=0):.3f}")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=LABELS, cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"{name}: Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_{name.replace(' ', '_').lower()}.png", dpi=150)
    plt.show()

# ---------- 5. Compare ----------
results_df = pd.DataFrame(results).set_index("Algorithm")
print("\nMODEL COMPARISON (macro-averaged)")
print(results_df.round(3))
results_df.plot(kind="bar", figsize=(8, 5), ylim=(0, 1.05))
plt.title("Decision Tree vs. Random Forest vs. AdaBoost")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/comparison_tree_models.png", dpi=150)
plt.show()

# ---------- 6. New machine ----------
new_machine = pd.DataFrame([{"Temperature_C": 89, "Vibration_mm_s": 6.8,
                             "Speed_RPM": 1325, "Operating_Hours": 6500}])
print("\nNEW MACHINE:\n", new_machine)
for name, model in models.items():
    print(f"{name:14s} prediction: {model.predict(new_machine)[0]}")
