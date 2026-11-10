"""
Machine condition classification: NORMAL / WARNING / FAULT
Part 1 of the group project: Support Vector Machine + Gaussian Naive Bayes
(Decision Tree, Random Forest and AdaBoost are done by the other group member.)
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report,
                             ConfusionMatrixDisplay)

from machine_data import load_data, FEATURES, LABELS

FOCUS = "WARNING"  # the label assigned to this group member
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 1. Data ----------
df = load_data()
print("DATASET:", len(df), "records")
print(df.to_string(index=False))
print("\nClass distribution:\n", df["Condition"].value_counts())

X = df[FEATURES]
y = df["Condition"]

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Temperature_C", y="Vibration_mm_s",
                hue="Condition", hue_order=LABELS, style="Condition", s=100)
plt.title("Temperature vs. vibration by machine condition")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/scatter.png", dpi=150)
plt.show()

# ---------- 2. Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
print(f"\nTraining records: {len(X_train)} | Testing records: {len(X_test)}")

# ---------- 3. Models ----------
models = {
    "SVM": Pipeline([("scaler", StandardScaler()),
                     ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale"))]),
    "Naive Bayes": Pipeline([("scaler", StandardScaler()),
                             ("classifier", GaussianNB())]),
}

# ---------- 4. Train + evaluate ----------
results = []
for name, model in models.items():
    print("\n" + "=" * 60 + f"\n{name.upper()}\n" + "=" * 60)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Multi-class: macro average treats the 3 classes equally.
    results.append({
        "Algorithm": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1-score": f1_score(y_test, y_pred, average="macro", zero_division=0),
    })
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    # My assigned focus class: WARNING (one-vs-rest metrics)
    w_true, w_pred = (y_test == FOCUS), (y_pred == FOCUS)
    print(f"[{FOCUS} focus] precision={precision_score(w_true, w_pred, zero_division=0):.3f} "
          f"recall={recall_score(w_true, w_pred, zero_division=0):.3f} "
          f"f1={f1_score(w_true, w_pred, zero_division=0):.3f}")

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
plt.title("SVM vs. Naive Bayes")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/comparison.png", dpi=150)
plt.show()

# ---------- 6. New machine ----------
new_machine = pd.DataFrame([{"Temperature_C": 89, "Vibration_mm_s": 6.8,
                             "Speed_RPM": 1325, "Operating_Hours": 6500}])
print("\nNEW MACHINE:\n", new_machine)
for name, model in models.items():
    print(f"{name:12s} prediction: {model.predict(new_machine)[0]}")
