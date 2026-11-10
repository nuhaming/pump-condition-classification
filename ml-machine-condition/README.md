# Machine Condition Classification (NORMAL / FAULT)

Supervised ML case study that classifies a machine (pump) as **NORMAL** or
**FAULT** from sensor readings. This is a **two-class (binary) classification**
problem. The earlier intermediate WARNING class has been removed from the
dataset and from all code.

Algorithms:

- **Support Vector Machine (SVM)** and **Gaussian Naive Bayes** (`svm_naive_bayes.py`)
- **Decision Tree**, **Random Forest** and **AdaBoost** (`tree_models.py`)

All scripts use the same dataset in `machine_data.py`, the same
train/test split (70/30, stratified, `random_state=42`) and the same metrics.

## Labels
| Label  | Meaning |
|--------|---------|
| NORMAL | Operating within safe conditions |
| FAULT  | Severe readings, high risk of failure |

## Dataset
24 records (12 NORMAL, 12 FAULT) with four features: `Temperature_C`,
`Vibration_mm_s`, `Speed_RPM`, `Operating_Hours`. The classes are balanced, and
the stratified split keeps 4 NORMAL and 4 FAULT records in the test set.

## Evaluation
- Confusion matrices are 2x2 (NORMAL, FAULT).
- Classification reports list NORMAL and FAULT.
- Accuracy, precision, recall and F1-score in the model comparison are
  macro-averaged over the two classes.
- Per-class precision/recall/F1 for **FAULT** (the positive class) are also
  printed, since missing a fault is the costly error.

## Run
```bash
pip install -r requirements.txt
python svm_naive_bayes.py
python tree_models.py
```
Figures are saved in `figures/`:
`scatter.png`, `confusion_<model>.png` for each of the five models,
`comparison.png` (SVM vs. Naive Bayes) and `comparison_tree_models.png`
(Decision Tree vs. Random Forest vs. AdaBoost).
