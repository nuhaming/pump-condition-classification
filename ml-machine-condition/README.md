# Machine Condition Classification (NORMAL / WARNING / FAULT)

Supervised ML case study. This part covers:

- **Support Vector Machine (SVM)**
- **Gaussian Naive Bayes**

Decision Tree, Random Forest and AdaBoost are done by the other group member,
using the same dataset in `machine_data.py`.

Focus class for this part: **WARNING** (per-class metrics are printed for it).

## Labels
| Label   | Meaning |
|---------|---------|
| NORMAL  | Operating within safe conditions |
| WARNING | Elevated readings, may need inspection |
| FAULT   | Severe readings, high risk of failure |

## Run
```bash
pip install -r requirements.txt
python svm_naive_bayes.py
```
Figures are saved in `figures/`.
