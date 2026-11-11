# Pump Condition Classification

This is a small machine learning project I made for my Software Engineering course. It uses sensor readings from a machine to decide if the machine is running normally or if there is a risk of failure.

## About the data

The dataset is written directly inside the Python file. It has 30 machine records. Each record has four readings:

- Temperature (°C)
- Vibration (mm/s)
- Speed (RPM)
- Operating hours

Each record is labelled either Normal (0) or Failure risk (1). There are 15 records of each kind. Machines that run hotter, vibrate more, run slower and have more operating hours are the ones labelled as failure risk.

## What the code does

1. Builds the dataset and draws a scatter plot and a correlation heatmap.
2. Splits the data into 70% for training and 30% for testing.
3. Trains five classification algorithms:
   - Support Vector Machine (SVM)
   - Naive Bayes
   - Decision Tree
   - Random Forest
   - AdaBoost
4. Tests each model and prints accuracy, precision, recall and F1-score, plus a classification report and a confusion matrix.
5. Compares the five models in one bar chart.
6. Shows which features matter most for the Decision Tree and Random Forest, and draws the Decision Tree.
7. Predicts the condition of one new machine reading.

## How to run

Install the libraries:

```
pip install numpy pandas matplotlib seaborn scikit-learn
```

Then run:

```
python Chapter_3_Examples_all_in_one.py
```

The plots open in separate windows one after another. Close each window to see the next one.

## Note

The dataset is very small, so the test results change a lot with just one record. The scores should only be used to learn how the algorithms work, not as a real diagnosis of a machine.
