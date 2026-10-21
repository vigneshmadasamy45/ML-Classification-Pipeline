# Task 1: Multi-Class Benchmark Classification Pipeline

**Internship:** Machine Learning & Data Science  
**Dataset:** Iris dataset (included in scikit-learn)

## Objective
Build and compare two supervised multi-class classifiers:
- Logistic Regression
- Decision Tree

The pipeline uses an 80/20 stratified train-test split and evaluates accuracy, macro precision, macro recall, macro F1-score, classification reports, and confusion matrices.

## Project files
- `main.py` — complete runnable Python pipeline
- `Task1_Iris_Classification_Pipeline.ipynb` — guided notebook version
- `requirements.txt` — Python dependencies
- `outputs/` — generated charts, CSV metrics, and classification reports after running

## Setup and run

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies and run:

```bash
python -m pip install -r requirements.txt
python main.py
```

The dataset is loaded directly from scikit-learn; no separate dataset download is required.

## Methodology
1. Load and inspect the Iris dataset.
2. Visualize relationships between features using a pairplot.
3. Split the data into 80% training and 20% testing, preserving class proportions.
4. Train Logistic Regression with `StandardScaler` inside a pipeline to avoid data leakage.
5. Train a Decision Tree without scaling (tree models do not require feature scaling).
6. Evaluate both models using accuracy, macro precision, macro recall, macro F1-score, and confusion matrices.
7. Save a performance comparison chart, classification reports, confusion matrices, pairplot, and two-feature decision-boundary plot.

## Outputs
After `python main.py`, the `outputs/` directory contains:
- `model_comparison.csv`
- `classification_reports.txt`
- `confusion_matrices.png`
- `model_comparison.png`
- `iris_pairplot.png`
- `decision_boundaries.png`

## Notes
- Results are calculated at runtime; do not manually enter or invent scores.
- The decision-boundary chart uses only petal length and petal width for visualization. The main benchmark models use all four features.
- The fixed random seed makes the train/test split reproducible.
