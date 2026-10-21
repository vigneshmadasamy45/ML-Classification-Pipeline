"""
Task 1: Multi-Class Benchmark Classification Pipeline
Dataset: Iris (built into scikit-learn)

Run:
    python main.py

Outputs are saved in the outputs/ directory.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Allows plots to be saved in headless environments.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def load_data():
    """Load the built-in Iris dataset and return features, labels, and metadata."""
    iris = load_iris(as_frame=True)
    frame = iris.data.copy()
    frame["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return iris, frame, iris.data, pd.Series(iris.target, name="target")


def build_models():
    """Create models; scaling is fit only on training data via Pipeline."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    }


def save_eda_plot(frame):
    """Save a feature pairplot."""
    grid = sns.pairplot(frame, hue="species", diag_kind="hist")
    grid.fig.suptitle("Iris Feature Relationships by Species", y=1.02)
    grid.fig.savefig(OUTPUT_DIR / "iris_pairplot.png", dpi=150, bbox_inches="tight")
    plt.close(grid.fig)


def evaluate_models(models, X_test, y_test, target_names):
    """Evaluate classifiers and save classification reports and confusion matrices."""
    rows = []
    reports = []
    for name, model in models.items():
        predictions = model.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision (macro)": precision_score(y_test, predictions, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_test, predictions, average="macro", zero_division=0),
            "F1-score (macro)": f1_score(y_test, predictions, average="macro", zero_division=0),
        })
        reports.append(f"## {name}\n{classification_report(y_test, predictions, target_names=target_names, zero_division=0)}")
    results = pd.DataFrame(rows).set_index("Model")
    results.to_csv(OUTPUT_DIR / "model_comparison.csv")
    (OUTPUT_DIR / "classification_reports.txt").write_text("\n\n".join(reports), encoding="utf-8")

    fig, axes = plt.subplots(1, len(models), figsize=(12, 4.5))
    if len(models) == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, models.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            model.predict(X_test),
            labels=np.arange(len(target_names)),
            display_labels=target_names,
            cmap="Blues",
            ax=ax,
            colorbar=False,
        )
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    ax = results.plot(kind="bar", figsize=(10, 5), ylim=(0, 1.05))
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="lower right")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return results


def save_decision_boundaries(frame, target_names):
    """Save 2-feature decision boundaries as an explanatory visualization."""
    feature_names = ["petal length (cm)", "petal width (cm)"]
    X_2d = frame[feature_names]
    y_2d = frame["target"].astype(int) if "target" in frame.columns else None
    if y_2d is None:
        iris = load_iris()
        y_2d = pd.Series(iris.target, index=frame.index)

    X_train, _, y_train, _ = train_test_split(
        X_2d, y_2d, test_size=0.20, random_state=RANDOM_STATE, stratify=y_2d
    )
    models_2d = {
        "Logistic Regression (2 features)": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree (2 features)": DecisionTreeClassifier(random_state=RANDOM_STATE),
    }
    x_min, x_max = X_2d.iloc[:, 0].min() - 0.5, X_2d.iloc[:, 0].max() + 0.5
    y_min, y_max = X_2d.iloc[:, 1].min() - 0.5, X_2d.iloc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid = pd.DataFrame({
        feature_names[0]: xx.ravel(),
        feature_names[1]: yy.ravel(),
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (name, model) in zip(axes, models_2d.items()):
        model.fit(X_train, y_train)
        Z = model.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.25, levels=np.arange(-0.5, len(target_names) + 0.5, 1))
        sns.scatterplot(
            x=X_2d[feature_names[0]],
            y=X_2d[feature_names[1]],
            hue=pd.Categorical.from_codes(y_2d, target_names),
            ax=ax,
            s=35,
            edgecolor="black",
            legend="full",
        )
        ax.set_title(name)
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.legend(title="Species", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "decision_boundaries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    iris, frame, X, y = load_data()
    frame["target"] = y.to_numpy()

    print("Dataset shape:", frame.shape)
    print("\nSpecies counts:")
    print(frame["species"].value_counts())
    print("\nMissing values:")
    print(frame.isnull().sum())

    save_eda_plot(frame.drop(columns="target"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTraining samples: {len(X_train)} ({len(X_train) / len(X):.0%})")
    print(f"Testing samples: {len(X_test)} ({len(X_test) / len(X):.0%})")

    models = build_models()
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"Trained: {name}")

    results = evaluate_models(models, X_test, y_test, iris.target_names)
    save_decision_boundaries(frame, iris.target_names)

    print("\nModel comparison:")
    print(results.round(4).to_string())
    best_model = results["F1-score (macro)"].idxmax()
    print(f"\nBest macro F1-score on this test split: {best_model}")
    print(f"\nAll output files saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
