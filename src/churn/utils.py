"""Small helpers: ROC/PR/confusion-matrix plotting for the churn pipeline."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


def plot_roc_curve(model, X_test, y_test, output_path: str) -> None:
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_pr_curve(model, X_test, y_test, output_path: str) -> None:
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title("Precision-Recall Curve")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(model, X_test, y_test, output_path: str) -> None:
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Confusion Matrix")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
