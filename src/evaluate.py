import numpy as np

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
import matplotlib.pyplot as plt


def plot_roc_curve(y_true, y_proba, model_name: str = ""):
    """Draw ROC curve and print AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"{model_name or 'Model'} (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    return roc_auc


def plot_pr_curve(y_true, y_proba, model_name: str = ""):
    """Draw Precision-Recall curve and print Average Precision."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_prec = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(6, 5))
    plt.step(recall, precision, color="darkorange", where="post",
             label=f"{model_name or 'Model'} (AP = {avg_prec:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    return avg_prec


def plot_confusion(y_true, y_pred, normalize: bool = True):
    """Draw confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Fraud"],
    )
    disp.plot(cmap="Blues", values_format=".0f" if not normalize else ".2f")
    plt.title("Confusion Matrix")
    plt.tight_layout()


def print_classification_report(y_true, y_pred):
    """Print full scikit-learn classification report to console."""
    report = classification_report(y_true, y_pred, target_names=["Normal", "Fraud"])
    print(report)
    return report
