"""
Complete pipeline for credit card fraud detection.
Loads data, preprocesses, loads pre-trained model, evaluates, and saves results.
"""
import logging
import os
import sys

import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data
from preprocess import preprocess
from evaluate import (
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion,
    print_classification_report,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def ensure_data_exists() -> bool:
    """
    Ensure raw data file exists; if not, try to download it.
    Returns True if data is available, False otherwise.
    """
    raw_path = os.path.join("data", "raw", "creditcard.csv")
    if os.path.exists(raw_path):
        logging.info(f"Found raw data: {raw_path}")
        return True

    logging.warning(f"Raw data not found at {raw_path}")
    logging.info("Attempting to download dataset...")
    try:
        from download_data import main as download_main
        download_main()
        if os.path.exists(raw_path):
            logging.info("Download successful.")
            return True
    except Exception as e:
        logging.error(f"Download failed: {e}")

    logging.error(
        "Please manually download the dataset from:\n"
        "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
        f"and place it into {raw_path}"
    )
    return False


def main():
    """
    Execute the full pipeline using a pre-trained model.
    """
    try:
        logging.info("Starting fraud detection pipeline")

        # 1. Check data availability
        if not ensure_data_exists():
            logging.error("Data missing, exiting.")
            sys.exit(1)

        # 2. Load raw data
        logging.info("Loading data...")
        raw_path = os.path.join("data", "raw", "creditcard.csv")
        df = load_data(raw_path)

        # 3. Preprocess (split, scale, balance)
        logging.info("Preprocessing...")
        _, X_test, _, y_test = preprocess(df)
        logging.info(f"Test set: {X_test.shape}")

        # 4. Load pre-trained RandomForest model
        logging.info("Loading pre-trained model...")
        model_path = os.path.join("models", "rf.pkl")
        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}")
            logging.error("Please train the model first by running model.py")
            sys.exit(1)
        selected_model = joblib.load(model_path)
        selected_name = "random_forest"
        logging.info(f"Loaded model from {model_path}")

        # 5. Generate predictions and probabilities
        logging.info("Evaluating model on test set...")
        if hasattr(selected_model, "predict_proba"):
            y_proba = selected_model.predict_proba(X_test)[:, 1]
        else:
            y_proba = selected_model.decision_function(X_test)
        y_pred = selected_model.predict(X_test)

        # 6. Ensure the reports directory exists
        os.makedirs("reports", exist_ok=True)

        # 7. Generate and save ROC curve
        logging.info("Plotting ROC curve...")
        roc_auc = plot_roc_curve(y_test, y_proba, selected_name)
        roc_path = os.path.join("reports", "roc_curve.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()
        logging.info(f"ROC curve saved to {roc_path} (AUC = {roc_auc:.4f})")

        # 8. Generate and save Precision-Recall curve
        logging.info("Plotting Precision-Recall curve...")
        avg_precision = plot_pr_curve(y_test, y_proba, selected_name)
        pr_path = os.path.join("reports", "pr_curve.png")
        plt.savefig(pr_path, dpi=150)
        plt.close()
        logging.info(f"PR curve saved to {pr_path} (AP = {avg_precision:.4f})")

        # 9. Generate and save confusion matrix
        logging.info("Plotting confusion matrix...")
        plot_confusion(y_test, y_pred)
        cm_path = os.path.join("reports", "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        logging.info(f"Confusion matrix saved to {cm_path}")

        # 10. Print and save classification report
        logging.info("Generating classification report...")
        report = print_classification_report(y_test, y_pred)
        report_path = os.path.join("reports", "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        logging.info(f"Classification report saved to {report_path}")

        logging.info("Pipeline finished successfully.")

    except Exception as e:
        logging.exception(f"Pipeline error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
