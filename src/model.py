import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def train_baseline(X_train, y_train):
    """
    Train baseline models with 5-fold stratified cross-validation.
    
    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.array
        Binary labels (0 = normal, 1 = fraud).
        
    Returns
    -------
    dict
        {
            "models": {"logistic_regression": lr_model, ...},
            "cv_results": {"logistic_regression": {...metrics...}, ...}
        }
        Each model entry also has a .fit() call executed (trained on full X_train, y_train).
    """
    os.makedirs("models", exist_ok=True)
    
    # ── Models ────────────────────────────────────────────────────────────────
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "svm_linear": SVC(kernel="linear", probability=True, random_state=42),
    }
    
    # ── Scoring functions ─────────────────────────────────────────────────────
    scorers = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, average="binary", pos_label=1),
        "recall": make_scorer(recall_score, average="binary", pos_label=1),
        "f1": make_scorer(f1_score, average="binary", pos_label=1),
        "roc_auc": make_scorer(roc_auc_score),
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"[{name}] Cross-validating ...")
        
        # 5-fold CV — each fold is fit/evaluated internally
        cv_res = cross_validate(model, X_train, y_train, cv=cv, scoring=scorers, return_train_score=False)
        
        # Aggregate mean ± std for every metric
        metrics = {}
        for sc_name in scorers:
            key = f"test_{sc_name}"
            values = cv_res[key]                    # ndarray shape (5,)
            metrics[sc_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
            bar = "█" * int(metrics[sc_name]["mean"] * 30)
            print(
                f"  {sc_name:12s}  mean={metrics[sc_name]['mean']:.4f}  "
                f"std={metrics[sc_name]['std']:.4f}  [{bar}]"
            )
        
        cv_results[name] = metrics
        
        # Finally, fit on ALL training data so the returned model can predict
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    # ── Extra step: train & save a standalone RandomForest ────────────────────
    rf_path = os.path.join("models", "rf.pkl")
    if "random_forest" not in trained_models:
        # Defensive: even if RF wasn't in `models`, still save one
        rf_final = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_final.fit(X_train, y_train)
    else:
        rf_final = trained_models["random_forest"]
    
    import joblib
    joblib.dump(rf_final, rf_path)
    print(f"\nRandomForest saved → {rf_path}")
    
    return {
        "models": trained_models,
        "cv_results": cv_results,
    }