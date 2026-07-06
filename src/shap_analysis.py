import os
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

import shap
import matplotlib.pyplot as plt
plt.switch_backend('Agg')


def main():
    # 1. Load data and model
    print('[1/5] Loading data and preprocessor...')
    from src.data_loader import load_data
    from src.preprocess import preprocess
    
    df = load_data('data/raw/creditcard.csv')
    X_train_res, X_test, y_train_res, y_test = preprocess(df)
    
    rf_path = os.path.join('models', 'rf.pkl')
    rf_model = joblib.load(rf_path)
    print(f'[OK] Loaded RandomForest model: {type(rf_model).__name__}')
    
    # 2. Predict on test set
    print('\n[2/5] Making predictions...')
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    print('\nConfusion Matrix:')
    print(cm)
    
    # Find misclassified fraud samples (actual=1, pred=0)
    misclassified_fraud_idx = np.where((y_test == 1) & (y_pred == 0))[0]
    print(f'\nFound {len(misclassified_fraud_idx)} misclassified fraud samples')
    
    # 3. Compute TreeExplainer SHAP values for first 200 test samples
    print('\n[3/5] Computing SHAP values (TreeExplainer)...')
    feature_names = X_test.columns.tolist()
    
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test.iloc[:200])
    print(f'[OK] SHAP values computed: shape = {np.array(shap_values).shape}')
    
    # For binary classification, shap_values shape is (n_samples, n_features, 2)
    # Use class 1 (fraud) SHAP values for analysis
    shap_values_class1 = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values
    
    # 4. Summary plot (beeswarm)
    print('\n[4/5] Generating summary plot...')
    shap.summary_plot(
        shap_values_class1,
        X_test.iloc[:200],
        feature_names=feature_names,
        plot_type='beeswarm',
        show=False
    )
    reports_dir = os.path.join('reports', 'shap')
    os.makedirs(reports_dir, exist_ok=True)
    summary_path = os.path.join(reports_dir, 'summary_beeswarm.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[OK] Summary plot saved to {summary_path}')
    
    # 5. Force plot for a misclassified fraud sample
    if len(misclassified_fraud_idx) > 0:
        sample_idx = misclassified_fraud_idx[0]
        print(f'\n[5/5] Analyzing misclassified fraud sample at index {sample_idx}')
        
        sample_data = X_test.iloc[sample_idx]
        true_label = y_test.iloc[sample_idx]
        pred_label = y_pred[sample_idx]
        pred_proba = y_proba[sample_idx]
        
        print(f'True label: {true_label} (Fraud), Predicted label: {pred_label} (Normal)')
        print(f'Prediction probability for fraud: {pred_proba:.4f}')
        
        print('\nClassification report:')
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
        
        # Get SHAP values for this specific sample (need to compute if outside first 200)
        if sample_idx < 200:
            sample_shap = shap_values_class1[sample_idx]
        else:
            # Compute SHAP for this specific sample
            sample_shap_full = explainer.shap_values(X_test.iloc[sample_idx:sample_idx+1])
            sample_shap = sample_shap_full[:, :, 1].flatten() if len(np.array(sample_shap_full).shape) == 3 else sample_shap_full.flatten()
        
        # Get top-3 features by absolute SHAP value
        abs_shap = np.abs(sample_shap)
        top3_indices = np.argsort(abs_shap)[-3:][::-1]
        top3_features = [feature_names[i] for i in top3_indices]
        top3_shap_values = [sample_shap[i] for i in top3_indices]
        top3_feature_values = [sample_data.iloc[i] for i in top3_indices]
        
        print(f'\nTop 3 features for this misclassified sample:')
        for i, (feat, shap_val, feat_val) in zip(top3_indices, zip(top3_features, top3_shap_values, top3_feature_values)):
            direction = 'pushes toward fraud' if shap_val > 0 else 'pushes toward normal'
            print(f'  {feat}: SHAP={shap_val:.4f} ({direction}), value={feat_val:.4f}')
        
        # Force plot using shap.plots.force (v0.20+ API)
        print('\nGenerating force plot...')
        base_value = explainer.expected_value[1] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value
        
        fig = shap.plots.force(
            base_value,
            sample_shap,
            sample_data,
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        
        force_path = os.path.join(reports_dir, f'force_misclassified_{sample_idx}.png')
        plt.gcf().set_size_inches(12, 4)
        plt.savefig(force_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'[OK] Force plot saved to {force_path}')
        
        # Also create a bar plot for top-3 features
        print('\nGenerating top-3 feature bar plot...')
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#ff6b6b' if v > 0 else '#4ecdc4' for v in top3_shap_values]
        bars = ax.barh(top3_features, top3_shap_values, color=colors)
        ax.set_xlabel('SHAP Value')
        ax.set_title(f'Top 3 Features for Misclassified Fraud Sample #{sample_idx}')
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, top3_shap_values):
            ax.text(bar.get_width() + 0.01 if val > 0 else bar.get_width() - 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}',
                    ha='left' if val > 0 else 'right',
                    va='center')
        
        plt.tight_layout()
        bar_path = os.path.join(reports_dir, f'top3_bar_{sample_idx}.png')
        plt.savefig(bar_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'[OK] Top-3 bar plot saved to {bar_path}')
    else:
        print('\n[SKIP] No misclassified fraud samples found.')
    
    print('\n' + '='*60)
    print('SHAP analysis completed successfully!')
    print('='*60)


if __name__ == '__main__':
    main()
