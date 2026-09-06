"""
CardioRisk AI - SHAP Explainability for Tabular Model
"""
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

def load_data_and_model():
    X_train = pd.read_parquet("data/processed/X_train.parquet")
    model = joblib.load("models/tabular/best_model.joblib")
    return X_train, model

def generate_shap_explanations():
    X_train, model = load_data_and_model()
    
    # Use a smaller sample for speed
    X_sample = X_train.sample(500, random_state=42)
    
    # Create SHAP explainer for tree-based model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # shap_values shape: (n_samples, n_features, n_classes)
    # Take positive class (index 1)
    shap_positive = shap_values[:, :, 1]
    
    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_positive, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("models/tabular/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP summary plot saved.")
    
    # Feature importance bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_positive, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("models/tabular/shap_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP importance plot saved.")
    
    # Save mean absolute SHAP values
    mean_abs = np.abs(shap_positive).mean(axis=0)
    
    feature_importance = pd.DataFrame({
        'feature': X_sample.columns,
        'mean_abs_shap': mean_abs
    }).sort_values('mean_abs_shap', ascending=False)
    
    feature_importance.to_csv("models/tabular/feature_importance.csv", index=False)
    print("\nTop features by SHAP importance:")
    print(feature_importance.head(10).to_string(index=False))

if __name__ == "__main__":
    generate_shap_explanations()