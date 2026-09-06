"""
CardioRisk AI - Tabular Model Training
Trains multiple models and selects the best one.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              roc_curve)
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def load_data():
    X_train = pd.read_parquet("data/processed/X_train.parquet")
    X_test = pd.read_parquet("data/processed/X_test.parquet")
    y_train = pd.read_parquet("data/processed/y_train.parquet")['target']
    y_test = pd.read_parquet("data/processed/y_test.parquet")['target']
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=10, 
                                                 random_state=42, n_jobs=-1),
        "xgboost": xgb.XGBClassifier(n_estimators=200, max_depth=6, 
                                      learning_rate=0.1, random_state=42, 
                                      use_label_encoder=False, eval_metric='logloss'),
        "mlp": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, 
                              random_state=42, early_stopping=True)
    }
    
    trained = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"  Done.")
    
    return trained

def evaluate_models(models, X_test, y_test):
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_prob)
        }
        
        print(f"\n{name}:")
        for metric, value in results[name].items():
            print(f"  {metric}: {value:.4f}")
    
    return results

def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC=0.5)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves - Tabular Models", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("models/tabular/roc_curves.png", dpi=150)
    plt.close()
    print("\nROC curves saved to models/tabular/roc_curves.png")

def save_best_model(models, results):
    os.makedirs("models/tabular", exist_ok=True)
    
    # Select best model by AUC
    best_name = max(results, key=lambda x: results[x]["auc"])
    best_model = models[best_name]
    
    # Save model
    joblib.dump(best_model, "models/tabular/best_model.joblib")
    
    # Save results
    with open("models/tabular/model_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nBest model: {best_name}")
    print(f"AUC: {results[best_name]['auc']:.4f}")
    print("Model saved to models/tabular/best_model.joblib")

def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    print("\nTraining models...")
    models = train_models(X_train, y_train)
    
    print("\nEvaluating models...")
    results = evaluate_models(models, X_test, y_test)
    
    plot_roc_curves(models, X_test, y_test)
    save_best_model(models, results)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()