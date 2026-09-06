"""
CardioRisk AI - Train Tabular Models on CVD Dataset
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_tabular_data():
    """Load Kaggle CVD dataset."""
    csv_path = r"C:\Users\tonda\Desktop\hrp\data\raw\tabular\cardio_train.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return None, None
    
    df = pd.read_csv(csv_path, sep=';')
    print(f"✅ Loaded {len(df)} samples from {os.path.basename(csv_path)}")
    
    # Handle invalid values first
    df = df[(df['ap_hi'] > 0) & (df['ap_lo'] > 0)]
    df = df[(df['height'] > 0) & (df['weight'] > 0)]
    
    # Feature engineering with safety checks
    df['age_years'] = df['age'] / 365
    df['bmi'] = df['weight'] / ((df['height']/100) ** 2)
    
    # Avoid division by zero
    df['bp_ratio'] = df['ap_hi'] / df['ap_lo'].replace(0, np.nan)
    df['bp_ratio'] = df['bp_ratio'].fillna(df['bp_ratio'].median())
    
    df['map'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3
    
    # Handle outliers
    for col in ['ap_hi', 'ap_lo', 'height', 'weight', 'bmi']:
        q99 = df[col].quantile(0.99)
        q01 = df[col].quantile(0.01)
        df[col] = df[col].clip(q01, q99)
    
    # Remove NaN and infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    features = ['age_years', 'gender', 'height', 'weight', 
                'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 
                'smoke', 'alco', 'active', 'bmi', 'bp_ratio']
    
    X = df[features]
    y = df['cardio']
    
    print(f"Features: {len(features)}")
    print(f"Samples after cleaning: {len(df)}")
    print(f"Class distribution: Normal={sum(y==0)}, Abnormal={sum(y==1)}")
    
    return X, y

def train_tabular_models():
    print("=" * 60)
    print("CardioRisk AI - Tabular Model Training")
    print("=" * 60)
    
    X, y = load_tabular_data()
    if X is None:
        return
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_model = None
    best_auc = 0
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_proba = np.asarray(model.predict_proba(X_test_scaled))[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {'accuracy': acc, 'auc': auc}
        print(f"  Accuracy: {acc:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        
        if auc > best_auc:
            best_auc = auc
            best_model = (name, model)
    
    print("\n" + "=" * 60)
    print("📊 Results Summary")
    print("=" * 60)
    for name, metrics in results.items():
        print(f"{name:25} Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
    
    if best_model is None:
        print("\n❌ No model was successfully trained.")
        return

    print(f"\n🏆 Best Model: {best_model[0]} (AUC: {best_auc:.4f})")
    
    os.makedirs("models/tabular", exist_ok=True)
    joblib.dump(best_model[1], "models/tabular/best_tabular_model.pkl")
    joblib.dump(scaler, "models/tabular/scaler.pkl")
    
    results_df = pd.DataFrame(results).T
    results_df.to_csv("models/tabular/results.csv")
    print("\n✅ Models saved to models/tabular/")

if __name__ == "__main__":
    train_tabular_models()