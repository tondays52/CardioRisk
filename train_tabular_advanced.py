import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

DATASET_ROOT = r"C:\Users\tonda\Desktop\dataset"
OUTPUT_DIR = r"C:\Users\tonda\Desktop\hrp\models_scaled"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=== 1. Ingesting and Harmonizing Multi-Source Tabular Datasets ===")

def load_and_standardize(csv_name):
    path = os.path.join(DATASET_ROOT, csv_name)
    if not os.path.exists(path):
        return None
    
    with open(path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    sep = ';' if ';' in first_line else (',' if ',' in first_line else r'\s+')
    df = pd.read_csv(path, sep=sep)
    df.columns = df.columns.str.strip().str.lower()
    
    col_map = {
        'ap_hi': 'sysbp', 'systolic': 'sysbp', 'sbp': 'sysbp', 'trestbps': 'sysbp',
        'ap_lo': 'diabp', 'diastolic': 'diabp', 'dbp': 'diabp',
        'chol': 'totchol', 'cholesterol': 'totchol',
        'gluc': 'glucose', 'fbs': 'glucose',
        'cardio': 'target', 'tenyearchd': 'target', 'heartdisease': 'target', 'output': 'target', 'num': 'target',
        'smoke': 'currentsmoker', 'smoking': 'currentsmoker',
        'cigs': 'cigsperday'
    }
    df.rename(columns=col_map, inplace=True)
    
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)
        
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        if df['age'].max() > 150:
            df['age'] = (df['age'] / 365.25).astype(int)
            
    return df

csv_candidates = [
    "CVD Dataset.csv", "cardiovascular_dataset.csv", "cardio vascular_dataset.csv",
    "heart.csv", "heart_disease_data.csv", "cardiovascular_health_monitoring_dataset.csv"
]

df_list: list[pd.DataFrame] = []
for name in csv_candidates:
    d = load_and_standardize(name)
    if d is not None and 'target' in d.columns:
        print(f"Loaded '{name}' with {len(d):,} records.")
        df_list.append(d)

combined_df: pd.DataFrame = pd.concat(df_list, axis=0, ignore_index=True)
combined_df.dropna(subset=['target'], inplace=True)

# Format target and numerical features
combined_df['target'] = pd.to_numeric(combined_df['target'], errors='coerce')
combined_df.dropna(subset=['target'], inplace=True)
combined_df['target'] = (combined_df['target'] > 0).astype(int)

for col in ['sysbp', 'diabp', 'currentsmoker', 'cigsperday', 'totchol', 'glucose', 'age', 'bmi', 'heartrate']:
    if col in combined_df.columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

if 'sysbp' in combined_df.columns:
    combined_df.loc[~combined_df['sysbp'].between(70, 260), 'sysbp'] = np.nan
if 'diabp' in combined_df.columns:
    combined_df.loc[~combined_df['diabp'].between(30, 160), 'diabp'] = np.nan

print(f"Total Unified Patient Records: {len(combined_df):,}")

# ==========================================================
# 2. CARDIOVASCULAR FEATURE ENGINEERING
# ==========================================================
print("=== 2. Generating Cardiovascular Interaction Markers ===")

if 'sysbp' in combined_df.columns and 'diabp' in combined_df.columns:
    combined_df['map'] = (2 * combined_df['diabp'] + combined_df['sysbp']) / 3.0
    combined_df['pulse_pressure'] = combined_df['sysbp'] - combined_df['diabp']
    combined_df['hypertension_stage'] = np.where(
        (combined_df['sysbp'] >= 140) | (combined_df['diabp'] >= 90), 2,
        np.where((combined_df['sysbp'] >= 130) | (combined_df['diabp'] >= 80), 1, 0)
    )

if 'currentsmoker' in combined_df.columns and 'cigsperday' in combined_df.columns:
    combined_df['smoking_intensity'] = combined_df['currentsmoker'].fillna(0) * combined_df['cigsperday'].fillna(0)

X_raw = combined_df.drop('target', axis=1)
y = combined_df['target']

X_encoded = pd.get_dummies(X_raw, drop_first=True)

# Drop any column that contains only NaN values
X_encoded = X_encoded.dropna(axis=1, how='all')

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_imputed_array = imputer.fit_transform(X_encoded)
X_imputed = pd.DataFrame(X_imputed_array, columns=X_encoded.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.15, random_state=42, stratify=y
)

# ==========================================================
# 3. TRAINING CALIBRATED XGBOOST CLASSIFIER
# ==========================================================
print("=== 3. Training Cost-Sensitive Calibrated XGBoost ===")
neg_pos_ratio = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))

xgb_base = XGBClassifier(
    n_estimators=350,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=neg_pos_ratio,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

calibrated_model = CalibratedClassifierCV(estimator=xgb_base, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)

# ==========================================================
# 4. EVALUATION & EXPORT
# ==========================================================
y_probs = calibrated_model.predict_proba(X_test)[:, 1]
y_preds = (y_probs >= 0.5).astype(int)

acc = accuracy_score(y_test, y_preds) * 100
auc = roc_auc_score(y_test, y_probs)
f1 = f1_score(y_test, y_preds)

print("\n" + "="*50)
print("Optimized Multi-Source Tabular Model Evaluation:")
print(f"Accuracy : {acc:.2f}%")
print(f"AUC-ROC  : {auc:.4f}")
print(f"F1-Score : {f1:.4f}")
print("="*50)
print("\nClassification Report:\n", classification_report(y_test, y_preds))

save_path = os.path.join(OUTPUT_DIR, "best_tabular_model_scaled.pkl")
joblib.dump(calibrated_model, save_path)
joblib.dump(list(X_imputed.columns), os.path.join(OUTPUT_DIR, "tabular_feature_names.pkl"))
print(f"\nModel exported to: {save_path}")