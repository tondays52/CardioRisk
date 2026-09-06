import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

DATASET_ROOT = r"C:\Users\tonda\Desktop\dataset"
OUTPUT_DIR = r"C:\Users\tonda\Desktop\hrp\models_scaled"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("1. INGESTING TABULAR DATASETS FROM DIRECTORY")
print("="*60)

def load_and_standardize(filename):
    path = os.path.join(DATASET_ROOT, filename)
    if not os.path.exists(path):
        return None
    
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
    sep = ';' if ';' in first_line else (',' if ',' in first_line else r'\s+')
    
    df = pd.read_csv(path, sep=sep, low_memory=False, nrows=100000)
    df.columns = df.columns.str.strip().str.lower()
    
    col_map = {
        'ap_hi': 'sysbp', 'systolic': 'sysbp', 'sbp': 'sysbp', 'trestbps': 'sysbp',
        'ap_lo': 'diabp', 'diastolic': 'diabp', 'dbp': 'diabp',
        'chol': 'totchol', 'cholesterol': 'totchol',
        'gluc': 'glucose', 'fbs': 'glucose',
        'cardio': 'target', 'tenyearchd': 'target', 'heartdisease': 'target', 
        'output': 'target', 'num': 'target', 'target': 'target',
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
            
    if 'target' in df.columns:
        df['target'] = pd.to_numeric(df['target'], errors='coerce')
        df.dropna(subset=['target'], inplace=True)
        df['target'] = (df['target'] > 0).astype(int)
        print(f"[+] Loaded '{filename}' -> {len(df):,} patient records.")
        return df
    return None

exact_csv_list = [
    "cardiovascular_dataset.csv",
    "CVD_Dataset.csv",
    "cardiovascular_patient_profiles.csv",
    "cardio vascular_dataset.csv",
    "heart.csv",
    "heart_disease_data.csv",
    "data.csv",
    "health_data.csv",
    "healthcare_synthetic_data.csv"
]

dataframes = [df for df in [load_and_standardize(name) for name in exact_csv_list] if df is not None]
combined_df = pd.concat(dataframes, axis=0, ignore_index=True)

# Format numerical fields
for col in ['sysbp', 'diabp', 'totchol', 'glucose', 'age', 'bmi', 'heartrate', 'currentsmoker', 'cigsperday']:
    if col in combined_df.columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

if 'sysbp' in combined_df.columns:
    combined_df.loc[~combined_df['sysbp'].between(70, 260), 'sysbp'] = np.nan
if 'diabp' in combined_df.columns:
    combined_df.loc[~combined_df['diabp'].between(30, 160), 'diabp'] = np.nan

print(f"\n[OK] Unified Training Cohort: {len(combined_df):,} total patients.")

# ==========================================================
# 2. FEATURE ENGINEERING (CLINICAL INTERACTION MARKERS)
# ==========================================================
print("\n" + "="*60)
print("2. COMPUTING CARDIOVASCULAR BIOMARKER INTERACTION RATIOS")
print("="*60)

if 'sysbp' in combined_df.columns and 'diabp' in combined_df.columns:
    combined_df['map'] = (2 * combined_df['diabp'] + combined_df['sysbp']) / 3.0
    combined_df['pulse_pressure'] = combined_df['sysbp'] - combined_df['diabp']
    combined_df['hypertension_stage'] = np.where(
        (combined_df['sysbp'] >= 140) | (combined_df['diabp'] >= 90), 2,
        np.where((combined_df['sysbp'] >= 130) | (combined_df['diabp'] >= 80), 1, 0)
    )

if 'totchol' in combined_df.columns and 'glucose' in combined_df.columns:
    combined_df['chol_glucose_ratio'] = combined_df['totchol'] / (combined_df['glucose'] + 1e-5)

if 'currentsmoker' in combined_df.columns and 'cigsperday' in combined_df.columns:
    combined_df['smoking_intensity'] = combined_df['currentsmoker'].fillna(0) * combined_df['cigsperday'].fillna(0)

print("[OK] Biomarkers: map, pulse_pressure, hypertension_stage, chol_glucose_ratio, smoking_intensity")

# ==========================================================
# 3. IMPUTATION & SPLITTING
# ==========================================================
X_raw = combined_df.drop('target', axis=1)
y = combined_df['target']

X_encoded = pd.get_dummies(X_raw, drop_first=True).dropna(axis=1, how='all')

imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X_encoded), columns=X_encoded.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.15, random_state=42, stratify=y
)

# ==========================================================
# 4. TRAINING COST-SENSITIVE CALIBRATED XGBOOST
# ==========================================================
print("\n" + "="*60)
print("3. TRAINING COST-SENSITIVE CALIBRATED XGBOOST")
print("="*60)

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = float(neg_count / max(1, pos_count))
print(f"Class Balance -> Normal: {neg_count:,} | High Risk: {pos_count:,} (Scale Weight: {scale_pos_weight:.3f})")

xgb_base = XGBClassifier(
    n_estimators=450,
    max_depth=5,
    learning_rate=0.03,
    scale_pos_weight=scale_pos_weight,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)

calibrated_model = CalibratedClassifierCV(estimator=xgb_base, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)

# ==========================================================
# 5. CLINICAL THRESHOLD OPTIMIZATION (YOUDEN'S J)
# ==========================================================
y_probs = calibrated_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

y_preds_optimal = (y_probs >= optimal_threshold).astype(int)

acc = accuracy_score(y_test, y_preds_optimal) * 100
auc = roc_auc_score(y_test, y_probs)
f1 = f1_score(y_test, y_preds_optimal)

print("\n" + "="*60)
print("SCALED TABULAR EVALUATION RESULTS:")
print("="*60)
print(f"Optimal Threshold : {optimal_threshold:.4f}")
print(f"Accuracy          : {acc:.2f}%")
print(f"AUC-ROC           : {auc:.4f}")
print(f"F1-Score          : {f1:.4f}")
print("="*60)
print("\nClassification Report:\n", classification_report(y_test, y_preds_optimal))

save_path = os.path.join(OUTPUT_DIR, "best_tabular_model_scaled.pkl")
joblib.dump(calibrated_model, save_path)
joblib.dump(list(X_imputed.columns), os.path.join(OUTPUT_DIR, "tabular_feature_names.pkl"))
joblib.dump({'optimal_threshold': float(optimal_threshold)}, os.path.join(OUTPUT_DIR, "tabular_threshold.pkl"))
print(f"\nModel exported to: {save_path}")