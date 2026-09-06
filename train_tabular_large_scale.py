import os
import glob
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

print("="*60)
print("1. INGESTING & HARMONIZING ALL LARGE-SCALE TABULAR DATASETS")
print("="*60)

def standardize_schema(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower()
    
    col_map = {
        'ap_hi': 'sysbp', 'systolic': 'sysbp', 'sbp': 'sysbp', 'trestbps': 'sysbp',
        'ap_lo': 'diabp', 'diastolic': 'diabp', 'dbp': 'diabp',
        'chol': 'totchol', 'cholesterol': 'totchol',
        'gluc': 'glucose', 'fbs': 'glucose',
        'cardio': 'target', 'tenyearchd': 'target', 'heartdisease': 'target', 
        'output': 'target', 'num': 'target', 'cad_status': 'target',
        'smoke': 'currentsmoker', 'smoking': 'currentsmoker',
        'cigs': 'cigsperday'
    }
    df.rename(columns=col_map, inplace=True)
    
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)
        
    # Standardize age to years
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        if df['age'].max() > 150:
            df['age'] = (df['age'] / 365.25).astype(int)
            
    df['dataset_source'] = source_name
    return df

dataframes: list[pd.DataFrame] = []

# List all potential candidate files
files_to_load = [
    ("cardiovascular_health_monitoring_dataset.csv", "health_monitoring"),
    ("CVD Dataset.csv", "kaggle_cvd"),
    ("cardiovascular_dataset.csv", "cvd_dataset"),
    ("cardio vascular_dataset.csv", "framingham"),
    ("heart.csv", "uci_heart"),
    ("heart_disease_data.csv", "cleveland")
]

for filename, tag in files_to_load:
    filepath = os.path.join(DATASET_ROOT, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline()
            sep = ';' if ';' in first_line else (',' if ',' in first_line else r'\s+')
            
            # Read first 100k rows if extremely large to prevent OOM
            df = pd.read_csv(filepath, sep=sep, low_memory=False, nrows=100000)
            df = standardize_schema(df, tag)
            
            if 'target' in df.columns:
                print(f"[+] Loaded '{filename}' -> {len(df):,} records.")
                dataframes.append(df)
        except Exception as e:
            print(f"[-] Could not parse {filename}: {e}")

if not dataframes:
    raise RuntimeError("No tabular datasets could be loaded from disk.")

combined_df: pd.DataFrame = pd.concat(dataframes, axis=0, ignore_index=True)

# Coerce target column to strictly binary (0 or 1)
combined_df['target'] = pd.to_numeric(combined_df['target'], errors='coerce')
combined_df = combined_df.dropna(subset=['target'])
combined_df['target'] = (combined_df['target'] > 0).astype(int)

# Numeric conversion for clinical biometric columns
biometric_cols = [
    'sysbp', 'diabp', 'totchol', 'glucose', 'age', 'bmi', 
    'heartrate', 'currentsmoker', 'cigsperday'
]
for col in biometric_cols:
    if col in combined_df.columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

# Filter blood pressure extremes (outlier removal)
if 'sysbp' in combined_df.columns:
    combined_df.loc[~combined_df['sysbp'].between(70, 260), 'sysbp'] = np.nan
if 'diabp' in combined_df.columns:
    combined_df.loc[~combined_df['diabp'].between(30, 160), 'diabp'] = np.nan

print(f"\n[OK] Unified Multi-Source Dataset: {len(combined_df):,} total patients.")

print("\n" + "="*60)
print("2. COMPUTING NON-LINEAR CARDIOVASCULAR INTERACTION FEATURES")
print("="*60)

# 1. Mean Arterial Pressure (MAP)
if 'sysbp' in combined_df.columns and 'diabp' in combined_df.columns:
    combined_df['map'] = (2 * combined_df['diabp'] + combined_df['sysbp']) / 3.0
    # 2. Pulse Pressure (Vascular Stiffness Marker)
    combined_df['pulse_pressure'] = combined_df['sysbp'] - combined_df['diabp']
    # 3. ACC/AHA Hypertension Indicator
    combined_df['hypertension_stage'] = np.where(
        (combined_df['sysbp'] >= 140) | (combined_df['diabp'] >= 90), 2,
        np.where((combined_df['sysbp'] >= 130) | (combined_df['diabp'] >= 80), 1, 0)
    )

# 4. Cholesterol-to-Glucose Risk Index
if 'totchol' in combined_df.columns and 'glucose' in combined_df.columns:
    combined_df['chol_glucose_ratio'] = combined_df['totchol'] / (combined_df['glucose'] + 1e-5)

# 5. Smoking Pack-Years Proxy
if 'currentsmoker' in combined_df.columns and 'cigsperday' in combined_df.columns:
    combined_df['smoking_intensity'] = combined_df['currentsmoker'].fillna(0) * combined_df['cigsperday'].fillna(0)

print("[OK] Engineered Features: map, pulse_pressure, hypertension_stage, chol_glucose_ratio, smoking_intensity")

# ==========================================================
# 3. DATA SPLITTING & IMPUTATION
# ==========================================================
X_raw = combined_df.drop(['target', 'dataset_source'], axis=1, errors='ignore')
y = combined_df['target']

X_encoded = pd.get_dummies(X_raw, drop_first=True)
# Drop completely empty columns
X_encoded = X_encoded.dropna(axis=1, how='all')

# Impute median values across missing multi-dataset fields
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X_encoded), columns=X_encoded.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.15, random_state=42, stratify=y
)

# ==========================================================
# 4. COST-SENSITIVE XGBOOST TRAINING & CALIBRATION
# ==========================================================
print("\n" + "="*60)
print("3. TRAINING COST-SENSITIVE CALIBRATED XGBOOST ENSEMBLE")
print("="*60)

# Compute scale_pos_weight to handle class imbalance
neg_count = int(np.sum(y_train == 0))
pos_count = int(np.sum(y_train == 1))
scale_pos_weight = neg_count / max(1, pos_count)
print(f"Class Distribution: {neg_count:,} Normal vs {pos_count:,} High Risk | scale_pos_weight = {scale_pos_weight:.3f}")

xgb_model = XGBClassifier(
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

# 5-fold cross-validated isotonic probability calibration
calibrated_xgb = CalibratedClassifierCV(estimator=xgb_model, method='isotonic', cv=5)
calibrated_xgb.fit(X_train, y_train)

# ==========================================================
# 5. METRICS & ARTIFACT EXPORT
# ==========================================================
y_probs = calibrated_xgb.predict_proba(X_test)[:, 1]
y_preds = (y_probs >= 0.5).astype(int)

acc = accuracy_score(y_test, y_preds) * 100
auc = roc_auc_score(y_test, y_probs)
f1 = f1_score(y_test, y_preds)

print("\n" + "="*60)
print("FINAL LARGE-SCALE TABULAR PERFORMANCE EVALUATION:")
print("="*60)
print(f"Accuracy  : {acc:.2f}%")
print(f"AUC-ROC   : {auc:.4f}")
print(f"F1-Score  : {f1:.4f}")
print("="*60)
print("\nDetailed Clinical Report:\n", classification_report(y_test, y_preds))

save_path = os.path.join(OUTPUT_DIR, "best_tabular_model_scaled.pkl")
joblib.dump(calibrated_xgb, save_path)
joblib.dump(list(X_imputed.columns), os.path.join(OUTPUT_DIR, "tabular_feature_names.pkl"))
print(f"\n[SUCCESS] Exported scaled tabular model to: {save_path}")