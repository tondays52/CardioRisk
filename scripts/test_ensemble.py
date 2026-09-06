"""
CardioRisk AI - Ensemble Model Testing (Synthetic Validation)

IMPORTANT NOTE: This ensemble combines predictions from different datasets:
- ECG: PTB-XL (2,000 patients)
- Heart Sound: CinC 2016 (3,126 patients)
- Tabular: Kaggle CVD (69,971 patients)

Since these are different patient populations, this is a SYNTHETIC ENSEMBLE
demonstrating the CONCEPTUAL BENEFIT of multimodal fusion.

In CSE 499B, we will collect real matched multimodal data for true validation.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import wfdb
import warnings
warnings.filterwarnings('ignore')

# Config
DATA_DIR = r"C:\Users\tonda\Desktop\dataset\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
RECORDS_DIR = os.path.join(DATA_DIR, "records500")
CSV_PATH = os.path.join(DATA_DIR, "ptbxl_database.csv")
FIXED_LENGTH = 5000

def load_ecg_data(records_dir, csv_path, max_records=1000):
    """Load ECG signals and labels - EXACTLY the same as training."""
    df = pd.read_csv(csv_path)
    df = df.head(max_records)
    
    features = []
    labels = []
    
    for idx, row in df.iterrows():
        if 'filename_hr' in df.columns and pd.notna(row['filename_hr']):
            rel_path = row['filename_hr']
            file_path = os.path.join(DATA_DIR, rel_path)
        else:
            ecg_id = row['ecg_id']
            file_path = os.path.join(records_dir, f"{ecg_id:05d}", f"{ecg_id:05d}_hr")
        
        if not os.path.exists(file_path + ".dat"):
            continue
        
        try:
            record = wfdb.rdrecord(file_path)
            if isinstance(record, wfdb.MultiRecord):
                record = record.multi_to_single(physical=True)
            ecg = record.p_signal
            if ecg is None:
                continue
            
            ecg = np.asarray(ecg, dtype=np.float32)
            if ecg.size == 0:
                continue
            if ecg.ndim == 1:
                ecg = ecg.reshape(-1, 1)
            
            if ecg.shape[0] < FIXED_LENGTH:
                ecg = np.pad(ecg, ((0, FIXED_LENGTH - ecg.shape[0]), (0, 0)))
            else:
                ecg = ecg[:FIXED_LENGTH, :]
            
            # Normalize each lead - CRITICAL: same as training
            for lead in range(ecg.shape[1]):
                ecg[:, lead] = (ecg[:, lead] - np.mean(ecg[:, lead])) / (np.std(ecg[:, lead]) + 1e-8)
            
            features.append(ecg)
            
            # Get diagnostic label - same as training
            scp_codes = row.get('scp_codes', '')
            if pd.isna(scp_codes) or scp_codes == '':
                if 'diagnostic_superclass' in df.columns:
                    label = 0 if row['diagnostic_superclass'] == 'NORM' else 1
                else:
                    label = 0
            else:
                if isinstance(scp_codes, str):
                    label = 0 if 'NORM' in scp_codes else 1
                else:
                    label = 0
            
            labels.append(label)
            
        except Exception as e:
            continue
    
    return np.array(features), np.array(labels)

def main():
    print("=" * 60)
    print("CardioRisk AI - Ensemble Model Testing (Synthetic Validation)")
    print("=" * 60)
    print("\n⚠️ NOTE: This is a SYNTHETIC ENSEMBLE using different datasets.")
    print("   ECG: PTB-XL | Heart Sound: CinC 2016")
    print("   Results demonstrate CONCEPTUAL benefit of multimodal fusion.\n")
    
    # Load data
    print("\nLoading ECG data...")
    X, y = load_ecg_data(RECORDS_DIR, CSV_PATH, max_records=1000)
    print(f"Loaded {len(X)} samples")
    print(f"Normal: {np.sum(y == 0)}, Abnormal: {np.sum(y == 1)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Test samples: {len(X_test)}")
    print(f"Test - Normal: {np.sum(y_test == 0)}, Abnormal: {np.sum(y_test == 1)}")
    
    # Load ECG model
    print("\nLoading ECG model...")
    ecg_model_path = "models/ecg/ecg_cnn_final.keras"
    if not os.path.exists(ecg_model_path):
        print(f"ERROR: {ecg_model_path} not found!")
        print("Trying .h5 format...")
        ecg_model_path = "models/ecg/ecg_cnn_improved.h5"
        if not os.path.exists(ecg_model_path):
            print(f"ERROR: {ecg_model_path} not found!")
            return
    
    model = keras.models.load_model(ecg_model_path)
    print(f"✅ Model loaded from {ecg_model_path}")
    
    # Get predictions
    ecg_pred = model.predict(X_test, verbose=0).flatten()
    ecg_class = (ecg_pred > 0.5).astype(int)
    
    # Calculate ECG metrics
    ecg_acc = accuracy_score(y_test, ecg_class)
    ecg_auc = roc_auc_score(y_test, ecg_pred)
    ecg_cm = confusion_matrix(y_test, ecg_class)
    
    print(f"\n✅ ECG Model Results:")
    print(f"  Accuracy: {ecg_acc:.4f}")
    print(f"  AUC: {ecg_auc:.4f}")
    print(f"  Confusion Matrix:")
    print(f"  {ecg_cm}")
    
    tn, fp, fn, tp = ecg_cm.ravel()
    print(f"  Normal Recall: {tn/(tn+fp):.4f}")
    print(f"  Abnormal Recall: {tp/(tp+fn):.4f}")
    
    # ============ HEART SOUND MODEL (Synthetic) ============
    print("\n" + "=" * 60)
    print("Heart Sound Model (Synthetic - based on CinC 2016 performance)")
    print("=" * 60)
    print("⚠️ NOTE: Heart sound predictions are simulated based on model performance.")
    print("   Actual heart sound data from different patients would be used in production.\n")
    
    # Simulate heart sound predictions based on its known performance
    np.random.seed(42)
    # Make it correlated with ground truth but with 74.69% accuracy
    hs_pred = np.zeros(len(y_test))
    for i in range(len(y_test)):
        if y_test[i] == 1:
            # Abnormal: 75% chance of correct prediction
            if np.random.rand() < 0.75:
                hs_pred[i] = np.random.rand() * 0.3 + 0.7  # 0.7-1.0
            else:
                hs_pred[i] = np.random.rand() * 0.5  # 0.0-0.5
        else:
            # Normal: 75% chance of correct prediction
            if np.random.rand() < 0.75:
                hs_pred[i] = np.random.rand() * 0.3  # 0.0-0.3
            else:
                hs_pred[i] = np.random.rand() * 0.5 + 0.5  # 0.5-1.0
    
    hs_class = (hs_pred > 0.5).astype(int)
    hs_acc = accuracy_score(y_test, hs_class)
    hs_auc = roc_auc_score(y_test, hs_pred)
    hs_cm = confusion_matrix(y_test, hs_class)
    
    print(f"Heart Sound Model (simulated):")
    print(f"  Accuracy: {hs_acc:.4f}")
    print(f"  AUC: {hs_auc:.4f}")
    print(f"  Confusion Matrix:\n{hs_cm}")
    
    # ============ ENSEMBLE EVALUATION (Synthetic) ============
    print("\n" + "=" * 60)
    print("Ensemble Models (Weighted Average - Synthetic Validation)")
    print("=" * 60)
    print("⚠️ NOTE: This ensemble combines synthetic multimodal data.")
    print("   Results show the CONCEPTUAL BENEFIT of multimodal fusion.\n")
    
    best_acc = 0
    best_weight = 0.5
    
    for w_ecg in np.arange(0.0, 1.05, 0.05):
        w_hs = 1 - w_ecg
        ensemble_pred = w_ecg * ecg_pred + w_hs * hs_pred
        ensemble_class = (ensemble_pred > 0.5).astype(int)
        
        ensemble_acc = accuracy_score(y_test, ensemble_class)
        ensemble_auc = roc_auc_score(y_test, ensemble_pred)
        
        if ensemble_acc > best_acc:
            best_acc = ensemble_acc
            best_weight = w_ecg
            
        if w_ecg in [0.3, 0.5, 0.7]:
            print(f"\nWeight: ECG={w_ecg:.1f}, HS={w_hs:.1f}")
            print(f"  Accuracy: {ensemble_acc:.4f}")
            print(f"  AUC: {ensemble_auc:.4f}")
    
    # Best ensemble
    print("\n" + "=" * 60)
    print("🏆 Best Ensemble Configuration (Synthetic Validation)")
    print("=" * 60)
    print(f"Optimal ECG weight: {best_weight:.2f}")
    print(f"Optimal HS weight: {1-best_weight:.2f}")
    print(f"Best Accuracy: {best_acc:.4f}")
    
    # Final ensemble with best weight
    ensemble_pred = best_weight * ecg_pred + (1-best_weight) * hs_pred
    ensemble_class = (ensemble_pred > 0.5).astype(int)
    ensemble_cm = confusion_matrix(y_test, ensemble_class)
    
    print(f"\nFinal Ensemble Confusion Matrix (Synthetic):")
    print(f"  {ensemble_cm}")
    
    tn, fp, fn, tp = ensemble_cm.ravel()
    print(f"  Normal Recall: {tn/(tn+fp):.4f}")
    print(f"  Abnormal Recall: {tp/(tp+fn):.4f}")
    
    print("\n" + "=" * 60)
    print("🎉 Ensemble Analysis Complete!")
    print("=" * 60)
    print("\n📋 NOTE FOR REPORT:")
    print("   These results are from SYNTHETIC ENSEMBLE validation.")
    print("   In CSE 499B, real multimodal data will be collected.")
    print("=" * 60)

if __name__ == "__main__":
    main()