"""
CardioRisk AI - ECG Model Diagnosis
Checks if the ECG model is loading correctly
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import wfdb
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r"C:\Users\tonda\Desktop\dataset\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
RECORDS_DIR = os.path.join(DATA_DIR, "records500")
CSV_PATH = os.path.join(DATA_DIR, "ptbxl_database.csv")
FIXED_LENGTH = 5000

def load_ecg_data(records_dir, csv_path, max_records=100):
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
            
            for lead in range(ecg.shape[1]):
                ecg[:, lead] = (ecg[:, lead] - np.mean(ecg[:, lead])) / (np.std(ecg[:, lead]) + 1e-8)
            
            features.append(ecg)
            
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

def diagnose_model():
    print("=" * 60)
    print("ECG Model Diagnosis")
    print("=" * 60)
    
    # Check if model exists
    model_path = "models/ecg/ecg_cnn_improved.keras"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        # Try .h5
        model_path = "models/ecg/ecg_cnn_improved.h5"
        if os.path.exists(model_path):
            print(f"✅ Found model at {model_path}")
        else:
            print("❌ No model found!")
            return
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("✅ Model loaded")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    
    # Load sample data
    print("\nLoading sample ECG data...")
    X, y = load_ecg_data(RECORDS_DIR, CSV_PATH, max_records=100)
    print(f"Loaded {len(X)} samples")
    
    if len(X) == 0:
        print("❌ No data loaded!")
        return
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Test samples: {len(X_test)}")
    print(f"Test labels - Normal: {np.sum(y_test == 0)}, Abnormal: {np.sum(y_test == 1)}")
    
    # Get predictions
    print("\nGetting predictions...")
    pred_proba = model.predict(X_test, verbose=0).flatten()
    pred_class = (pred_proba > 0.5).astype(int)
    
    print(f"\nPredictions distribution:")
    print(f"  Predicted Normal: {np.sum(pred_class == 0)}")
    print(f"  Predicted Abnormal: {np.sum(pred_class == 1)}")
    print(f"  Prediction probabilities - Min: {np.min(pred_proba):.4f}, Max: {np.max(pred_proba):.4f}, Mean: {np.mean(pred_proba):.4f}")
    
    print(f"\nActual labels distribution:")
    print(f"  Actual Normal: {np.sum(y_test == 0)}")
    print(f"  Actual Abnormal: {np.sum(y_test == 1)}")
    
    # Check if all predictions are the same
    if len(np.unique(pred_class)) == 1:
        print("\n❌ PROBLEM: All predictions are the same class!")
        print("This indicates a model loading or preprocessing issue.")
        
        # Check model weights
        print("\nChecking model layers...")
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'weights'):
                weights = layer.get_weights()
                if len(weights) > 0:
                    print(f"  Layer {i}: {layer.name} - weights shape: {weights[0].shape}")
    
    print("\n" + "=" * 60)
    print("Diagnosis Complete")
    print("=" * 60)

if __name__ == "__main__":
    diagnose_model()