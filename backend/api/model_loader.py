"""
CardioRisk AI - Model Loader
Handles loading of TensorFlow models without direct TF import issues.
"""
import os
import numpy as np
import joblib
import pickle
import sys

from typing import Any

# Try importing TensorFlow with error handling
tf: Any = None
keras: Any = None
try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    TF_AVAILABLE = True
    print("[OK] TensorFlow loaded successfully")
except ImportError as e:
    tf = None
    keras = None
    TF_AVAILABLE = False
    print(f"[INFO] TensorFlow not available: {e}")
except Exception as e:
    tf = None
    keras = None
    TF_AVAILABLE = False
    print(f"[INFO] TensorFlow error: {e}")

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(backend_dir)

def load_model_safe(model_path: str):
    """Load a model safely, handling different formats."""
    # Resolve relative path if needed
    if not os.path.isabs(model_path):
        resolved_path = os.path.join(project_root, model_path)
        if not os.path.exists(resolved_path) and os.path.exists(model_path):
            resolved_path = os.path.abspath(model_path)
    else:
        resolved_path = model_path

    if not os.path.exists(resolved_path):
        print(f"[INFO] Model path does not exist: {resolved_path}")
        return None

    if resolved_path.endswith('.keras') or resolved_path.endswith('.h5'):
        if keras is not None:
            try:
                model = keras.models.load_model(resolved_path, compile=False)
                print(f"[OK] Loaded TF model from {resolved_path}")
                return model
            except Exception as e:
                print(f"[INFO] Could not load TF model {resolved_path}: {e}")
                return None
        else:
            print(f"[INFO] TensorFlow / Keras not available, cannot load {resolved_path}")
            return None
    elif resolved_path.endswith('.pkl') or resolved_path.endswith('.joblib'):
        try:
            model = joblib.load(resolved_path)
            print(f"[OK] Loaded model from {resolved_path}")
            return model
        except Exception as e:
            print(f"[INFO] Could not load model {resolved_path}: {e}")
            return None
    else:
        print(f"[INFO] Unknown model format: {resolved_path}")
        return None

# Load models with fallback
def load_models():
    """Load all models with graceful fallback."""
    models = {}
    
    # Tabular model
    tabular_path = os.path.join(project_root, "models", "tabular", "best_tabular_model.pkl")
    scaler_path = os.path.join(project_root, "models", "tabular", "scaler.pkl")
    
    if os.path.exists(tabular_path):
        try:
            models['tabular'] = joblib.load(tabular_path)
            print("[OK] Tabular model loaded")
        except Exception as e:
            print(f"[INFO] Tabular model error: {e}")
            models['tabular'] = None
    else:
        models['tabular'] = None
        print("[INFO] Tabular model not found")
    
    if os.path.exists(scaler_path):
        try:
            models['scaler'] = joblib.load(scaler_path)
            print("[OK] Scaler loaded")
        except Exception as e:
            print(f"[INFO] Scaler error: {e}")
            models['scaler'] = None
    else:
        models['scaler'] = None
        print("[INFO] Scaler not found")
    
    # ECG model (try to load if TF available)
    ecg_path = os.path.join(project_root, "models", "ecg", "ecg_cnn_final.keras")
    models['ecg'] = load_model_safe(ecg_path)
    
    # Heart Sound model
    hs_keras = os.path.join(project_root, "models", "audio", "heart_sound_cnn.keras")
    hs_h5 = os.path.join(project_root, "models", "audio", "heart_sound_cnn.h5")
    hs_path = hs_keras if os.path.exists(hs_keras) else hs_h5
    models['heart_sound'] = load_model_safe(hs_path)

    # PPG Optical 1D-CNN Model
    ppg_path = os.path.join(project_root, "models", "ppg_cnn_model.keras")
    models['ppg'] = load_model_safe(ppg_path)

    # SCG Accelerometer 1D-CNN Model
    scg_path = os.path.join(project_root, "models", "scg_cnn_model.keras")
    models['scg'] = load_model_safe(scg_path)

    # Retinal Fundus U-Net Model
    retinal_path = os.path.join(project_root, "models", "retinal_unet_model.keras")
    models['retinal'] = load_model_safe(retinal_path)
    
    return models