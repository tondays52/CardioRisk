"""
CardioRisk AI - Train Heart Sound CNN on CinC 2016
Fixed version: proper labels, resized spectrograms, and class weights.
"""
import os
import numpy as np
import librosa
import cv2
import tensorflow as tf
from tensorflow import keras

layers = keras.layers
models = keras.models
metrics = keras.metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Config
DATA_DIR = r"C:\Users\tonda\Desktop\dataset\audio\cinc2016"
SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 64
FIXED_TIME_STEPS = 157  # Fixed width for resizing
N_FFT = 1024
HOP_LENGTH = 256
BATCH_SIZE = 32
EPOCHS = 15

def load_heart_sounds(data_dir):
    """Load heart sounds and labels from CinC 2016."""
    features = []
    labels = []
    
    for folder in ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']:
        folder_path = os.path.join(data_dir, folder)
        
        if not os.path.exists(folder_path):
            print(f"Skipping {folder} - not found")
            continue
        
        labels_file = os.path.join(folder_path, 'REFERENCE.csv')
        if os.path.exists(labels_file):
            df = pd.read_csv(labels_file, header=None)
            label_map = dict(zip(df[0], df[1]))
        else:
            label_map = {}
            print(f"No REFERENCE.csv in {folder}")
        
        wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        print(f"Loading {len(wav_files)} files from {folder}...")
        
        for file in wav_files:
            file_path = os.path.join(folder_path, file)
            name = file.replace('.wav', '')
            
            try:
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                
                if len(audio) < SAMPLE_RATE * DURATION:
                    audio = np.pad(audio, (0, SAMPLE_RATE * DURATION - len(audio)))
                
                mel = librosa.feature.melspectrogram(
                    y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)
                
                # Resize to fixed shape (N_MELS, FIXED_TIME_STEPS)
                mel_resized = cv2.resize(mel_db, (FIXED_TIME_STEPS, N_MELS))
                
                features.append(mel_resized)
                
                # Label: 1 = normal, -1 = abnormal
                # Convert to binary: 0 = normal, 1 = abnormal
                label_val = label_map.get(name, 1)
                label = 1 if label_val == -1 else 0
                labels.append(label)
                
            except Exception as e:
                continue
    
    features = np.array(features)
    labels = np.array(labels)
    features = features[..., np.newaxis]  # Add channel dimension
    
    return features, labels

def build_model(input_shape):
    """Build 2D CNN for spectrogram classification."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', metrics.AUC(name='auc')]
    )
    return model

if __name__ == "__main__":
    print("=" * 50)
    print("CardioRisk AI - Heart Sound CNN Training")
    print("=" * 50)
    
    print("\nLoading heart sounds...")
    X, y = load_heart_sounds(DATA_DIR)
    
    print(f"\nLoaded {len(X)} samples")
    print(f"Normal (0): {np.sum(y == 0)}, Abnormal (1): {np.sum(y == 1)}")
    
    if len(X) < 10:
        print("ERROR: Not enough data loaded!")
        exit(1)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Input shape: {X.shape[1:]}")
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.array([0, 1]),
        y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"Class weights: {class_weight_dict}")
    
    # Build model
    model = build_model(X.shape[1:])
    model.summary()
    
    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating...")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test AUC: {auc:.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    os.makedirs("models/audio", exist_ok=True)
    history_df.to_csv("models/audio/training_history.csv", index=False)
    print("\n✅ Training history saved to models/audio/training_history.csv")
    
    # Save model
    model.save("models/audio/heart_sound_cnn.h5")
    print("✅ Model saved to models/audio/heart_sound_cnn.h5")