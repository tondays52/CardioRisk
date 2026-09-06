"""
CardioRisk AI - Train ECG CNN on PTB-XL (FIXED with more data)
Trains a 1D CNN on 12-lead ECG signals with improved performance.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import wfdb
import warnings
warnings.filterwarnings('ignore')

# Config
DATA_DIR = r"C:\Users\tonda\Desktop\dataset\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
RECORDS_DIR = os.path.join(DATA_DIR, "records500")
CSV_PATH = os.path.join(DATA_DIR, "ptbxl_database.csv")
FIXED_LENGTH = 5000  # 10 seconds at 500 Hz
BATCH_SIZE = 32
EPOCHS = 30

def load_ecg_data(records_dir, csv_path, max_records=2000):
    """Load ECG signals and labels."""
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
            if ecg.shape[0] < FIXED_LENGTH:
                ecg = np.pad(ecg, ((0, FIXED_LENGTH - ecg.shape[0]), (0, 0)))
            else:
                ecg = ecg[:FIXED_LENGTH, :]
            
            # Normalize each lead
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
    
    X = np.array(features)
    y = np.array(labels)
    
    print(f"Loaded {len(X)} samples")
    print(f"Class distribution: Normal={np.sum(y==0)}, Abnormal={np.sum(y==1)}")
    
    return X, y

def build_improved_model(input_shape):
    """Build improved 1D CNN for ECG classification."""
    inputs = keras.Input(shape=input_shape)
    
    # First convolutional block
    x = keras.layers.Conv1D(64, kernel_size=7, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling1D(2)(x)
    
    # Second block
    x = keras.layers.Conv1D(128, kernel_size=5, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling1D(2)(x)
    
    # Third block
    x = keras.layers.Conv1D(256, kernel_size=5, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling1D(2)(x)
    
    # Fourth block - deeper
    x = keras.layers.Conv1D(512, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    
    # Dense layers with dropout
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Use FIXED learning rate - ReduceLROnPlateau will handle scheduling
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model

if __name__ == "__main__":
    print("=" * 60)
    print("CardioRisk AI - ECG CNN Training (FIXED with 2000 samples)")
    print("=" * 60)
    
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("\nLoading ECG data...")
    X, y = load_ecg_data(RECORDS_DIR, CSV_PATH, max_records=2000)
    
    if len(X) < 10:
        print("ERROR: Not enough data!")
        exit(1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    print(f"Input shape: {X.shape[1:]}")
    
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: float(class_weights[0]), 1: float(class_weights[1])}
    print(f"\nClass weights: {class_weight_dict}")
    
    model = build_improved_model(X.shape[1:])
    print("\nModel Architecture:")
    model.summary()
    
    os.makedirs("models/ecg", exist_ok=True)
    
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=10,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'models/ecg/best_ecg_model.keras',
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("\n" + "=" * 60)
    print("Training Enhanced ECG CNN Model...")
    print("=" * 60)
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "=" * 60)
    print("Evaluating Model...")
    print("=" * 60)
    
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    print(f"\n{'='*30} RESULTS {'='*30}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test AUC: {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"               Predicted")
    print(f"               Normal  Abnormal")
    print(f"Actual Normal   {tn:3d}     {fp:3d}")
    print(f"       Abnormal {fn:3d}     {tp:3d}")
    print(f"\nDetailed Metrics:")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))
    
    model.save("models/ecg/ecg_cnn_final.keras")
    print("\n✅ Model saved to models/ecg/ecg_cnn_final.keras")
    
    history_df = pd.DataFrame(history.history)
    history_df.to_csv("models/ecg/training_history_final.csv", index=False)
    print("✅ Training history saved to models/ecg/training_history_final.csv")
    
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print("=" * 60)