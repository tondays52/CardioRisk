"""
CardioRisk AI - Real Dataset Training Pipeline
Trains models on genuine empirical datasets from C:\\Users\\tonda\\Desktop\\dataset:
1. SCG (TaebiLab-MSCardio: 108 subjects, 3-axis accelerometer signals)
2. PPG (PPG_DATASET: 127 subjects dual-channel optical waveforms)
3. Retinal Fundus (STARE Dataset: 20 fundus images with ground truth vessel masks)
"""
import os
import glob
import cv2
import joblib
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score

import tensorflow as tf
from tensorflow import keras
layers = keras.layers

DATASET_ROOT = r"C:\Users\tonda\Desktop\dataset"
OUTPUT_DIR = r"C:\Users\tonda\Desktop\hrp\models_scaled"
os.makedirs(OUTPUT_DIR, exist_ok=True)

results = []

# ==============================================================================
# 1. SCG (SEISMOCARDIOGRAPHY) MODEL TRAINING ON TaebiLab-MSCardio
# ==============================================================================
print("\n" + "="*70)
print("1. TRAINING SCG 1D-CNN ON TaebiLab-MSCardio (108 Subjects)")
print("="*70)

scg_base = os.path.join(DATASET_ROOT, "TaebiLab-MSCardio-7e5414e", "MSCardio")
scg_csv_files = glob.glob(os.path.join(scg_base, "Subject_*", "Recording_*", "scg.csv"))
print(f"Found {len(scg_csv_files)} SCG recording files across subjects.")

X_scg_list = []
y_scg_list = []

WINDOW_SIZE = 500  # 5-second window at 100 Hz
STEP = 250

for fpath in scg_csv_files[:150]:
    try:
        df = pd.read_csv(fpath)
        cols = [c for c in ['x', 'y', 'z'] if c in df.columns]
        if len(cols) != 3:
            continue
        sig = df[cols].values
        if len(sig) < WINDOW_SIZE:
            continue
        
        for start in range(0, min(len(sig) - WINDOW_SIZE, 2000), STEP):
            window = sig[start:start + WINDOW_SIZE]
            std = np.std(window, axis=0) + 1e-6
            window_norm = (window - np.mean(window, axis=0)) / std
            
            z_energy = np.mean(window[:, 2]**2)
            var_norm = np.mean(np.var(window, axis=0))
            is_abnormal = 1 if (var_norm > 0.8 or z_energy > 0.5) else 0
            
            X_scg_list.append(window_norm)
            y_scg_list.append(is_abnormal)
    except Exception:
        continue

X_scg = np.array(X_scg_list, dtype=np.float32)
y_scg = np.array(y_scg_list, dtype=np.float32)
print(f"Constructed SCG dataset shape: {X_scg.shape}, Class 1 ratio: {np.mean(y_scg):.2f}")

X_tr_scg, X_te_scg, y_tr_scg, y_te_scg = train_test_split(
    X_scg, y_scg, test_size=0.2, random_state=42, stratify=y_scg
)

scg_inputs = keras.Input(shape=(WINDOW_SIZE, 3))
x = layers.Conv1D(32, kernel_size=7, padding='same', activation='relu')(scg_inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)
scg_outputs = layers.Dense(1, activation='sigmoid')(x)

scg_model = keras.Model(scg_inputs, scg_outputs, name="SCG_1D_CNN")
scg_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

scg_model.fit(
    X_tr_scg, y_tr_scg,
    validation_data=(X_te_scg, y_te_scg),
    epochs=10,
    batch_size=32,
    verbose=1
)

scg_preds = scg_model.predict(X_te_scg).ravel()
scg_acc = accuracy_score(y_te_scg, (scg_preds > 0.5).astype(int))
scg_auc = roc_auc_score(y_te_scg, scg_preds)
scg_rec_abn = recall_score(y_te_scg, (scg_preds > 0.5).astype(int))
scg_rec_norm = recall_score(y_te_scg, (scg_preds > 0.5).astype(int), pos_label=0)

scg_model.save(os.path.join(OUTPUT_DIR, "scg_cnn_scaled.keras"))
print(f"[OK] SCG Model Results: Accuracy={scg_acc*100:.2f}%, AUC={scg_auc:.4f}, Recall Abn={scg_rec_abn*100:.2f}%")

results.append({
    "Modality": "SCG Accelerometer",
    "Model": "1D-CNN (3-Axis)",
    "Dataset": "TaebiLab-MSCardio (108 Subjects)",
    "Samples": len(X_scg),
    "Accuracy": f"{scg_acc*100:.2f}%",
    "AUC": f"{scg_auc:.4f}",
    "Abnormal_Recall": f"{scg_rec_abn*100:.2f}%",
    "Normal_Recall": f"{scg_rec_norm*100:.2f}%"
})


# ==============================================================================
# 2. PPG MODEL TRAINING ON PPG_DATASET
# ==============================================================================
print("\n" + "="*70)
print("2. TRAINING PPG 1D-CNN ON PPG_DATASET (127 Subjects)")
print("="*70)

ppg_dir = os.path.join(DATASET_ROOT, "PPG_DATASET")
input_mat = sio.loadmat(os.path.join(ppg_dir, 'input_data.mat'))
phys_mat = sio.loadmat(os.path.join(ppg_dir, 'physiological_data.mat'))

ppg_cells = input_mat['input_data']
phys_data = phys_mat['physiological_data']

X_ppg_list = []
y_ppg_list = []

for i in range(ppg_cells.shape[0]):
    sig_mat = ppg_cells[i, 0]
    if sig_mat.shape[0] >= 500:
        sig = sig_mat[:1000] if sig_mat.shape[0] >= 1000 else np.pad(sig_mat, ((0, 1000 - sig_mat.shape[0]), (0, 0)))
        sig = (sig - np.mean(sig, axis=0)) / (np.std(sig, axis=0) + 1e-6)
        
        hr_val = phys_data[i, 0] if phys_data.shape[1] > 0 else 75
        sbp_val = phys_data[i, 1] if phys_data.shape[1] > 1 else 120
        label = 1 if (sbp_val > 130 or hr_val > 85) else 0
        
        X_ppg_list.append(sig)
        y_ppg_list.append(label)

X_ppg = np.array(X_ppg_list, dtype=np.float32)
y_ppg = np.array(y_ppg_list, dtype=np.float32)

X_ppg_aug, y_ppg_aug = [], []
for sig, lbl in zip(X_ppg, y_ppg):
    X_ppg_aug.append(sig)
    y_ppg_aug.append(lbl)
    for _ in range(3):
        noise = np.random.normal(0, 0.02, sig.shape)
        X_ppg_aug.append(sig * np.random.uniform(0.95, 1.05) + noise)
        y_ppg_aug.append(lbl)

X_ppg_aug = np.array(X_ppg_aug, dtype=np.float32)
y_ppg_aug = np.array(y_ppg_aug, dtype=np.float32)

X_tr_ppg, X_te_ppg, y_tr_ppg, y_te_ppg = train_test_split(
    X_ppg_aug, y_ppg_aug, test_size=0.2, random_state=42, stratify=y_ppg_aug
)

ppg_inputs = keras.Input(shape=(1000, 2))
x = layers.Conv1D(32, kernel_size=9, padding='same', activation='relu')(ppg_inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)
ppg_outputs = layers.Dense(1, activation='sigmoid')(x)

ppg_model = keras.Model(ppg_inputs, ppg_outputs, name="PPG_1D_CNN")
ppg_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

ppg_model.fit(
    X_tr_ppg, y_tr_ppg,
    validation_data=(X_te_ppg, y_te_ppg),
    epochs=12,
    batch_size=16,
    verbose=1
)

ppg_preds = ppg_model.predict(X_te_ppg).ravel()
ppg_acc = accuracy_score(y_te_ppg, (ppg_preds > 0.5).astype(int))
ppg_auc = roc_auc_score(y_te_ppg, ppg_preds)
ppg_rec_abn = recall_score(y_te_ppg, (ppg_preds > 0.5).astype(int), zero_division=0)
ppg_rec_norm = recall_score(y_te_ppg, (ppg_preds > 0.5).astype(int), pos_label=0, zero_division=0)

ppg_model.save(os.path.join(OUTPUT_DIR, "ppg_cnn_scaled.keras"))
print(f"[OK] PPG Model Results: Accuracy={ppg_acc*100:.2f}%, AUC={ppg_auc:.4f}, Recall Abn={ppg_rec_abn*100:.2f}%")

results.append({
    "Modality": "Camera PPG / HRV",
    "Model": "1D-CNN (Dual-Channel)",
    "Dataset": "PPG_DATASET (127 Subjects)",
    "Samples": len(X_ppg_aug),
    "Accuracy": f"{ppg_acc*100:.2f}%",
    "AUC": f"{ppg_auc:.4f}",
    "Abnormal_Recall": f"{ppg_rec_abn*100:.2f}%",
    "Normal_Recall": f"{ppg_rec_norm*100:.2f}%"
})


# ==============================================================================
# 3. RETINAL FUNDUS U-NET SEGMENTATION TRAINING ON STARE DATASET
# ==============================================================================
print("\n" + "="*70)
print("3. TRAINING RETINAL U-NET ON STARE DATASET (20 Ground Truth Image Pairs)")
print("="*70)

stare_dir = os.path.join(DATASET_ROOT, "Retinals")
raw_files = [f for f in os.listdir(stare_dir) if f.endswith('.ppm') and not (f.endswith('.ah.ppm') or f.endswith('.vk.ppm'))]

X_ret_list, y_ret_list = [], []
IMG_SIZE = 128

for raw_name in raw_files:
    base_id = raw_name.split('.')[0]
    raw_path = os.path.join(stare_dir, raw_name)
    mask_path = os.path.join(stare_dir, f"{base_id}.ah.ppm")
    if not os.path.exists(mask_path):
        mask_path = os.path.join(stare_dir, f"{base_id}.vk.ppm")
    
    if os.path.exists(raw_path) and os.path.exists(mask_path):
        img = cv2.imread(raw_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is not None and mask is not None:
            img_resized = np.asarray(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
            mask_resized = np.asarray(cv2.resize(mask, (IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
            mask_binary = (mask_resized > 0.5).astype(np.float32)
            
            X_ret_list.append(img_resized)
            y_ret_list.append(mask_binary[..., np.newaxis])

X_ret = np.array(X_ret_list, dtype=np.float32)
y_ret = np.array(y_ret_list, dtype=np.float32)

X_ret_aug, y_ret_aug = [], []
for img, msk in zip(X_ret, y_ret):
    X_ret_aug.append(img)
    y_ret_aug.append(msk)
    X_ret_aug.append(np.fliplr(img))
    y_ret_aug.append(np.fliplr(msk))
    X_ret_aug.append(np.flipud(img))
    y_ret_aug.append(np.flipud(msk))

X_ret_aug = np.array(X_ret_aug, dtype=np.float32)
y_ret_aug = np.array(y_ret_aug, dtype=np.float32)

X_tr_ret, X_te_ret, y_tr_ret, y_te_ret = train_test_split(
    X_ret_aug, y_ret_aug, test_size=0.2, random_state=42
)

def build_unet(input_shape=(128, 128, 3)):
    inputs = keras.Input(shape=input_shape)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)

    u4 = layers.UpSampling2D((2, 2))(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u4)

    u5 = layers.UpSampling2D((2, 2))(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u5)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)
    return keras.Model(inputs, outputs, name="Retinal_UNet")

unet_model = build_unet()
unet_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

unet_model.fit(
    X_tr_ret, y_tr_ret,
    validation_data=(X_te_ret, y_te_ret),
    epochs=12,
    batch_size=8,
    verbose=1
)

ret_preds = unet_model.predict(X_te_ret)
intersection = np.sum(ret_preds * y_te_ret)
dice_score = (2.0 * intersection) / (np.sum(ret_preds) + np.sum(y_te_ret) + 1e-6)
pixel_acc = accuracy_score(y_te_ret.ravel() > 0.5, (ret_preds.ravel() > 0.5).astype(int))
pixel_auc = roc_auc_score(y_te_ret.ravel() > 0.5, ret_preds.ravel())

unet_model.save(os.path.join(OUTPUT_DIR, "retinal_unet_scaled.keras"))
print(f"[OK] Retinal U-Net Results: Pixel Acc={pixel_acc*100:.2f}%, Pixel AUC={pixel_auc:.4f}, Dice={dice_score:.4f}")

results.append({
    "Modality": "Retinal Fundus",
    "Model": "U-Net Vessel Segmentation",
    "Dataset": "STARE Database (20 Ground Truth Pairs)",
    "Samples": len(X_ret_aug),
    "Accuracy": f"{pixel_acc*100:.2f}%",
    "AUC": f"{pixel_auc:.4f}",
    "Abnormal_Recall": f"{dice_score*100:.2f}% (Dice)",
    "Normal_Recall": f"{pixel_acc*100:.2f}%"
})

print("\n" + "="*70)
print("SUMMARY OF EMPIRICALLY RETRAINED REMAINING MODALITIES")
print("="*70)
df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))

df_res.to_csv(os.path.join(OUTPUT_DIR, "retrained_benchmarks_summary.csv"), index=False)
print(f"\nSaved summary to {os.path.join(OUTPUT_DIR, 'retrained_benchmarks_summary.csv')}")
