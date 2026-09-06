import os
import glob
import numpy as np
import pandas as pd
import librosa
import joblib
import wfdb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
import tensorflow as tf
from tensorflow import keras

layers = keras.layers
Conv1D = layers.Conv1D
Conv2D = layers.Conv2D
BatchNormalization = layers.BatchNormalization
Activation = layers.Activation
MaxPooling1D = layers.MaxPooling1D
MaxPooling2D = layers.MaxPooling2D
GlobalAveragePooling1D = layers.GlobalAveragePooling1D
GlobalAveragePooling2D = layers.GlobalAveragePooling2D
Dense = layers.Dense
Dropout = layers.Dropout


DATASET_ROOT = r"C:\Users\tonda\Desktop\dataset"
OUTPUT_DIR = r"C:\Users\tonda\Desktop\hrp\models_scaled"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# 1. TABULAR MODEL (FRAMINGHAM / CVD DATASET WITH IMPUTATION)
# ==========================================================
print("\n=== 1. Training Calibrated Tabular Gradient Boosting ===")
possible_csvs = [
    os.path.join(DATASET_ROOT, "cardio vascular_dataset.csv"),
    os.path.join(DATASET_ROOT, "cardiovascular_dataset.csv"),
    os.path.join(DATASET_ROOT, "CVD Dataset.csv"),
    os.path.join(DATASET_ROOT, "cardiovascular_health_monitoring_dataset.csv"),
    os.path.join(DATASET_ROOT, "heart.csv")
]

csv_path = next((p for p in possible_csvs if os.path.exists(p)), None)
if csv_path is None:
    raise FileNotFoundError(f"Could not find any tabular dataset CSV in: {DATASET_ROOT}")

print(f"Loading Tabular Data from: {csv_path}")

with open(csv_path, 'r', encoding='utf-8') as f:
    first_line = f.readline()
sep = ';' if ';' in first_line else (',' if ',' in first_line else r'\s+')

df = pd.read_csv(csv_path, sep=sep)
df.columns = df.columns.str.strip().str.lower()

# Detect and isolate target column
possible_targets = ['tenyearchd', 'cardio', 'target', 'heartdisease', 'output']
target_col = next((c for c in possible_targets if c in df.columns), df.columns[-1])
print(f"Identified Target Column: '{target_col}'")

# Drop rows where target y itself is NaN
df = df.dropna(subset=[target_col])

if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

if 'age' in df.columns and df['age'].max() > 150:
    df['age_years'] = (df['age'] / 365.25).astype(int)
    df.drop('age', axis=1, inplace=True)

X = df.drop(target_col, axis=1)
y = df[target_col].astype(int)

# Handle categorical variables if any
X = pd.get_dummies(X, drop_first=True)

# Median imputation for clinical missing feature values (NaNs)
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

base_gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.08, random_state=42)
calibrated_gb = CalibratedClassifierCV(estimator=base_gb, method='sigmoid', cv=3)
calibrated_gb.fit(X_train, y_train)

y_prob_tab = calibrated_gb.predict_proba(X_test)[:, 1]
y_pred_tab = (y_prob_tab >= 0.5).astype(int)

print(f"Tabular Model Results -> Accuracy: {accuracy_score(y_test, y_pred_tab)*100:.2f}% | AUC: {roc_auc_score(y_test, y_prob_tab):.4f}")
joblib.dump(calibrated_gb, os.path.join(OUTPUT_DIR, "best_tabular_model_scaled.pkl"))

# ==========================================================
# 2. HEART SOUND 2D-CNN (CinC 2016 + CirCor + EPHNOGRAM)
# ==========================================================
print("\n=== 2. Training Scaled Heart Sound 2D-CNN ===")
audio_files = []
for pattern in ["cinic2016/**/*.wav", "audio/**/*.wav", "*circor*/**/*.wav", "*Heart Sound*/**/*.wav", "*ephnogram*/**/*.wav"]:
    audio_files.extend(glob.glob(os.path.join(DATASET_ROOT, pattern), recursive=True))

audio_files = list(set(audio_files))
print(f"Identified {len(audio_files)} Heart Sound Audio Files.")

def extract_mel_spectrogram(file_path, duration=5.0, sr=2000):
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        target_len = int(duration * sr)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, fmax=800)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
        if mel_norm.shape[1] < 64:
            mel_norm = np.pad(mel_norm, ((0, 0), (0, 64 - mel_norm.shape[1])))
        else:
            mel_norm = mel_norm[:, :64]
        return mel_norm
    except Exception:
        return None

X_audio, y_audio = [], []
for f in audio_files[:2500]:
    spec = extract_mel_spectrogram(f)
    if spec is not None and spec.shape == (64, 64):
        X_audio.append(spec)
        label = 0 if any(norm_tag in f for norm_tag in ['Normal', '_N', 'a0001', 'b0001']) else 1
        y_audio.append(label)

X_audio = np.expand_dims(np.array(X_audio), -1)
y_audio = np.array(y_audio)
print(f"Prepared Mel-Spectrogram input tensor shape: {X_audio.shape}")

X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(X_audio, y_audio, test_size=0.2, random_state=42)

hs_model = keras.Sequential([
    keras.Input(shape=(64, 64, 1)),
    Conv2D(32, (3, 3), padding='same'), BatchNormalization(), Activation('relu'), MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), padding='same'), BatchNormalization(), Activation('relu'), MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), padding='same'), BatchNormalization(), Activation('relu'),
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'), Dropout(0.3),
    Dense(1, activation='sigmoid')
])

hs_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
hs_model.fit(X_tr_a, y_tr_a, epochs=10, batch_size=32, validation_data=(X_te_a, y_te_a), verbose=1)
hs_model.save(os.path.join(OUTPUT_DIR, "heart_sound_cnn_scaled.keras"))

# ==========================================================
# 3. 12-LEAD ECG 1D-CNN (PTB-XL Multi-Lead Waveforms)
# ==========================================================
print("\n=== 3. Training 12-Lead ECG 1D-CNN on PTB-XL Waveforms ===")
ptb_folder = os.path.join(DATASET_ROOT, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
hea_files = glob.glob(os.path.join(ptb_folder, "records100", "**", "*.hea"), recursive=True)
print(f"Identified {len(hea_files)} PTB-XL 12-Lead Header Records.")

X_ecg, y_ecg = [], []
for hea in hea_files[:2500]:
    record_name = os.path.splitext(hea)[0]
    try:
        signal, fields = wfdb.rdsamp(record_name)
        if signal.shape[1] == 12:
            if len(signal) < 1000:
                signal = np.pad(signal, ((0, 1000 - len(signal)), (0, 0)))
            else:
                signal = signal[:1000, :]
            X_ecg.append(signal)
            comments = " ".join(fields.get('comments', []))
            label = 0 if 'NORM' in comments else 1
            y_ecg.append(label)
    except Exception:
        continue

X_ecg = np.array(X_ecg)
y_ecg = np.array(y_ecg)
print(f"Loaded 12-lead ECG tensor shape: {X_ecg.shape}")

X_tr_e, X_te_e, y_tr_e, y_te_e = train_test_split(X_ecg, y_ecg, test_size=0.2, random_state=42)

def build_ecg_12lead_model(input_shape=(1000, 12)):
    inputs = keras.Input(shape=input_shape)
    x = Conv1D(64, kernel_size=7, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(256, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs, outputs)

ecg_model = build_ecg_12lead_model(input_shape=(1000, 12))
ecg_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
ecg_model.fit(X_tr_e, y_tr_e, epochs=10, batch_size=32, validation_data=(X_te_e, y_te_e), verbose=1)
ecg_model.save(os.path.join(OUTPUT_DIR, "ecg_cnn_scaled.keras"))

print(f"\nTraining pipeline complete! All retrained models exported to: {OUTPUT_DIR}")