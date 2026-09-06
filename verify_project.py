"""
Project Verification Script
Run this to check everything at once
"""
import os
import sqlite3
import pandas as pd

import sys
import io

# Ensure UTF-8 output for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print('=' * 60)
print('CARDIO RISK AI - PROJECT VERIFICATION')
print('=' * 60)

# ============================================
# 1. DATASET CHECK
# ============================================
print('\n1. DATASETS:')
print('-' * 40)

datasets = {
    'ECG (PTB-XL)': 'data/raw/ecg',
    'Tabular (Kaggle CVD)': 'data/raw/tabular/cardio_train.csv',
    'Heart Sound Model': 'models/audio/heart_sound_cnn.h5',
    'Retinal Features': 'models/retinal/retinal_features.csv',
    'PPG Features': 'models/ppg/ppg_features.csv',
}

for name, path in datasets.items():
    exists = os.path.exists(path)
    status = '✅' if exists else '❌'
    print(f'{status} {name}')

# ============================================
# 2. DATABASE PATIENTS CHECK
# ============================================
print('\n2. DATABASE PATIENTS:')
print('-' * 40)

try:
    conn = sqlite3.connect('data/cardiorisk.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT a.patient_id, p.age, p.gender, a.risk_category, a.fused_risk_score 
        FROM assessments a 
        LEFT JOIN patients p ON a.patient_id = p.id
    ''')
    rows = cursor.fetchall()
    
    if rows:
        print(f'Total Patients: {len(rows)}')
        print('Patient Details:')
        for row in rows:
            risk = row[4] if row[4] else 'N/A'
            print(f'  ID {row[0]}: Age {row[1]}, Category {row[3]}, Risk {risk}%')
    else:
        print('No patients found in database.')
    conn.close()
except Exception as e:
    print(f'Error reading database: {e}')

# ============================================
# 3. MODEL FILES CHECK
# ============================================
print('\n3. MODEL FILES:')
print('-' * 40)

model_files = {
    'ECG Model': 'models/ecg/ecg_cnn_final.keras',
    'Heart Sound Model': 'models/audio/heart_sound_cnn.h5',
    'Tabular Model': 'models/tabular/best_tabular_model.pkl',
    'Tabular Scaler': 'models/tabular/scaler.pkl',
}

for name, path in model_files.items():
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024 * 1024)
        print(f'✅ {name}: {size:.2f} MB')
    else:
        print(f'❌ {name}: NOT FOUND')

# ============================================
# 4. TRAINING HISTORY CHECK
# ============================================
print('\n4. TRAINING HISTORY:')
print('-' * 40)

history_files = [
    'models/audio/training_history.csv',
    'models/ecg/training_history.csv',
    'models/ecg/training_history_final.csv'
]

for hf in history_files:
    if os.path.exists(hf):
        try:
            df = pd.read_csv(hf)
            print(f'✅ {hf}: {len(df)} rows')
            if 'accuracy' in df and 'val_accuracy' in df:
                print(f'   Final Train Acc: {df["accuracy"].iloc[-1]:.4f}')
                print(f'   Final Val Acc: {df["val_accuracy"].iloc[-1]:.4f}')
        except:
            print(f'⚠️ {hf}: exists but could not read')

# ============================================
# 5. SUMMARY
# ============================================
print('\n' + '=' * 60)
print('VERIFICATION COMPLETE')
print('=' * 60)