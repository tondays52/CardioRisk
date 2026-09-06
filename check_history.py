import pandas as pd
import os

print('=' * 50)
print('TRAINING HISTORY CHECK')
print('=' * 50)

# Heart Sound
print('\n1. HEART SOUND MODEL:')
path = 'models/audio/training_history.csv'
if os.path.exists(path):
    df = pd.read_csv(path)
    print(f'   Final Training Accuracy: {df["accuracy"].iloc[-1]:.4f}')
    print(f'   Final Validation Accuracy: {df["val_accuracy"].iloc[-1]:.4f}')
    print(f'   Final Training Loss: {df["loss"].iloc[-1]:.4f}')
    print(f'   Final Validation Loss: {df["val_loss"].iloc[-1]:.4f}')
else:
    print('   No training history found')

# ECG
print('\n2. ECG MODEL:')
path = 'models/ecg/training_history_final.csv'
if os.path.exists(path):
    df = pd.read_csv(path)
    print(f'   Final Training Accuracy: {df["accuracy"].iloc[-1]:.4f}')
    print(f'   Final Validation Accuracy: {df["val_accuracy"].iloc[-1]:.4f}')
    print(f'   Final Training AUC: {df["auc"].iloc[-1]:.4f}')
    print(f'   Final Validation AUC: {df["val_auc"].iloc[-1]:.4f}')
else:
    print('   No training history found')

# Check if heart sound history has same values (the issue mentioned by reviewer)
print('\n3. HEART SOUND - TRAIN/VAL COMPARISON:')
if os.path.exists('models/audio/training_history.csv'):
    df = pd.read_csv('models/audio/training_history.csv')
    train_acc = df["accuracy"].iloc[-1]
    val_acc = df["val_accuracy"].iloc[-1]
    if train_acc == val_acc:
        print('   ⚠️ Training and validation accuracy are identical!')
        print(f'   Both are: {train_acc:.4f}')
        print('   This suggests possible data leakage or the same number was pasted twice.')
    else:
        print(f'   Train Acc: {train_acc:.4f}')
        print(f'   Val Acc: {val_acc:.4f}')
        print(f'   Difference: {abs(train_acc - val_acc):.4f}')

print('\n' + '=' * 50)