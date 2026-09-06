import numpy
import pandas
import sklearn
import torch
import librosa
import fastapi
import shap
import mlflow
import optuna
import cv2
import noisereduce

print("All packages imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {numpy.__version__}")
print(f"Pandas version: {pandas.__version__}")
print(f"Librosa version: {getattr(librosa, '__version__', 'N/A')}")
print(f"Scikit-learn version: {sklearn.__version__}")