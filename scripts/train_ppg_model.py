"""
CardioRisk AI - PPG/HRV Feature Extraction
Extracts heart rate variability features from PPG signals.
"""
import os
import numpy as np
import pandas as pd
import wfdb
from scipy import signal
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

def find_ppg_data():
    """Find PPG dataset in various locations."""
    paths = [
        r"C:\Users\tonda\Desktop\dataset\bidmc-ppg-and-respiration-dataset-1.0.0",
        r"C:\Users\tonda\Desktop\dataset\bidmc-ppg-and-respiration-dataset-1.0.0\bidmc_csv",
        r"C:\Users\tonda\Desktop\data\bidmc-ppg-and-respiration-dataset-1.0.0",
    ]
    
    for path in paths:
        if os.path.exists(path):
            print(f"✅ Found PPG data at: {path}")
            return path
    
    # Search recursively
    for root, dirs, files in os.walk(r"C:\Users\tonda\Desktop"):
        for dir_name in dirs:
            if 'ppg' in dir_name.lower() or 'bidmc' in dir_name.lower():
                full_path = os.path.join(root, dir_name)
                print(f"✅ Found PPG data at: {full_path}")
                return full_path
    
    print("❌ PPG dataset not found")
    return None

def extract_hrv_features(ppg_signal, fs=125):
    """Extract HRV features from PPG signal."""
    try:
        # Bandpass filter
        b, a = signal.butter(4, [0.5, 10], btype='band', fs=fs)
        filtered = signal.filtfilt(b, a, ppg_signal)
        
        # Detect peaks
        peaks, _ = find_peaks(filtered, distance=fs*0.4, prominence=np.std(filtered)*0.5)
        
        if len(peaks) < 5:
            return None
        
        rr_intervals = np.diff(peaks) / fs * 1000  # in ms
        
        if len(rr_intervals) < 2:
            return None
        
        features = {
            'heart_rate': 60000 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0,
            'sdnn': np.std(rr_intervals) if len(rr_intervals) > 0 else 0,
            'rmssd': np.sqrt(np.mean(np.diff(rr_intervals)**2)) if len(rr_intervals) > 1 else 0,
            'pnn50': np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100 if len(rr_intervals) > 1 else 0,
            'hrv_range': np.max(rr_intervals) - np.min(rr_intervals) if len(rr_intervals) > 0 else 0,
            'num_peaks': len(peaks)
        }
        
        return features
    except Exception as e:
        return None

def process_ppg():
    """Process PPG data."""
    print("=" * 60)
    print("CardioRisk AI - PPG/HRV Feature Extraction")
    print("=" * 60)
    
    data_dir = find_ppg_data()
    if data_dir is None:
        return
    
    features_list = []
    
    # Look for .dat files in the main directory
    for f in os.listdir(data_dir):
        if f.endswith('.dat'):
            base = f.replace('.dat', '')
            try:
                file_path = os.path.join(data_dir, base)
                record = wfdb.rdrecord(file_path)
                
                if isinstance(record, wfdb.MultiRecord):
                    record = record.multi_to_single(physical=True)
                
                if record.p_signal is not None and len(record.p_signal) > 0:
                    channel_idx = 0
                    if hasattr(record, 'sig_name') and record.sig_name:
                        for idx, name in enumerate(record.sig_name):
                            if any(k in name.lower() for k in ['pleth', 'ppg']):
                                channel_idx = idx
                                break
                    p_signal = np.asarray(record.p_signal)
                    signal_data = p_signal[:, channel_idx]
                    fs = getattr(record, 'fs', 125) or 125
                    features = extract_hrv_features(signal_data, fs)
                    if features:
                        features['filename'] = f
                        features['record_id'] = base
                        features_list.append(features)
                        print(f"✅ Processed {f}")
            except Exception as e:
                print(f"⚠️ Could not process {f}: {e}")
                continue
    
    if features_list:
        df = pd.DataFrame(features_list)
        os.makedirs("models/ppg", exist_ok=True)
        df.to_csv("models/ppg/ppg_features.csv", index=False)
        print(f"\n✅ Extracted features from {len(df)} PPG recordings")
        print("\n📊 Feature Summary:")
        print(df.describe())
        return df
    
    print("⚠️ No PPG features extracted")
    return None

if __name__ == "__main__":
    process_ppg()