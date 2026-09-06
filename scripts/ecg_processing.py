"""
CardioRisk AI - ECG Signal Processing
Handles 12-lead ECG signal processing and feature extraction.
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

def load_ecg_signal(file_path, fs=500):
    """
    Load ECG signal from file.
    Supports CSV, NPY, and TXT formats.
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        return df.values.T, fs
    elif file_path.endswith('.npy'):
        return np.load(file_path), fs
    elif file_path.endswith('.txt'):
        return np.loadtxt(file_path).T, fs
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def clean_ecg(ecg_signal, fs=500):
    """
    Apply bandpass and notch filters to ECG signal.
    """
    # Remove DC
    ecg = ecg_signal - np.mean(ecg_signal, axis=1, keepdims=True)
    
    # Bandpass 0.5-40 Hz
    nyquist = fs / 2
    low = 0.5 / nyquist
    high = 40 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    ecg_filtered = signal.filtfilt(b, a, ecg, axis=1)
    
    # Notch filter 50/60 Hz
    for freq in [50, 60]:
        if freq < nyquist:
            b_notch, a_notch = signal.iirnotch(freq, 30, fs)
            ecg_filtered = signal.filtfilt(b_notch, a_notch, ecg_filtered, axis=1)
    
    return ecg_filtered

def detect_r_peaks(ecg_lead, fs=500):
    """
    Detect R-peaks in a single ECG lead using Pan-Tompkins-like approach.
    """
    # Derivative
    diff = np.diff(ecg_lead)
    diff = np.append(diff, 0)
    
    # Squaring
    squared = diff ** 2
    
    # Moving average
    window = int(0.15 * fs)
    ma = np.convolve(squared, np.ones(window)/window, mode='same')
    
    # Threshold
    threshold = 0.5 * np.max(ma)
    peaks, _ = find_peaks(ma, height=threshold, distance=int(0.3*fs))
    
    # Refine peak positions on original signal
    refined_peaks = []
    search_window = int(0.05 * fs)
    for p in peaks:
        start = max(0, p - search_window)
        end = min(len(ecg_lead), p + search_window)
        if start < end:
            refined = start + np.argmax(ecg_lead[start:end])
            refined_peaks.append(refined)
    
    return np.array(refined_peaks)

def compute_ecg_features(ecg_signal, fs=500):
    """
    Compute ECG features from 12-lead signal.
    """
    # Use lead II for R-peak detection (usually index 1 in standard 12-lead)
    lead_ii = ecg_signal[1] if ecg_signal.shape[0] > 1 else ecg_signal[0]
    
    r_peaks = detect_r_peaks(lead_ii, fs)
    
    if len(r_peaks) < 2:
        return {'error': 'Insufficient R-peaks detected'}
    
    # Heart rate
    rr_intervals = np.diff(r_peaks) / fs  # seconds
    heart_rate = 60 / np.mean(rr_intervals)
    
    # QRS duration estimation
    qrs_durations = []
    qrs_window = int(0.12 * fs)  # 120ms max QRS
    for p in r_peaks:
        start = max(0, p - int(0.05*fs))
        end = min(len(lead_ii), p + int(0.07*fs))
        segment = lead_ii[start:end]
        qrs_durations.append(len(segment) / fs)
    
    # QT interval estimation (simplified)
    qt_window = int(0.45 * fs)
    qt_intervals = []
    for p in r_peaks:
        end = min(len(lead_ii), p + qt_window)
        if end > p:
            # Find T-wave end (simplified - lowest point after R in this window)
            segment = lead_ii[p:end]
            if len(segment) > 10:
                t_end = p + np.argmin(segment)
                qt_intervals.append((t_end - p) / fs)
    
    # ST segment analysis (simplified)
    st_elevations = []
    st_window_start = int(0.06 * fs)
    st_window_end = int(0.10 * fs)
    for p in r_peaks:
        start = p + st_window_start
        end = p + st_window_end
        if end < len(lead_ii):
            st_value = np.mean(lead_ii[start:end])
            st_elevations.append(st_value)
    
    features = {
        'heart_rate_bpm': round(float(heart_rate), 1),
        'mean_rr_interval_ms': round(float(np.mean(rr_intervals) * 1000), 2),
        'rr_interval_std_ms': round(float(np.std(rr_intervals) * 1000), 2),
        'mean_qrs_duration_ms': round(float(np.mean(qrs_durations) * 1000), 2),
        'mean_qt_interval_ms': round(float(np.mean(qt_intervals) * 1000), 2) if qt_intervals else None,
        'st_elevation_mean_mv': round(float(np.mean(st_elevations)), 4) if st_elevations else None,
        'st_elevation_std_mv': round(float(np.std(st_elevations)), 4) if st_elevations else None,
        'num_beats_detected': len(r_peaks),
        'signal_quality': round(float(np.std(lead_ii) / 0.1), 2)
    }
    
    return features

def generate_synthetic_ecg(duration=10, fs=500, heart_rate=72):
    """
    Generate synthetic ECG for testing.
    """
    t = np.linspace(0, duration, duration * fs)
    beat_period = 60 / heart_rate
    ecg = np.zeros_like(t)
    
    for i, ti in enumerate(t):
        cycle = ti % beat_period
        
        # P wave
        if 0.0 < cycle < 0.1:
            ecg[i] += 0.15 * np.exp(-((cycle - 0.05)**2) / 0.001)
        # QRS complex
        elif 0.15 < cycle < 0.25:
            if 0.15 < cycle < 0.17:
                ecg[i] -= 0.1
            elif 0.17 < cycle < 0.20:
                ecg[i] += 1.0
            elif 0.20 < cycle < 0.25:
                ecg[i] -= 0.3
        # T wave
        elif 0.25 < cycle < 0.45:
            ecg[i] += 0.3 * np.exp(-((cycle - 0.35)**2) / 0.002)
    
    # Add noise
    ecg += np.random.normal(0, 0.02, ecg.shape)
    
    # Create 12 leads with variations
    ecg_12lead = np.tile(ecg, (12, 1))
    for lead in range(12):
        ecg_12lead[lead] *= (0.5 + 0.5 * np.random.random())
    
    return ecg_12lead

if __name__ == "__main__":
    print("ECG Processing Module")
    print("=" * 40)
    
    # Generate synthetic ECG
    print("Generating synthetic 12-lead ECG...")
    ecg = generate_synthetic_ecg(duration=10, fs=500, heart_rate=72)
    
    # Process
    print("Processing ECG signal...")
    ecg_clean = clean_ecg(ecg, fs=500)
    features = compute_ecg_features(ecg_clean, fs=500)
    
    print("\nECG Analysis Results:")
    print(f"  Heart Rate: {features['heart_rate_bpm']} bpm")
    print(f"  Mean RR: {features['mean_rr_interval_ms']} ms")
    print(f"  RR Std: {features['rr_interval_std_ms']} ms")
    print(f"  QRS Duration: {features['mean_qrs_duration_ms']} ms")
    print(f"  QT Interval: {features['mean_qt_interval_ms']} ms")
    print(f"  Beats Detected: {features['num_beats_detected']}")
    print("\nModule ready for real ECG input.")