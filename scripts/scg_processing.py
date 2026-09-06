"""
CardioRisk AI - Seismocardiography (SCG) Processing
Processes accelerometer-based chest vibration signals.
"""
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

def load_accelerometer_signal(file_path):
    """
    Load accelerometer data from CSV/NPY.
    Expected format: 3 columns (x, y, z) or 1 column (magnitude).
    """
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(file_path)
        return df.values
    else:
        raise ValueError(f"Unsupported format: {file_path}")

def preprocess_scg(accel_data, fs=100):
    """
    Preprocess SCG signal: bandpass filter and normalize.
    """
    # If 3-axis, compute magnitude
    if len(accel_data.shape) > 1 and accel_data.shape[1] >= 3:
        scg_signal = np.sqrt(np.sum(accel_data[:, :3]**2, axis=1))
    else:
        scg_signal = accel_data.flatten() if len(accel_data.shape) > 1 else accel_data
    
    # Remove DC
    scg = scg_signal - np.mean(scg_signal)
    
    # Bandpass filter 0.5-40 Hz
    nyquist = fs / 2
    low = 0.5 / nyquist
    high = 40 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    scg_filtered = signal.filtfilt(b, a, scg)
    
    # Normalize
    scg_normalized = (scg_filtered - np.mean(scg_filtered)) / np.std(scg_filtered)
    
    return scg_normalized

def detect_scg_events(scg_signal, fs=100):
    """
    Detect SCG events (MC, AO, AC) from filtered signal.
    """
    # Find all peaks
    peaks, properties = find_peaks(scg_signal, distance=int(0.2*fs), 
                                    height=0.3*np.std(scg_signal))
    
    # Classify events based on amplitude and timing
    # MC (Mitral Closure): first peak after R-wave
    # AO (Aortic Opening): largest peak
    # AC (Aortic Closure): second major peak
    events = []
    
    for i, peak in enumerate(peaks):
        events.append({
            'time_idx': peak,
            'time_sec': peak / fs,
            'amplitude': scg_signal[peak]
        })
    
    return events

def extract_scg_features(scg_signal, fs=100):
    """
    Extract SCG features for cardiovascular assessment.
    """
    scg_clean = preprocess_scg(scg_signal, fs)
    events = detect_scg_events(scg_clean, fs)
    
    if len(events) < 3:
        return {'error': 'Insufficient SCG events detected'}
    
    # Inter-event intervals
    event_times = np.array([e['time_sec'] for e in events])
    intervals = np.diff(event_times)
    
    # Heart rate from major peaks (AO)
    amplitudes = np.array([e['amplitude'] for e in events])
    major_peaks_idx = np.argsort(amplitudes)[-len(events)//2:]  # Top 50%
    major_times = event_times[major_peaks_idx]
    major_times = np.sort(major_times)
    
    if len(major_times) >= 2:
        heart_rate = 60 / np.mean(np.diff(major_times))
    else:
        heart_rate = None
    
    features = {
        'heart_rate_bpm': round(float(heart_rate), 1) if heart_rate else None,
        'num_events_detected': len(events),
        'mean_interval_sec': round(float(np.mean(intervals)), 3) if len(intervals) > 0 else None,
        'interval_std_sec': round(float(np.std(intervals)), 3) if len(intervals) > 0 else None,
        'mean_peak_amplitude': round(float(np.mean(amplitudes)), 4),
        'max_peak_amplitude': round(float(np.max(amplitudes)), 4),
        'signal_energy': round(float(np.sum(scg_clean**2)), 4),
        'snr_estimate': round(float(np.max(np.abs(scg_clean)) / (np.std(scg_clean) + 1e-10)), 2)
    }
    
    return features

def generate_synthetic_scg(duration=10, fs=100, heart_rate=72):
    """
    Generate synthetic SCG signal for testing.
    """
    t = np.linspace(0, duration, duration * fs)
    beat_period = 60 / heart_rate
    scg = np.zeros_like(t)
    
    for i, ti in enumerate(t):
        cycle = ti % beat_period
        
        # MC (Mitral Closure)
        if 0.05 < cycle < 0.10:
            scg[i] += 0.5 * np.exp(-((cycle - 0.075)**2) / 0.0004)
        # AO (Aortic Opening)
        elif 0.10 < cycle < 0.15:
            scg[i] += 1.0 * np.exp(-((cycle - 0.125)**2) / 0.0003)
        # AC (Aortic Closure)
        elif 0.30 < cycle < 0.38:
            scg[i] += 0.7 * np.exp(-((cycle - 0.34)**2) / 0.0005)
    
    # Add noise
    scg += np.random.normal(0, 0.1, scg.shape)
    
    return scg

if __name__ == "__main__":
    print("SCG Processing Module")
    print("=" * 40)
    
    # Generate synthetic SCG
    print("Generating synthetic SCG...")
    scg = generate_synthetic_scg(duration=10, fs=100, heart_rate=72)
    
    # Process
    print("Processing...")
    features = extract_scg_features(scg, fs=100)
    
    print("\nSCG Analysis:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    print("\nModule ready for real accelerometer input.")