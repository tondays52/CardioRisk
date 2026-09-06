"""
DSP Processing for Laptop Webcam PPG
Converts raw video frames to a PPG signal matching clinical dataset characteristics.
"""
import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline
import cv2

def process_webcam_ppg(video_frames, timestamps=None, target_fs=30):
    """
    Process webcam video frames to extract PPG signal.
    
    Args:
        video_frames: List of frames (numpy arrays) from webcam
        timestamps: Optional list of timestamps (if None, assume uniform)
        target_fs: Target sampling frequency (Hz)
    
    Returns:
        normalized_signal: Preprocessed PPG signal ready for HRV extraction
    """
    if not video_frames:
        return np.array([]), np.array([])
    
    # 1. Extract green channel (blood absorbs green light best)
    green_values = []
    for frame in video_frames:
        # Convert to RGB if needed
        if frame.ndim == 3:
            green = frame[:, :, 1].mean()  # Green channel
        else:
            green = frame.mean()
        green_values.append(green)
    
    green_signal = np.array(green_values)
    
    # 2. Handle timestamps and interpolate to fixed sampling rate
    n_frames = len(green_signal)
    if timestamps is None:
        # Assume uniform sampling
        timestamps = np.arange(n_frames) / target_fs
    else:
        timestamps = np.array(timestamps)
    
    # Create uniform time grid
    total_duration = timestamps[-1] - timestamps[0]
    n_target = int(total_duration * target_fs)
    if n_target < 2:
        n_target = n_frames  # fallback
    uniform_times = np.linspace(timestamps[0], timestamps[-1], n_target)
    
    # Cubic spline interpolation
    try:
        cs = CubicSpline(timestamps, green_signal, extrapolate=False)
        interpolated = cs(uniform_times)
    except Exception:
        # Fallback to linear interpolation
        interpolated = np.interp(uniform_times, timestamps, green_signal)
    
    # 3. Detrend (remove baseline wander from finger pressure changes)
    detrended = signal.detrend(interpolated)
    
    # 4. Bandpass filter (0.7 Hz to 4.0 Hz for 42-240 BPM)
    nyquist = target_fs / 2
    if nyquist > 4.0:
        b, a = signal.butter(4, [0.7 / nyquist, 4.0 / nyquist], btype='band')
        filtered = signal.filtfilt(b, a, detrended)
    else:
        # If sampling rate is too low, skip filtering
        filtered = detrended
    
    # 5. Normalize (zero mean, unit variance)
    if np.std(filtered) > 0:
        normalized = (filtered - np.mean(filtered)) / np.std(filtered)
    else:
        normalized = filtered - np.mean(filtered)
    
    return normalized, uniform_times

def extract_hrv_features(ppg_signal, fs=30):
    """
    Extract HRV features from PPG signal.
    """
    if len(ppg_signal) < 10:
        return None
    
    # Find peaks (simple threshold-based)
    threshold = np.std(ppg_signal) * 0.8
    peaks, _ = signal.find_peaks(ppg_signal, height=threshold, distance=fs*0.4)
    
    if len(peaks) < 3:
        return None
    
    # RR intervals in ms
    rr_intervals = np.diff(peaks) / fs * 1000
    
    features = {
        'heart_rate_bpm': 60000 / np.mean(rr_intervals),
        'sdnn_ms': np.std(rr_intervals),
        'rmssd_ms': np.sqrt(np.mean(np.diff(rr_intervals)**2)),
        'pnn50_pct': np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100,
        'num_peaks': len(peaks)
    }
    return features