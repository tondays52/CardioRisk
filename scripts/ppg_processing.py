"""
CardioRisk AI - PPG Signal Processing & HRV Analysis
Extracts PPG from video, computes HRV features, estimates BP.
"""
import numpy as np
import cv2
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

def extract_ppg_from_video(video_path, fps=30):
    """
    Extract PPG signal from fingertip video.
    Returns raw PPG signal and timestamps.
    """
    cap = cv2.VideoCapture(video_path)
    red_intensities = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Average red channel intensity
        red = frame[:, :, 2].mean()
        red_intensities.append(red)
    
    cap.release()
    
    if len(red_intensities) == 0:
        raise ValueError("Could not read video frames")
    
    ppg_raw = np.array(red_intensities)
    return ppg_raw, fps

def clean_ppg_signal(ppg_raw, fps=30):
    """
    Filter and normalize PPG signal.
    """
    # Remove DC component
    ppg = ppg_raw - np.mean(ppg_raw)
    
    # Bandpass filter 0.5-8 Hz (30-480 bpm)
    nyquist = fps / 2
    low = 0.5 / nyquist
    high = 8.0 / nyquist
    
    b, a = signal.butter(4, [low, high], btype='band')
    ppg_filtered = signal.filtfilt(b, a, ppg)
    
    # Normalize
    ppg_normalized = (ppg_filtered - np.mean(ppg_filtered)) / np.std(ppg_filtered)
    
    return ppg_normalized

def detect_peaks(ppg_signal, fps=30, min_distance=0.5):
    """
    Detect systolic peaks in PPG signal.
    min_distance in seconds.
    """
    min_dist_samples = int(min_distance * fps)
    peaks, _ = find_peaks(ppg_signal, distance=min_dist_samples, 
                          prominence=0.3 * np.std(ppg_signal))
    return peaks

def compute_hrv_features(peaks, fps=30):
    """
    Compute HRV features from PPG peaks.
    """
    if len(peaks) < 4:
        return None
    
    # Inter-beat intervals (IBI) in milliseconds
    ibi = np.diff(peaks) * (1000 / fps)  # ms
    
    # Time domain features
    sdnn = np.std(ibi)
    rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2))
    pnn50 = np.sum(np.abs(np.diff(ibi)) > 50) / len(ibi) * 100
    
    # Frequency domain features (simplified)
    if len(ibi) >= 4:
        # Interpolate to regular time series for spectral analysis
        ibi_interp = np.interp(
            np.linspace(0, len(ibi)-1, len(ibi)*4),
            np.arange(len(ibi)),
            ibi
        )
        
        # Simple FFT
        freqs = np.fft.rfftfreq(len(ibi_interp), d=1/4)  # Hz
        spectrum = np.abs(np.fft.rfft(ibi_interp - np.mean(ibi_interp)))
        
        # LF: 0.04-0.15 Hz, HF: 0.15-0.4 Hz
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.4)
        
        lf_power = np.sum(spectrum[lf_mask] ** 2)
        hf_power = np.sum(spectrum[hf_mask] ** 2)
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
    else:
        lf_power = hf_power = lf_hf_ratio = 0
    
    # Heart rate
    heart_rate = 60000 / np.mean(ibi) if len(ibi) > 0 else 0
    
    return {
        'heart_rate_bpm': round(heart_rate, 1),
        'sdnn_ms': round(sdnn, 2),
        'rmssd_ms': round(rmssd, 2),
        'pnn50_pct': round(pnn50, 2),
        'lf_power': round(lf_power, 4),
        'hf_power': round(hf_power, 4),
        'lf_hf_ratio': round(lf_hf_ratio, 2),
        'ibi_mean_ms': round(np.mean(ibi), 2),
        'ibi_std_ms': round(np.std(ibi), 2),
        'num_beats': len(peaks)
    }

def compute_arterial_stiffness(ppg_signal, fps=30):
    """
    Estimate arterial stiffness from PPG waveform morphology.
    """
    # Average pulse shape
    peaks, _ = find_peaks(ppg_signal, distance=int(0.5*fps))
    
    if len(peaks) < 2:
        return None
    
    # Extract one representative pulse
    pulse_width = int(0.4 * fps)  # 400ms window
    pulses = []
    
    for peak in peaks[:-1]:
        start = peak - pulse_width // 3
        end = peak + 2 * pulse_width // 3
        if start >= 0 and end < len(ppg_signal):
            pulses.append(ppg_signal[start:end])
    
    if len(pulses) < 3:
        return None
    
    avg_pulse = np.mean(pulses, axis=0)
    
    # Find systolic peak and reflected wave
    systolic_peak_idx = np.argmax(avg_pulse)
    systolic_peak = avg_pulse[systolic_peak_idx]
    
    # Look for reflected wave in second half of pulse
    second_half = avg_pulse[systolic_peak_idx + len(avg_pulse)//4:]
    if len(second_half) < 5:
        return None
    
    reflected_idx = np.argmax(second_half)
    reflected_peak = second_half[reflected_idx]
    
    # Stiffness index based on time delay
    reflected_time = reflected_idx + len(avg_pulse)//4
    time_delay = reflected_time / fps  # seconds
    stiffness_index = 10 / time_delay if time_delay > 0.1 else 8.0
    
    # Reflection index
    reflection_index = reflected_peak / systolic_peak if systolic_peak > 0 else 0
    
    return {
        'stiffness_index': round(stiffness_index, 2),
        'reflection_index': round(reflection_index, 3),
        'pulse_wave_velocity_est': round(stiffness_index * 1.2, 2)
    }

def estimate_blood_pressure(ppg_features, age, gender, height_cm, weight_kg):
    """
    Simplified cuffless BP estimation.
    Uses PPG features + demographics.
    """
    # This is a placeholder formula.
    # In production, this would be a trained ML model.
    if ppg_features is None:
        return None
    
    si = ppg_features.get('stiffness_index', 8.0)
    ri = ppg_features.get('reflection_index', 0.6)
    
    # Simplified regression (placeholder - replace with real model later)
    systolic = 110 + 0.5 * age + (si - 8) * 3 + (ri - 0.6) * 15
    diastolic = 70 + 0.2 * age + (si - 8) * 1.5 + (ri - 0.6) * 10
    
    return {
        'estimated_systolic_bp': round(systolic, 1),
        'estimated_diastolic_bp': round(diastolic, 1)
    }

def process_ppg_video(video_path, age=None, gender=None, height_cm=None, weight_kg=None):
    """
    Complete PPG processing pipeline.
    """
    # Extract PPG
    ppg_raw, fps = extract_ppg_from_video(video_path)
    
    # Clean signal
    ppg_clean = clean_ppg_signal(ppg_raw, fps)
    
    # Detect peaks
    peaks = detect_peaks(ppg_clean, fps)
    
    if len(peaks) < 4:
        return {
            'error': 'Insufficient peaks detected. Please ensure fingertip covers camera properly.',
            'num_beats_detected': len(peaks)
        }
    
    # Compute HRV
    hrv = compute_hrv_features(peaks, fps)
    
    # Compute arterial stiffness
    stiffness = compute_arterial_stiffness(ppg_clean, fps)
    
    # Estimate BP
    bp = estimate_blood_pressure(stiffness, age, gender, height_cm, weight_kg)
    
    return {
        'hrv': hrv,
        'stiffness': stiffness,
        'blood_pressure': bp,
        'signal_quality': {
            'num_beats': len(peaks),
            'signal_std': round(float(np.std(ppg_clean)), 4),
            'snr_estimate': round(float(np.std(ppg_clean) / 0.1), 2)
        }
    }

if __name__ == "__main__":
    # Test with simulated data
    print("PPG Processing Module")
    print("=" * 40)
    
    # Generate synthetic PPG
    fps = 30
    duration = 30  # seconds
    t = np.linspace(0, duration, duration * fps)
    
    # Simulate heartbeat at 72 bpm
    heart_rate = 72
    beat_period = 60 / heart_rate
    ppg_signal = np.zeros_like(t)
    
    for i, ti in enumerate(t):
        cycle_time = ti % beat_period
        if cycle_time < 0.15:
            ppg_signal[i] = np.exp(-((cycle_time - 0.075) ** 2) / 0.002)
        elif cycle_time < 0.3:
            ppg_signal[i] = 0.3 * np.exp(-((cycle_time - 0.225) ** 2) / 0.005)
    
    # Add noise
    ppg_signal += np.random.normal(0, 0.02, ppg_signal.shape)
    
    # Process
    peaks = detect_peaks(ppg_signal, fps)
    hrv = compute_hrv_features(peaks, fps)
    stiffness = compute_arterial_stiffness(ppg_signal, fps)
    bp = estimate_blood_pressure(stiffness, age=45, gender=2, height_cm=175, weight_kg=80)
    
    if hrv is not None:
        print(f"Heart Rate: {hrv['heart_rate_bpm']} bpm")
        print(f"SDNN: {hrv['sdnn_ms']} ms")
        print(f"RMSSD: {hrv['rmssd_ms']} ms")
    if stiffness is not None:
        print(f"Stiffness Index: {stiffness['stiffness_index']}")
    if bp is not None:
        print(f"Estimated BP: {bp['estimated_systolic_bp']}/{bp['estimated_diastolic_bp']}")
    print("\nModule ready for real video input.")