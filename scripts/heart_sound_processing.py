"""
CardioRisk AI - Heart Sound Processing with Noise Cancellation
Handles phonocardiography (PCG) signal processing.
"""
import numpy as np
import librosa
import noisereduce as nr
from scipy import signal
from scipy.signal import find_peaks
from typing import cast
import warnings
warnings.filterwarnings('ignore')

def load_audio(file_path, sr=16000):
    """Load audio file and resample."""
    audio, original_sr = librosa.load(file_path, sr=sr, mono=True)
    return audio, sr

def reduce_noise(audio, sr, stationary=True):
    """
    Apply noise reduction to heart sound recording.
    Uses spectral gating.
    """
    # Estimate noise from first 0.5 seconds (assumed to be noise-only)
    noise_sample = audio[:int(0.5 * sr)]
    
    # Apply noise reduction
    audio_denoised = nr.reduce_noise(
        y=audio, 
        sr=sr,
        y_noise=noise_sample,
        stationary=stationary
    )
    
    return audio_denoised

def bandpass_filter(audio, sr, low=25, high=400):
    """Apply bandpass filter to isolate heart sounds."""
    nyquist = sr / 2
    low_norm = low / nyquist
    high_norm = high / nyquist
    
    b, a = cast(tuple[np.ndarray, np.ndarray], signal.butter(4, [low_norm, high_norm], btype='band'))
    filtered = signal.filtfilt(b, a, audio)
    
    return filtered

def detect_heart_sounds(audio, sr):
    """
    Detect S1 and S2 heart sounds using energy-based method.
    Returns envelope and detected peaks.
    """
    # Compute envelope using Hilbert transform
    analytic = np.asarray(signal.hilbert(audio))
    envelope = np.abs(analytic)
    
    # Smooth envelope
    window = int(0.05 * sr)  # 50ms smoothing
    envelope_smooth = np.convolve(envelope, np.ones(window)/window, mode='same')
    
    # Find peaks (heart sounds)
    min_distance = int(0.25 * sr)  # Minimum 250ms between sounds
    peaks, properties = find_peaks(
        envelope_smooth, 
        distance=min_distance,
        height=0.3 * np.max(envelope_smooth)
    )
    
    return envelope_smooth, peaks

def classify_s1_s2(audio, peaks, sr):
    """
    Classify detected peaks as S1 or S2 based on timing intervals.
    S1-S2 interval (systole) is shorter than S2-S1 (diastole).
    """
    if len(peaks) < 2:
        return [], []
    
    # Compute intervals between successive peaks
    intervals = np.diff(peaks) / sr  # in seconds
    
    s1_indices = []
    s2_indices = []
    
    # First peak is typically S1
    for i in range(len(peaks) - 1):
        if i % 2 == 0:
            s1_indices.append(peaks[i])
        else:
            s2_indices.append(peaks[i])
    
    return s1_indices, s2_indices

def extract_heart_sound_features(audio, sr):
    """
    Extract comprehensive heart sound features.
    """
    # Denoise and filter
    audio_clean = reduce_noise(audio, sr)
    audio_filtered = bandpass_filter(audio_clean, sr)
    
    # Detect heart sounds
    envelope, peaks = detect_heart_sounds(audio_filtered, sr)
    
    if len(peaks) < 4:
        return {
            'error': 'Insufficient heart sounds detected',
            'num_sounds_detected': len(peaks)
        }
    
    # Heart rate from S1-S1 intervals
    s1_peaks = peaks[::2]  # Every other peak
    if len(s1_peaks) >= 2:
        rr_intervals = np.diff(s1_peaks) / sr
        heart_rate = 60 / np.mean(rr_intervals)
    else:
        heart_rate = None
    
    # Frequency features
    mel_spec = librosa.feature.melspectrogram(y=audio_filtered, sr=sr, n_mels=64)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_filtered, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_filtered, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_filtered)[0]
    
    features = {
        'heart_rate_bpm': round(float(heart_rate), 1) if heart_rate else None,
        'num_sounds_detected': len(peaks),
        'mean_envelope_amplitude': round(float(np.mean(envelope)), 6),
        'max_envelope_amplitude': round(float(np.max(envelope)), 6),
        'mean_spectral_centroid': round(float(np.mean(spectral_centroid)), 2),
        'mean_spectral_bandwidth': round(float(np.mean(spectral_bandwidth)), 2),
        'mean_zero_crossing_rate': round(float(np.mean(zero_crossing_rate)), 6),
        'signal_duration_sec': round(len(audio) / sr, 2),
        'snr_estimate': round(float(np.max(envelope) / (np.std(envelope) + 1e-10)), 2)
    }
    
    return features

def generate_synthetic_heart_sound(duration=10, sr=16000, heart_rate=72):
    """
    Generate synthetic heart sound for testing.
    """
    t = np.linspace(0, duration, duration * sr)
    beat_period = 60 / heart_rate
    audio = np.zeros_like(t)
    
    for i, ti in enumerate(t):
        cycle = ti % beat_period
        
        # S1 (lub) - longer, lower frequency
        if 0.0 < cycle < 0.12:
            audio[i] += 0.8 * np.exp(-((cycle - 0.06)**2) / 0.0008) * np.sin(2*np.pi*50*cycle)
        # S2 (dub) - shorter, higher frequency
        elif 0.30 < cycle < 0.40:
            audio[i] += 0.5 * np.exp(-((cycle - 0.35)**2) / 0.0006) * np.sin(2*np.pi*70*cycle)
    
    # Add noise
    audio += np.random.normal(0, 0.05, audio.shape)
    
    return audio

if __name__ == "__main__":
    print("Heart Sound Processing Module")
    print("=" * 40)
    
    # Generate synthetic heart sound
    print("Generating synthetic heart sound...")
    audio = generate_synthetic_heart_sound(duration=10, sr=16000, heart_rate=72)
    
    # Process
    print("Processing...")
    features = extract_heart_sound_features(audio, sr=16000)
    
    print("\nHeart Sound Analysis:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    print("\nModule ready for real audio input.")