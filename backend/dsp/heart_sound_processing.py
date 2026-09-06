"""
DSP Processing for Laptop Microphone Heart Sounds
Converts raw audio to a spectrogram/features matching CinC 2016 dataset.
"""
import numpy as np
import librosa
from scipy import signal

def process_laptop_heart_sound(audio_data, sample_rate, target_fs=2000, duration=5):
    """
    Process laptop microphone audio to extract heart sound features.
    
    Args:
        audio_data: Raw audio samples (numpy array)
        sample_rate: Original sampling rate
        target_fs: Target sampling frequency (Hz)
        duration: Desired duration (seconds)
    
    Returns:
        mel_spectrogram: Preprocessed mel-spectrogram for CNN input
        features: Dictionary of extracted features
    """
    if len(audio_data) == 0:
        return None, None
    
    # 1. Resample to target sampling rate (2 kHz is enough for heart sounds)
    audio_resampled = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_fs)
    
    # 2. Trim or pad to desired duration
    target_length = int(target_fs * duration)
    if len(audio_resampled) < target_length:
        audio_resampled = np.pad(audio_resampled, (0, target_length - len(audio_resampled)))
    else:
        audio_resampled = audio_resampled[:target_length]
    
    # 3. Bandpass filter (20 Hz - 400 Hz) to isolate heart sounds
    nyquist = target_fs / 2
    b, a = signal.butter(4, [20 / nyquist, 400 / nyquist], btype='band')
    filtered = signal.filtfilt(b, a, audio_resampled)
    
    # 4. Spectral Subtraction to remove ambient noise (e.g., fan)
    # Profile noise from first 0.5 seconds (assume no heart sound)
    noise_samples = int(0.5 * target_fs)
    noise_profile = np.mean(np.abs(np.fft.rfft(filtered[:noise_samples])))
    if noise_profile > 0:
        # Simple spectral subtraction in frequency domain
        fft_audio = np.fft.rfft(filtered)
        fft_audio = np.maximum(np.abs(fft_audio) - noise_profile, 0)
        filtered = np.fft.irfft(fft_audio, n=len(filtered))
    
    # 5. Shannon Energy Envelope (helps CNN identify heart beat peaks)
    envelope = np.square(filtered) * np.log(1 + np.square(filtered))
    
    # 6. Compute mel-spectrogram (64 mel bands, 157 time steps - to match training)
    mel_spec = librosa.feature.melspectrogram(
        y=envelope, sr=target_fs, n_mels=64, 
        n_fft=512, hop_length=256
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Resize to exactly 64x157 (as used in training)
    from cv2 import resize
    mel_resized = resize(mel_db, (157, 64))
    mel_resized = mel_resized[..., np.newaxis]  # Add channel dimension
    
    # 7. Extract features
    features = {
        'heart_rate_bpm': estimate_heart_rate_from_envelope(envelope, target_fs),
        'snr_estimate': compute_snr(filtered, noise_profile),
        'mean_spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=filtered, sr=target_fs)),
        'signal_duration_sec': duration
    }
    
    return mel_resized, features

def estimate_heart_rate_from_envelope(envelope, fs):
    """Estimate heart rate from Shannon energy envelope."""
    peaks, _ = signal.find_peaks(envelope, distance=fs*0.3, height=np.std(envelope)*0.5)
    if len(peaks) < 2:
        return 72  # default
    rr_intervals = np.diff(peaks) / fs
    if np.mean(rr_intervals) > 0:
        return 60 / np.mean(rr_intervals)
    return 72

def compute_snr(signal_clean, noise_estimate):
    """Compute SNR estimate."""
    signal_power = np.mean(signal_clean**2)
    if noise_estimate > 0:
        return 10 * np.log10(signal_power / (noise_estimate**2))
    return 0