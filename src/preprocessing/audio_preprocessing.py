import os
import io
import numpy as np
import librosa
try:
    import noisereduce as nr
except ImportError:
    nr = None
from scipy.signal import butter, sosfilt

# ============================================================
# PROJECT AUDIO SETTINGS
# ============================================================

TARGET_SR = 4000
TARGET_DURATION = 6
TARGET_SAMPLES = TARGET_SR * TARGET_DURATION  # 24,000 samples

TIME_STEPS = 47
N_MFCC = 13
N_MELS = 128
COMBINED_FEATURES = N_MFCC + N_MELS  # 141 features

CLASS_NAMES = [
    "Asthma",
    "Bronchial",
    "COPD",
    "Healthy",
    "Pneumonia",
]

CLASS_TO_INDEX = {
    "asthma": 0,
    "Bronchial": 1,
    "copd": 2,
    "healthy": 3,
    "pneumonia": 4,
    "Asthma": 0,
    "COPD": 2,
    "Healthy": 3,
    "Pneumonia": 4,
}


# ============================================================
# LOAD AND RESAMPLE AUDIO
# ============================================================

def load_audio(file_path_or_buffer, return_sr=False):
    """
    Load an audio file or buffer and resample it to 4000 Hz.
    Audio is always loaded as mono and returned as float32.

    Parameters
    ----------
    file_path_or_buffer : str, Path, or file-like object
        Audio source to load.
    return_sr : bool, default=False
        If True, return (audio, sample_rate).
        If False, return audio array only (maintains backward compatibility).

    Returns
    -------
    audio : numpy.ndarray
        Mono audio signal resampled to TARGET_SR.
    sr : int (optional)
        Sampling rate (4000 Hz).
    """
    audio, sample_rate = librosa.load(
        file_path_or_buffer,
        sr=TARGET_SR,
        mono=True
    )

    if audio is None or len(audio) == 0:
        raise ValueError("Audio file contains no usable audio samples.")

    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    if return_sr:
        return audio, sample_rate
    return audio


# ============================================================
# AUDIO SILENCE & QUALITY CHECK
# ============================================================

def is_silent_audio(audio, threshold=1e-5):
    """
    Check whether an audio signal is effectively silent.

    Parameters
    ----------
    audio : numpy.ndarray
        Audio signal array.
    threshold : float, default=1e-5
        Peak amplitude threshold below which signal is considered silent.

    Returns
    -------
    bool
        True if the maximum absolute amplitude is below threshold.
    """
    if len(audio) == 0:
        return True
    return bool(np.max(np.abs(audio)) < threshold)


# ============================================================
# FIX AUDIO LENGTH
# ============================================================

def fix_audio_length(audio, target_samples=TARGET_SAMPLES):
    """
    Make every audio recording exactly the target duration (default 6 seconds).

    Short recordings are padded with zeros.
    Long recordings are trimmed to the first target_samples.

    Parameters
    ----------
    audio : numpy.ndarray
        Audio signal.
    target_samples : int, default=24,000
        Number of samples expected.

    Returns
    -------
    audio : numpy.ndarray
        Fixed-length audio signal.
    """
    if len(audio) < target_samples:
        padding = target_samples - len(audio)
        audio = np.pad(
            audio,
            (0, padding),
            mode="constant"
        )
    elif len(audio) > target_samples:
        audio = audio[:target_samples]

    return audio.astype(np.float32)


# ============================================================
# CREATE OVERLAPPING AUDIO WINDOWS (FOR MULTI-WINDOW ANALYSIS)
# ============================================================

def create_audio_windows(audio, window_samples=TARGET_SAMPLES, step_seconds=3, sr=TARGET_SR):
    """
    Split audio into 6-second windows.
    If recording <= 6 seconds, returns [padded_audio].
    If recording > 6 seconds, generates overlapping windows with step_seconds.

    Parameters
    ----------
    audio : numpy.ndarray
        Mono audio signal.
    window_samples : int, default=24,000
        Samples per window (6s at 4,000 Hz).
    step_seconds : int, default=3
        Overlap step size in seconds.
    sr : int, default=4000
        Sample rate.

    Returns
    -------
    List[numpy.ndarray]
        List of 6-second audio segments.
    """
    if len(audio) == 0:
        raise ValueError("Audio array is empty.")

    if len(audio) <= window_samples:
        padded = np.pad(
            audio,
            (0, window_samples - len(audio)),
            mode="constant"
        )
        return [padded.astype(np.float32)]

    step = int(step_seconds * sr)
    windows = []
    start = 0

    while start + window_samples <= len(audio):
        windows.append(audio[start:start + window_samples].astype(np.float32))
        start += step

    # Ensure final segment is covered
    final_start = len(audio) - window_samples
    if final_start > 0 and (not windows or not np.array_equal(windows[-1], audio[final_start:final_start + window_samples])):
        windows.append(audio[final_start:final_start + window_samples].astype(np.float32))

    return windows


# ============================================================
# NORMALIZE AUDIO
# ============================================================

def normalize_audio(audio):
    """
    Normalize the amplitude of the audio signal by its peak absolute value.

    Parameters
    ----------
    audio : numpy.ndarray
        Audio signal.

    Returns
    -------
    audio : numpy.ndarray
        Peak-normalized float32 audio signal in range [-1.0, 1.0].
    """
    max_value = np.max(np.abs(audio))
    if max_value > 0:
        audio = audio / max_value
    return audio.astype(np.float32)


# ============================================================
# BAND-PASS FILTER (EXPERIMENTAL / FILTERED PIPELINES)
# ============================================================

def bandpass_filter(
    audio,
    lowcut=50,
    highcut=1800,
    sr=TARGET_SR
):
    """
    Apply a 4th-order Butterworth band-pass filter.

    Note: The original baseline model (saved_models/cnn_bilstm_best.keras)
    was trained without bandpass filtering. Keep apply_bandpass=False for
    baseline inference.
    """
    nyquist = sr / 2
    highcut = min(highcut, nyquist - 1)

    sos = butter(
        N=4,
        Wn=[lowcut, highcut],
        btype="bandpass",
        fs=sr,
        output="sos"
    )

    filtered_audio = sosfilt(sos, audio)
    return filtered_audio.astype(np.float32)


# ============================================================
# EXTERNAL NOISE REDUCTION (EXPERIMENTAL)
# ============================================================

def reduce_noise(audio, sr=TARGET_SR):
    """
    Reduce stationary/background noise using spectral noise reduction.

    Note: The original baseline model (saved_models/cnn_bilstm_best.keras)
    was trained without spectral noise reduction. Keep apply_noise_reduction=False
    for baseline inference.
    """
    if nr is None:
        return audio.astype(np.float32)
    try:
        cleaned_audio = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=True,
            prop_decrease=0.8,
            time_mask_smooth_ms=64
        )
        return cleaned_audio.astype(np.float32)
    except Exception as error:
        print(f"Noise reduction warning: {error}")
        return audio.astype(np.float32)


# ============================================================
# EXTRACT MFCC FEATURES
# ============================================================

def extract_mfcc(audio, sr=TARGET_SR, n_mfcc=N_MFCC):
    """
    Extract 13 MFCC features from preprocessed audio.
    Shape returned: (n_mfcc, time_frames) -> (13, 47) for 6s audio at 4000Hz.
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc
    )
    return mfcc.astype(np.float32)


# ============================================================
# EXTRACT MEL SPECTROGRAM
# ============================================================

def extract_mel_spectrogram(audio, sr=TARGET_SR, n_mels=N_MELS):
    """
    Extract Mel Spectrogram features and convert them to decibel scale.
    Shape returned: (n_mels, time_frames) -> (128, 47) for 6s audio at 4000Hz.
    """
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels
    )

    mel_db = librosa.power_to_db(
        mel_spectrogram,
        ref=np.max
    )
    return mel_db.astype(np.float32)


# ============================================================
# EXTRACT COMBINED 2D FEATURES FOR MODEL INFERENCE
# ============================================================

def extract_features_2d(audio, sr=TARGET_SR, expected_frames=TIME_STEPS):
    """
    Extract and combine MFCC (13) and Mel-spectrogram (128) into a 2D matrix
    of shape (TIME_STEPS, COMBINED_FEATURES) = (47, 141).

    Parameters
    ----------
    audio : numpy.ndarray
        Preprocessed, normalized audio signal (typically 24,000 samples).
    sr : int, default=4000
        Sampling rate.
    expected_frames : int, default=47
        Time frames expected by the CNN-BiLSTM model.

    Returns
    -------
    numpy.ndarray
        Array of shape (47, 141) with float32 dtype.
    """
    mfcc = extract_mfcc(audio, sr=sr)        # (13, 47)
    mel = extract_mel_spectrogram(audio, sr=sr)  # (128, 47)

    # Transpose to (time_frames, feature_dim)
    mfcc_t = mfcc.T  # (47, 13)
    mel_t = mel.T    # (47, 128)

    # Align time dimensions
    min_frames = min(mfcc_t.shape[0], mel_t.shape[0])
    combined = np.concatenate([mfcc_t[:min_frames], mel_t[:min_frames]], axis=1)

    # Ensure exactly expected_frames
    if combined.shape[0] < expected_frames:
        pad_len = expected_frames - combined.shape[0]
        padding = np.zeros((pad_len, combined.shape[1]), dtype=np.float32)
        combined = np.vstack([combined, padding])
    elif combined.shape[0] > expected_frames:
        combined = combined[:expected_frames]

    if combined.shape != (expected_frames, COMBINED_FEATURES):
        raise ValueError(
            f"Feature shape {combined.shape} does not match expected ({expected_frames}, {COMBINED_FEATURES})"
        )

    return combined.astype(np.float32)


# ============================================================
# COMPLETE AUDIO PREPROCESSING PIPELINE
# ============================================================

def preprocess_audio(file_path_or_buffer, apply_bandpass=False, apply_noise_reduction=False):
    """
    Complete audio preprocessing pipeline matching the baseline model.

    Steps:
    1. Load audio and resample to 4000 Hz mono
    2. Fix duration to 6 seconds (24,000 samples)
    3. (Optional) Band-pass filtering
    4. (Optional) Noise reduction
    5. Peak amplitude normalization
    6. Extract MFCC (13)
    7. Extract Mel Spectrogram (128)

    By default, apply_bandpass and apply_noise_reduction are False to match
    the training pipeline of the 93.41% accuracy baseline model.

    Returns
    -------
    audio : numpy.ndarray
        Cleaned 6s audio (24,000 samples).
    mfcc : numpy.ndarray
        MFCC features (13, 47).
    mel_spectrogram : numpy.ndarray
        Mel Spectrogram features (128, 47).
    """
    # Step 1: Load and resample
    audio = load_audio(file_path_or_buffer)

    # Step 2: Fix audio length
    audio = fix_audio_length(audio)

    # Step 3: Optional Band-pass filtering
    if apply_bandpass:
        audio = bandpass_filter(audio)

    # Step 4: Optional Noise reduction
    if apply_noise_reduction:
        audio = reduce_noise(audio)

    # Step 5: Normalize amplitude
    audio = normalize_audio(audio)

    # Step 6: Extract MFCC
    mfcc = extract_mfcc(audio)

    # Step 7: Extract Mel Spectrogram
    mel_spectrogram = extract_mel_spectrogram(audio)

    return (
        audio,
        mfcc,
        mel_spectrogram
    )
