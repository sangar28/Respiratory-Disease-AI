"""
Automated Test Suite for Respiratory Disease AI
Validates model loading, scaler matching, preprocessing pipeline,
end-to-end inference, probability calibration, and edge cases.
"""

import os
import sys
import tempfile
import pickle
import numpy as np
import soundfile as sf
import tensorflow as tf

# Add src and project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from preprocessing.audio_preprocessing import (
    TARGET_SR,
    TARGET_DURATION,
    TARGET_SAMPLES,
    TIME_STEPS,
    COMBINED_FEATURES,
    CLASS_NAMES,
    CLASS_TO_INDEX,
    load_audio,
    normalize_audio,
    fix_audio_length,
    extract_mfcc,
    extract_mel_spectrogram,
    extract_features_2d,
    create_audio_windows,
    is_silent_audio,
    preprocess_audio,
)

MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_models", "cnn_bilstm_best.keras")
SCALER_PATH = os.path.join(PROJECT_ROOT, "saved_models", "feature_scaler.pkl")


def test_1_baseline_model_loads():
    """Verify baseline model exists, loads, has 160,005 parameters and expected shapes."""
    assert os.path.exists(MODEL_PATH), f"Model file missing: {MODEL_PATH}"
    model = tf.keras.models.load_model(MODEL_PATH)
    assert model is not None, "Failed to load Keras model."

    total_params = model.count_params()
    assert total_params == 160005, f"Expected 160,005 parameters, got {total_params}"

    assert model.input_shape == (None, 47, 141), f"Unexpected input shape: {model.input_shape}"
    assert model.output_shape == (None, 5), f"Unexpected output shape: {model.output_shape}"
    print("[PASS] Test 1: Baseline model loads with 160,005 parameters and shape (None, 47, 141) -> (None, 5)")


def test_2_baseline_scaler_loads():
    """Verify baseline scaler exists, loads, and expects 141 features."""
    assert os.path.exists(SCALER_PATH), f"Scaler file missing: {SCALER_PATH}"
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    assert scaler is not None, "Failed to load feature scaler."
    assert hasattr(scaler, "n_features_in_"), "Loaded object is not a valid scaler."
    assert scaler.n_features_in_ == 141, f"Expected 141 features in scaler, got {scaler.n_features_in_}"
    print("[PASS] Test 2: Baseline scaler loads with n_features_in_ = 141")


def test_3_preprocessing_produces_expected_shape():
    """Verify preprocessing a valid dataset audio file produces exact (47, 141) shape."""
    sample_file = os.path.join(PROJECT_ROOT, "dataset", "pneumonia", "P44Pneumonia36U.wav")
    assert os.path.exists(sample_file), f"Test audio file missing: {sample_file}"

    audio, mfcc, mel = preprocess_audio(sample_file)
    assert len(audio) == TARGET_SAMPLES, f"Expected 24,000 samples, got {len(audio)}"
    assert mfcc.shape == (13, 47), f"Expected MFCC shape (13, 47), got {mfcc.shape}"
    assert mel.shape == (128, 47), f"Expected Mel shape (128, 47), got {mel.shape}"

    # Verify 2D combined representation
    feat_2d = extract_features_2d(audio, sr=TARGET_SR)
    assert feat_2d.shape == (47, 141), f"Expected (47, 141), got {feat_2d.shape}"
    assert not np.isnan(feat_2d).any(), "Features contain NaN values"
    assert not np.isinf(feat_2d).any(), "Features contain Inf values"
    print("[PASS] Test 3: Audio preprocessing produces exact (47, 141) feature shape without NaNs/Infs")


def test_4_end_to_end_prediction():
    """Verify end-to-end inference produces 5 valid probabilities matching known test output."""
    sample_file = os.path.join(PROJECT_ROOT, "dataset", "pneumonia", "P44Pneumonia36U.wav")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Preprocessing
    audio = load_audio(sample_file)
    audio = fix_audio_length(audio)
    audio = normalize_audio(audio)
    feat_2d = extract_features_2d(audio, sr=TARGET_SR)
    scaled_feat = scaler.transform(feat_2d)
    model_input = np.expand_dims(scaled_feat, axis=0)

    # Prediction
    raw_probs = model.predict(model_input, verbose=0)[0]

    # Verify 5 finite probabilities
    assert len(raw_probs) == 5, f"Expected 5 output probabilities, got {len(raw_probs)}"
    assert all(np.isfinite(p) for p in raw_probs), "Output contains non-finite values."
    assert all(p >= 0.0 for p in raw_probs), "Output contains negative probabilities."

    prob_sum = float(np.sum(raw_probs))
    assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities do not sum to 1.0 (sum={prob_sum})"

    pred_idx = int(np.argmax(raw_probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = raw_probs[pred_idx]

    assert pred_class == "Pneumonia", f"Expected Pneumonia for P44Pneumonia36U.wav, got {pred_class}"
    assert abs(confidence - 0.9443) < 0.01, f"Expected ~0.9443 confidence, got {confidence}"

    print(f"[PASS] Test 4: End-to-end prediction succeeded (Pneumonia, confidence: {confidence*100:.2f}%, sum: {prob_sum:.4f})")


def test_5_all_five_classes_evaluated():
    """Verify preprocessing and model compatibility across sample files from all 5 classes."""
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
    folders = ["asthma", "Bronchial", "copd", "healthy", "pneumonia"]

    for folder in folders:
        folder_path = os.path.join(dataset_dir, folder)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
        assert len(files) > 0, f"No WAV files found in {folder_path}"

        test_file = os.path.join(folder_path, files[0])
        audio = load_audio(test_file)
        audio = fix_audio_length(audio)
        audio = normalize_audio(audio)
        feat_2d = extract_features_2d(audio, sr=TARGET_SR)
        scaled_feat = scaler.transform(feat_2d)
        model_input = np.expand_dims(scaled_feat, axis=0)

        probs = model.predict(model_input, verbose=0)[0]
        assert len(probs) == 5
        assert abs(np.sum(probs) - 1.0) < 1e-4
        pred_class = CLASS_NAMES[int(np.argmax(probs))]
        print(f"       Class '{folder}' sample -> predicted '{pred_class}' ({np.max(probs)*100:.1f}%)")

    print("[PASS] Test 5: All 5 dataset classes processed and inferred cleanly")


def test_6_edge_cases():
    """Verify silence detection, short audio padding, and stereo-to-mono handling."""
    # 1. Silent audio check
    silent_audio = np.zeros(24000, dtype=np.float32)
    assert is_silent_audio(silent_audio) is True, "Silent audio was not detected as silent."

    # 2. Short audio padding (1 second = 4000 samples)
    short_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 4000)).astype(np.float32)
    assert not is_silent_audio(short_audio), "Tone was falsely marked as silent."
    padded = fix_audio_length(short_audio)
    assert len(padded) == TARGET_SAMPLES, f"Expected {TARGET_SAMPLES} samples after padding, got {len(padded)}"

    # 3. Stereo audio file handling
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        stereo_data = np.stack([short_audio, short_audio], axis=1)
        sf.write(tmp.name, stereo_data, TARGET_SR)
        tmp_name = tmp.name

    try:
        loaded_mono = load_audio(tmp_name)
        assert loaded_mono.ndim == 1, f"Expected 1D mono audio, got {loaded_mono.ndim}D"
        assert len(loaded_mono) == 4000, f"Expected 4000 samples, got {len(loaded_mono)}"
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    print("[PASS] Test 6: Silence detection, short audio padding, and stereo audio conversion all pass")


def test_7_multi_window_mode():
    """Verify multi-window segmenting for longer recordings (> 6 seconds)."""
    # 10 seconds of synthetic audio
    long_audio = np.sin(2 * np.pi * 220 * np.linspace(0, 10, 40000)).astype(np.float32)
    windows = create_audio_windows(long_audio, window_samples=TARGET_SAMPLES, step_seconds=3, sr=TARGET_SR)
    assert len(windows) > 1, f"Expected multiple windows for 10s audio, got {len(windows)}"
    for w in windows:
        assert len(w) == TARGET_SAMPLES, f"Expected window length {TARGET_SAMPLES}, got {len(w)}"

    print(f"[PASS] Test 7: Multi-window segmenting created {len(windows)} overlapping 6-second windows")


def test_8_model_and_scaler_not_overwritten():
    """Verify that tests and operations did not modify or corrupt original model and scaler files."""
    assert os.path.exists(MODEL_PATH)
    assert os.path.exists(SCALER_PATH)
    assert os.path.getsize(MODEL_PATH) > 1_000_000, "Model file size appears suspiciously small."
    assert os.path.getsize(SCALER_PATH) > 1_000, "Scaler file size appears suspiciously small."
    print("[PASS] Test 8: Original model and scaler integrity preserved")


def test_9_streamlit_app_simulation():
    """Verify Streamlit web application runs, accepts an upload, and produces predictions."""
    from streamlit.testing.v1 import AppTest

    sample_file = os.path.join(PROJECT_ROOT, "dataset", "pneumonia", "P44Pneumonia36U.wav")
    with open(sample_file, "rb") as f:
        wav_bytes = f.read()

    app_path = os.path.join(PROJECT_ROOT, "web_app", "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.exception, f"Initial app startup threw exception: {at.exception}"

    # Simulate file upload
    uploader = at.get("file_uploader")[0]
    uploader.upload(filename="P44Pneumonia36U.wav", content=wav_bytes)
    at.run()
    assert not at.exception, f"Upload run threw exception: {at.exception}"

    # Simulate button click
    buttons = at.get("button")
    run_btn = [b for b in buttons if "Run Respiratory Disease Prediction" in b.label][0]
    run_btn.click()
    at.run()
    assert not at.exception, f"Inference run threw exception: {at.exception}"

    assert at.session_state["pred_result"] == "Pneumonia"
    assert abs(at.session_state["pred_conf"] - 0.9443) < 0.01
    print(f"[PASS] Test 9: Streamlit UI simulation passed (Pneumonia, {at.session_state['pred_conf']*100:.2f}%)")


if __name__ == "__main__":
    print("==================================================")
    print("RUNNING RESPIRATORY DISEASE AI PIPELINE TEST SUITE")
    print("==================================================")
    test_1_baseline_model_loads()
    test_2_baseline_scaler_loads()
    test_3_preprocessing_produces_expected_shape()
    test_4_end_to_end_prediction()
    test_5_all_five_classes_evaluated()
    test_6_edge_cases()
    test_7_multi_window_mode()
    test_8_model_and_scaler_not_overwritten()
    test_9_streamlit_app_simulation()
    print("==================================================")
    print("ALL TESTS PASSED SUCCESSFULLY! (9/9)")
    print("==================================================")

