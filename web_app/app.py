# ============================================================
# AI-BASED RESPIRATORY DISEASE DETECTION
# Streamlit Web Application
# ============================================================

import os
import sys
import tempfile
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf

# ============================================================
# CONFIGURE PYTHON SEARCH PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(BASE_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from preprocessing.audio_preprocessing import (
    TARGET_SR,
    TARGET_DURATION,
    TARGET_SAMPLES,
    TIME_STEPS,
    COMBINED_FEATURES,
    CLASS_NAMES,
    load_audio,
    normalize_audio,
    fix_audio_length,
    extract_features_2d,
    create_audio_windows,
    is_silent_audio,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Respiratory Disease Sound Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "cnn_bilstm_best.keras",
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "feature_scaler.pkl",
)

# ============================================================
# CLASS DESCRIPTIONS
# ============================================================

CLASS_DESCRIPTIONS = {
    "Asthma": (
        "Asthma is a chronic inflammatory disorder of the airways characterized by "
        "recurrent episodes of wheezing, breathlessness, chest tightness, and coughing. "
        "The model identified acoustic patterns typically associated with musical, high-pitched wheezing."
    ),
    "Bronchial": (
        "Bronchial breath sounds are tubular, hollow sounds typically heard over the large airways. "
        "When detected in peripheral lung fields, they often signify lung consolidation, "
        "bronchiectasis, or localized airway inflammation."
    ),
    "COPD": (
        "Chronic Obstructive Pulmonary Disease (COPD) involves persistent respiratory symptoms and airflow "
        "limitation. Acoustic signatures often exhibit prolonged expiration, diminished baseline vesicular "
        "sounds, and coarse crackles or wheezes."
    ),
    "Healthy": (
        "Healthy respiratory sounds feature clear, soft vesicular breath sounds with unobstructed airflow "
        "and no adventitious acoustic phenomena such as wheezes, crackles, or stridor."
    ),
    "Pneumonia": (
        "Pneumonia is an infection that inflames the air sacs in one or both lungs, often causing them to fill "
        "with fluid. Acoustic markers commonly include late-inspiratory fine crackles (rales) and bronchial breathing."
    ),
}

LOW_CONFIDENCE_THRESHOLD = 0.60

# ============================================================
# CUSTOM CSS
# ============================================================

st.html(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 12%, rgba(20, 110, 245, 0.16), transparent 30%),
            radial-gradient(circle at 88% 22%, rgba(0, 210, 180, 0.12), transparent 28%),
            radial-gradient(circle at 50% 90%, rgba(90, 70, 220, 0.08), transparent 30%),
            linear-gradient(135deg, #06101d 0%, #071622 45%, #041116 100%);
        color: #f5f7fb;
        min-height: 100vh;
    }

    .block-container {
        max-width: 1050px;
        padding-top: 30px;
        padding-bottom: 60px;
    }

    .hero {
        position: relative;
        text-align: center;
        padding: 40px 25px;
        margin-bottom: 25px;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(16, 45, 78, 0.90), rgba(6, 32, 40, 0.90));
        border: 1px solid rgba(75, 180, 255, 0.28);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
    }

    .lung-icon {
        font-size: 58px;
        margin-bottom: 6px;
        display: inline-block;
        animation: breathing 4s ease-in-out infinite;
    }

    @keyframes breathing {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.08); }
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 10px;
        background: linear-gradient(90deg, #ffffff, #6ad6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        max-width: 720px;
        margin: auto;
        color: #b0c2d4;
        font-size: 16px;
        line-height: 1.6;
    }

    .badge-container {
        display: flex;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 15px;
    }

    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        background: rgba(40, 150, 255, 0.15);
        border: 1px solid rgba(80, 190, 255, 0.30);
        color: #7dd3fc;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    .section-card {
        margin-top: 20px;
        margin-bottom: 20px;
        padding: 24px;
        border-radius: 20px;
        background: rgba(13, 27, 42, 0.82);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.20);
    }

    .file-meta-card {
        padding: 16px;
        border-radius: 14px;
        background: rgba(18, 38, 58, 0.70);
        border: 1px solid rgba(75, 180, 255, 0.20);
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .result-card {
        margin-top: 25px;
        padding: 30px;
        text-align: center;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(8, 70, 55, 0.94), rgba(7, 43, 38, 0.94));
        border: 1px solid rgba(60, 230, 170, 0.35);
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
    }

    .result-card.warning {
        background: linear-gradient(135deg, rgba(77, 60, 12, 0.94), rgba(52, 43, 12, 0.94));
        border-color: rgba(245, 204, 70, 0.40);
    }

    .result-icon {
        font-size: 44px;
        margin-bottom: 6px;
    }

    .result-label {
        color: #a9dfd0;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 4px;
    }

    .result-class {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 6px;
        color: #ffffff;
    }

    .result-confidence {
        color: #c6e9df;
        font-size: 17px;
        font-weight: 600;
    }

    .probability-card {
        margin-top: 20px;
        padding: 22px;
        border-radius: 18px;
        background: rgba(18, 36, 48, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .probability-title {
        font-weight: 700;
        font-size: 19px;
        color: #ffffff;
        margin-bottom: 15px;
    }

    .probability-row {
        margin-bottom: 14px;
    }

    .probability-name {
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
        font-size: 14px;
        color: #dbeafe;
    }

    .probability-bar {
        height: 10px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.08);
        overflow: hidden;
    }

    .probability-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #1684df, #00cfae);
        transition: width 0.8s ease;
    }

    .probability-fill.top-match {
        background: linear-gradient(90deg, #00b4d8, #10b981);
    }

    .medical-note {
        margin-top: 30px;
        padding: 20px 24px;
        border-radius: 16px;
        background: rgba(78, 65, 15, 0.55);
        border: 1px solid rgba(240, 210, 80, 0.28);
        color: #fef08a;
        line-height: 1.6;
    }

    .medical-note-title {
        color: #fef9c3;
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 6px;
    }

    .footer {
        text-align: center;
        margin-top: 45px;
        padding: 20px;
        color: #64748b;
        font-size: 13px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    }

    audio {
        width: 100%;
        border-radius: 10px;
        margin-top: 8px;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
    </style>
    """
)

# ============================================================
# MODEL & SCALER RESOURCE LOADERS
# ============================================================

@st.cache_resource
def load_baseline_model():
    """
    Load the trained baseline CNN-BiLSTM model.
    Expected: (None, 47, 141) -> (None, 5) with 160,005 parameters.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Baseline model not found at:\n{MODEL_PATH}\n"
            "Please ensure saved_models/cnn_bilstm_best.keras exists."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    # Sanity check model inputs and outputs
    if model.input_shape != (None, TIME_STEPS, COMBINED_FEATURES):
        raise ValueError(
            f"Model input shape mismatch. Expected (None, {TIME_STEPS}, {COMBINED_FEATURES}), "
            f"got {model.input_shape}."
        )

    if model.output_shape != (None, len(CLASS_NAMES)):
        raise ValueError(
            f"Model output classes mismatch. Expected (None, {len(CLASS_NAMES)}), "
            f"got {model.output_shape}."
        )

    return model


@st.cache_resource
def load_baseline_scaler():
    """
    Load the StandardScaler fitted on baseline training features.
    Expected: n_features_in_ = 141.
    """
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Feature scaler not found at:\n{SCALER_PATH}\n"
            "Please ensure saved_models/feature_scaler.pkl exists."
        )

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    if getattr(scaler, "n_features_in_", None) != COMBINED_FEATURES:
        raise ValueError(
            f"Scaler features mismatch. Expected {COMBINED_FEATURES}, "
            f"got {getattr(scaler, 'n_features_in_', None)}."
        )

    return scaler


# ============================================================
# INFERENCE PIPELINE
# ============================================================

def run_audio_inference(audio_path, model, scaler, analysis_mode="baseline"):
    """
    Run the complete, verified inference pipeline on an uploaded WAV file.

    Parameters
    ----------
    audio_path : str
        Path to temporary WAV file.
    model : tf.keras.Model
        Loaded CNN-BiLSTM baseline model.
    scaler : StandardScaler
        Loaded baseline scaler.
    analysis_mode : str
        "baseline" for the clean 6-second evaluation pipeline.
        "multi_window" for overlapping sliding window analysis.

    Returns
    -------
    predicted_class : str
    confidence : float
    probabilities : numpy.ndarray (5,)
    window_count : int
    duration_sec : float
    """
    # 1. Load audio and resample to 4,000 Hz mono
    y, sr = load_audio(audio_path, return_sr=True)
    duration_sec = float(len(y)) / float(sr)

    # 2. Silence / energy check
    if is_silent_audio(y):
        raise ValueError(
            "The uploaded audio recording is silent or contains virtually no audible signal. "
            "Please provide a WAV file with audible respiratory breath sounds."
        )

    if analysis_mode == "baseline":
        # Baseline pipeline: 6 seconds fixed (padded if <6s, trimmed if >6s)
        audio_6s = fix_audio_length(y)
        audio_norm = normalize_audio(audio_6s)
        features_2d = extract_features_2d(audio_norm, sr=TARGET_SR)  # (47, 141)
        scaled_2d = scaler.transform(features_2d)  # (47, 141)
        model_input = np.expand_dims(scaled_2d, axis=0)  # (1, 47, 141)

        raw_probs = model.predict(model_input, verbose=0)[0]
        window_count = 1
    else:
        # Multi-window sliding pipeline
        windows = create_audio_windows(y, window_samples=TARGET_SAMPLES, step_seconds=3, sr=TARGET_SR)
        scaled_windows = []
        for window in windows:
            w_norm = normalize_audio(window)
            w_feat = extract_features_2d(w_norm, sr=TARGET_SR)
            w_scaled = scaler.transform(w_feat)
            scaled_windows.append(w_scaled)

        model_input = np.stack(scaled_windows, axis=0)
        probs_per_window = model.predict(model_input, verbose=0)
        raw_probs = np.mean(probs_per_window, axis=0)
        window_count = len(windows)

    # 3. Clean probabilities
    probs = np.asarray(raw_probs, dtype=np.float64)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.maximum(probs, 0.0)
    total = np.sum(probs)
    if total > 0:
        probs = probs / total

    predicted_index = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probs[predicted_index])

    return predicted_class, confidence, probs, window_count, duration_sec


# ============================================================
# HERO SECTION
# ============================================================

st.html(
    """
    <div class="hero">
        <div class="lung-icon">🫁</div>
        <div class="hero-title">Respiratory Sound Classifier</div>
        <div class="hero-subtitle">
            AI-based acoustic pattern recognition for respiratory auscultation sounds.
            Analyze lung audio recordings across five diagnostic classes using deep learning.
        </div>
        <div class="badge-container">
            <span class="badge">Baseline CNN-BiLSTM</span>
            <span class="badge">93.41% Clean Test Accuracy</span>
            <span class="badge">Target SR: 4,000 Hz</span>
            <span class="badge">141 Audio Features</span>
            <span class="badge">Research Prototype</span>
        </div>
    </div>
    """
)

# ============================================================
# PRE-FLIGHT VERIFICATION
# ============================================================

model_loaded_ok = False
model = None
scaler = None

try:
    model = load_baseline_model()
    scaler = load_baseline_scaler()
    model_loaded_ok = True
except Exception as err:
    st.error(f"⚠️ Model Initialization Error: {err}")
    st.stop()


# ============================================================
# UPLOAD SECTION
# ============================================================

st.html(
    """
    <div class="section-card">
        <div style="font-size: 20px; font-weight: 700; margin-bottom: 6px;">
            🎙️ Audio Input
        </div>
        <div style="color: #94a3b8; font-size: 14px; margin-bottom: 12px;">
            Upload an auscultation breathing recording in <code>.wav</code> format.
            Recordings are automatically resampled to 4,000 Hz mono and preprocessed.
        </div>
    </div>
    """
)

col_file, col_mode = st.columns([2, 1])

with col_file:
    uploaded_file = st.file_uploader(
        "Upload a WAV audio file",
        type=["wav"],
        label_visibility="collapsed",
        key="respiratory_wav_uploader",
    )

with col_mode:
    analysis_mode_choice = st.selectbox(
        "Inference Analysis Mode",
        options=["Baseline Standard (6s)", "Multi-Window Average"],
        index=0,
        help=(
            "Baseline Standard (6s): Matches the exact 93.41% baseline test set preprocessing "
            "(first 6s, zero-padded if shorter). Multi-Window Average: Evaluates sliding 6s windows "
            "across recordings longer than 6 seconds."
        ),
    )
    analysis_mode = "baseline" if "Baseline" in analysis_mode_choice else "multi_window"


# ============================================================
# FILE PREVIEW & METADATA
# ============================================================

if uploaded_file is not None:
    # Validate file size
    if uploaded_file.size == 0:
        st.error("The uploaded file is empty (0 bytes). Please upload a valid WAV recording.")
    else:
        file_size_kb = uploaded_file.size / 1024.0

        # Create temporary file to read audio info safely
        temp_preview = None
        duration_display = "Evaluating..."
        audio_valid = True

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_preview = tmp.name

            # Quick probe for duration and silence
            y_probe, sr_probe = load_audio(temp_preview, return_sr=True)
            duration_val = len(y_probe) / sr_probe
            duration_display = f"{duration_val:.2f} seconds"

            if is_silent_audio(y_probe):
                st.warning(
                    "⚠️ Notice: The uploaded recording has very low amplitude / appears silent. "
                    "Inference may yield uncalibrated results if no breath sounds are audible."
                )

        except Exception as probe_err:
            audio_valid = False
            st.error(
                f"Could not parse the audio file: {probe_err}. "
                "Please verify that the file is an uncorrupted standard WAV recording."
            )
        finally:
            if temp_preview and os.path.exists(temp_preview):
                os.remove(temp_preview)

        if audio_valid:
            col_meta1, col_meta2, col_meta3 = st.columns(3)
            with col_meta1:
                st.metric("Filename", uploaded_file.name)
            with col_meta2:
                st.metric("File Size", f"{file_size_kb:.1f} KB")
            with col_meta3:
                st.metric("Duration", duration_display)

            st.write("🎧 **Audio Playback:**")
            st.audio(uploaded_file, format="audio/wav")

            # Prediction & Reset Buttons
            col_btn1, col_btn2 = st.columns([2, 1])

            with col_btn1:
                run_btn = st.button("🔍 Run Respiratory Disease Prediction", type="primary", use_container_width=True)

            with col_btn2:
                if st.button("🔄 Reset / Clear", use_container_width=True):
                    for k in ["pred_result", "pred_conf", "pred_probs", "pred_windows", "pred_dur"]:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.rerun()

            if run_btn:
                temp_infer = None
                with st.spinner("Processing audio, extracting MFCC + Mel spectrograms, and running CNN-BiLSTM..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_inf:
                            tmp_inf.write(uploaded_file.getbuffer())
                            temp_infer = tmp_inf.name

                        (
                            pred_class,
                            conf,
                            probs,
                            w_count,
                            dur_sec,
                        ) = run_audio_inference(
                            temp_infer,
                            model,
                            scaler,
                            analysis_mode=analysis_mode,
                        )

                        st.session_state["pred_result"] = pred_class
                        st.session_state["pred_conf"] = conf
                        st.session_state["pred_probs"] = probs
                        st.session_state["pred_windows"] = w_count
                        st.session_state["pred_dur"] = dur_sec

                    except Exception as inf_err:
                        st.error(f"Inference failed: {inf_err}")
                    finally:
                        if temp_infer and os.path.exists(temp_infer):
                            os.remove(temp_infer)


# ============================================================
# RESULTS PRESENTATION
# ============================================================

if "pred_result" in st.session_state:
    pred_class = st.session_state["pred_result"]
    confidence = st.session_state["pred_conf"]
    probabilities = st.session_state["pred_probs"]
    window_count = st.session_state["pred_windows"]
    duration_sec = st.session_state["pred_dur"]

    conf_percent = confidence * 100.0
    is_low_conf = confidence < LOW_CONFIDENCE_THRESHOLD

    card_class = "result-card warning" if is_low_conf else "result-card"
    icon = "⚠️" if is_low_conf else "🫁"

    st.html(
        f"""
        <div class="{card_class}">
            <div class="result-icon">{icon}</div>
            <div class="result-label">Predicted Acoustic Pattern</div>
            <div class="result-class">{pred_class}</div>
            <div class="result-confidence">
                Model Softmax Confidence: <strong>{conf_percent:.2f}%</strong>
            </div>
        </div>
        """
    )

    if is_low_conf:
        st.warning(
            "⚠️ **Low Confidence Notice:** The highest model score is below 60%. "
            "The audio pattern may be noisy, ambiguous, or distinct from the patterns seen during training. "
            "The model is a closed-world classifier forced to choose among its 5 trained classes."
        )
    else:
        st.success(
            f"The CNN-BiLSTM model recognized acoustic features closely matching the **{pred_class}** class "
            f"with a confidence of {conf_percent:.2f}%."
        )

    # Tabs for Details
    tab_probs, tab_info, tab_model = st.tabs(["📊 Probabilities", "🩺 Clinical Pattern Context", "⚙️ Model & Features"])

    with tab_probs:
        st.write("### Predicted Probabilities Across All 5 Classes")
        st.caption(
            "Note: Softmax output scores represent relative classification scores across the five classes, "
            "not calibrated epidemiological probabilities."
        )

        # Sort classes descending by probability for readability
        sorted_indices = np.argsort(probabilities)[::-1]

        st.html(
            """
            <div class="probability-card">
                <div class="probability-title">Class Probability Breakdown</div>
            """
            + "".join(
                f"""
                <div class="probability-row">
                    <div class="probability-name">
                        <span><strong>{CLASS_NAMES[idx]}</strong>{' (Top Match)' if i == 0 else ''}</span>
                        <span><strong>{probabilities[idx] * 100.0:.2f}%</strong></span>
                    </div>
                    <div class="probability-bar">
                        <div class="probability-fill{' top-match' if i == 0 else ''}" style="width: {probabilities[idx] * 100.0:.2f}%"></div>
                    </div>
                </div>
                """
                for i, idx in enumerate(sorted_indices)
            )
            + "</div>"
        )

        # Streamlit Native Bar Chart for clear visualization
        chart_data = {CLASS_NAMES[i]: float(probabilities[i] * 100.0) for i in range(len(CLASS_NAMES))}
        st.bar_chart(chart_data, y_label="Probability (%)", color="#00cfae")

    with tab_info:
        st.write(f"### Understanding the '{pred_class}' Pattern")
        st.info(CLASS_DESCRIPTIONS.get(pred_class, "No description available."))

        st.write("#### All Diagnostic Classes in this Educational Model:")
        for c_name in CLASS_NAMES:
            with st.expander(f"📌 {c_name}"):
                st.write(CLASS_DESCRIPTIONS[c_name])

    with tab_model:
        st.write("### Model Architecture & Pipeline Parameters")
        st.markdown(
            f"""
            - **Model Type**: Sequential CNN-BiLSTM
            - **Architecture**:
              - Conv1D (64 filters, kernel size 3, ReLU) + BatchNorm + MaxPool1D(2)
              - Conv1D (128 filters, kernel size 3, ReLU) + BatchNorm + MaxPool1D(2)
              - Bidirectional LSTM (64 units)
              - Dense (64 units, ReLU) + Dropout(0.5)
              - Dense (5 units, Softmax)
            - **Total Parameters**: 160,005 (trainable)
            - **Clean Test Accuracy**: 93.41% (held-out test set)
            - **Audio Sample Rate**: {TARGET_SR} Hz (mono)
            - **Window Duration**: {TARGET_DURATION} seconds ({TARGET_SAMPLES} samples)
            - **Features**: 13 MFCCs + 128 Mel-spectrogram dB = **{COMBINED_FEATURES} total features**
            - **Feature Representation Shape**: ({TIME_STEPS}, {COMBINED_FEATURES})
            - **Scaling**: Scikit-Learn StandardScaler fitted strictly on training data
            - **Windows Evaluated**: {window_count} window(s) (Duration: {duration_sec:.2f}s)
            """
        )

# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.html(
    """
    <div class="medical-note">
        <div class="medical-note-title">⚠️ Mandatory Medical & Educational Disclaimer</div>
        This application is an educational and research prototype developed to study audio pattern recognition in respiratory sounds.
        <br><br>
        <strong>It is NOT a medical device, diagnostic system, or clinical decision tool.</strong>
        Predictions generated by this machine-learning model cannot diagnose, treat, cure, or rule out any disease or medical condition.
        Softmax scores reflect mathematical model similarities across its five trained categories and must never be interpreted as clinical confirmation.
        <br><br>
        If you or someone you know is experiencing cough, shortness of breath, chest pain, wheezing, or other respiratory symptoms,
        please seek immediate evaluation from a licensed healthcare professional or emergency medical services.
    </div>
    """
)

# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">
        🫁 Respiratory Disease AI &bull; Research & Educational Auscultation Audio Analysis<br>
        Five Target Classes: Asthma &bull; Bronchial &bull; COPD &bull; Healthy &bull; Pneumonia
    </div>
    """
)
