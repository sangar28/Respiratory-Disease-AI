# 🫁 Respiratory Disease AI

An educational research application for AI-assisted multi-class classification of respiratory auscultation sounds using deep learning (CNN-BiLSTM).

---

## 📋 1. Project Overview

**Respiratory Disease AI** analyzes respiratory sound recordings (`.wav` audio files) to identify acoustic patterns associated with five target classes:

1. **Asthma**
2. **Bronchial**
3. **COPD** (Chronic Obstructive Pulmonary Disease)
4. **Healthy**
5. **Pneumonia**

The core classification engine uses a **Convolutional Neural Network combined with Bidirectional LSTM (CNN-BiLSTM)** trained on auscultation features:
- **Baseline Model Clean Test Accuracy:** **93.41%** (held-out test set, 182 audio files)
- **Model Parameters:** **160,005** trainable parameters
- **Target Sampling Rate:** 4,000 Hz (mono)
- **Audio Duration:** 6.0 seconds (24,000 samples)
- **Features Extracted:** 13 MFCCs + 128 Mel-spectrogram bins in decibels = **141 combined features**
- **Model Input Shape:** `(47, 141)` (47 time steps &times; 141 features)

---

## 🗂️ 2. Repository Structure

```text
Respiratory-Disease-AI/
├── dataset/                            # Respiratory audio recordings (.wav)
│   ├── asthma/                         # Asthma breath recordings
│   ├── Bronchial/                      # Bronchial breath recordings
│   ├── copd/                           # COPD breath recordings
│   ├── healthy/                        # Healthy breath recordings
│   └── pneumonia/                      # Pneumonia breath recordings
├── notebooks/                          # Exploratory analysis notebooks
├── outputs/                            # Training & evaluation artifacts
│   ├── features/                       # Baseline extracted feature arrays (.npy, .pkl)
│   ├── features_original/              # Clean baseline feature backups
│   ├── features_noise_reduced/         # Noise-reduced feature experiment
│   ├── figures/                        # Confusion matrices and plots
│   ├── noise_augmented/                # Synthetic noise experiment evaluations
│   └── predictions/                    # Test metrics and CSV predictions
├── saved_models/                       # Active baseline production model and scaler
│   ├── cnn_bilstm_best.keras           # 160,005-param CNN-BiLSTM baseline model (93.41% acc)
│   ├── cnn_bilstm_final.keras          # Final epoch baseline checkpoint
│   └── feature_scaler.pkl              # Fitted StandardScaler (141 features)
├── saved_models_noise_augmented/       # Experimental noise-augmented models
├── saved_models_original/              # Original baseline model backup
├── src/                                # Core source code modules
│   ├── evaluation/                     # Model evaluation and noise robustness scripts
│   ├── feature_extraction/             # Dataset feature extraction script
│   ├── models/                         # Model architecture definition (cnn_bilstm.py)
│   ├── preprocessing/                  # Audio loading, normalization & feature pipeline
│   ├── training/                       # Model training scripts
│   └── utils/                          # Helper utilities
├── tests/                              # Automated test suite
│   └── test_pipeline.py                # 9-point pipeline and Streamlit simulation tests
├── web_app/                            # Streamlit web user interface
│   └── app.py                          # Streamlit application
├── requirements.txt                    # Project Python dependencies (UTF-8 encoded)
└── README.md                           # Documentation and instructions
```

---

## 🐍 3. Prerequisites & Environment Setup

- **Python Version:** Python **3.10 - 3.13** (Verified on Python 3.13.3).
- **Operating System:** Windows 10/11, Linux, or macOS.

### Windows PowerShell Setup

Open **Windows PowerShell** and navigate to the project directory:

```powershell
cd E:\Respiratory-Disease-AI
```

If PowerShell restricts script execution, temporarily allow local scripts for your current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*(If `.venv` already exists in your workspace, simply activate it with `.\.venv\Scripts\Activate.ps1`)*

---

## 📦 4. Dependency Installation

Install all required dependencies using `pip`:

```powershell
pip install -r requirements.txt
```

Key packages included:
- `tensorflow >= 2.16` (Deep learning framework)
- `librosa >= 0.10` (Audio processing and feature extraction)
- `soundfile >= 0.12` (WAV audio read/write)
- `scikit-learn >= 1.4` (StandardScaler and evaluation metrics)
- `streamlit >= 1.35` (Interactive web application interface)
- `noisereduce` & `scipy` (Signal processing utilities)

---

## 🚀 5. How to Launch the Web Application

With your virtual environment activated, run:

```powershell
streamlit run web_app/app.py
```

Streamlit will launch the web server and open the application in your default browser:

```text
Local URL: http://localhost:8501
Network URL: http://<your-ip>:8501
```

### Application Features:
1. **Audio Upload:** Accepts any standard `.wav` respiratory audio recording.
2. **Metadata & Playback:** Displays audio duration (seconds), file size, and provides an in-browser audio player.
3. **Analysis Modes:**
   - **Baseline Standard (6s) [Recommended]:** Exactly matches the 93.41% accuracy baseline evaluation pipeline (first 6 seconds, padded if shorter).
   - **Multi-Window Average:** Evaluates overlapping 6-second segments across recordings longer than 6 seconds and averages class probabilities.
4. **Validation & Quality Checks:** Detects silent audio, corrupted files, and zero-byte uploads with user-friendly notices.
5. **Prediction Card & Probabilities:** Highlights predicted class, confidence percentage, and visual breakdown of all five classes sorted by score.
6. **Clinical Context & Educational Information:** Explains acoustic hallmarks of each condition.
7. **Reset Workflow:** One-click clear and reset button to upload new files.

---

## 🧪 6. How to Run the Automated Test Suite

A comprehensive 9-test verification suite is provided in `tests/test_pipeline.py`. To run all tests:

```powershell
python tests/test_pipeline.py
```

### Tests Executed:
1. **Baseline Model Loading:** Verifies `saved_models/cnn_bilstm_best.keras` loads with 160,005 parameters and shapes `(None, 47, 141) -> (None, 5)`.
2. **Baseline Scaler Loading:** Verifies `saved_models/feature_scaler.pkl` loads with `n_features_in_ = 141`.
3. **Feature Preprocessing Shape:** Verifies dataset audio preprocessing produces exact `(47, 141)` shape with finite values (no NaNs/Infs).
4. **End-to-End Prediction Verification:** Tests known dataset sample `P44Pneumonia36U.wav` and verifies exact match with ground-truth evaluation predictions (Pneumonia, 94.43% confidence, probabilities sum to 1.0).
5. **Five-Class Verification:** Preprocesses and predicts sample audio from all 5 dataset folders.
6. **Edge Cases:** Validates silence detection, short audio padding, and stereo-to-mono conversion.
7. **Multi-Window Mode:** Validates overlapping sliding window generation on recordings > 6s.
8. **Artifact Integrity:** Guarantees that tests never overwrite or corrupt the trained models or scalers.
9. **Streamlit UI Simulation:** In-memory simulation of user upload, button click, and prediction output in Streamlit.

---

## 📊 7. Model Performance & Experimental Benchmarks

| Experiment | Model File | Scaler File | Clean Test Accuracy | 20 dB White Noise | 10 dB White Noise |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Baseline CNN-BiLSTM (Default)** | `saved_models/cnn_bilstm_best.keras` | `saved_models/feature_scaler.pkl` | **93.41%** | 32.97% | 32.97% |
| **Noise-Augmented Model** | `saved_models_noise_augmented/...` | `..._noise_augmented.pkl` | **89.56%** | **75.27%** | **64.29%** |

*Note: The web application defaults to the **Baseline CNN-BiLSTM** model and matching scaler to provide the established 93.41% accuracy on clean respiratory audio.*

---

## 🔧 8. Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `Execution of scripts is disabled on this system` | PowerShell execution policy restriction | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating `.venv`. |
| `Model not found: saved_models/cnn_bilstm_best.keras` | Application run from incorrect directory | Ensure you run commands from the project root (`E:\Respiratory-Disease-AI`). |
| `The uploaded audio recording is silent` | File has peak amplitude near zero | Upload a recording with audible breathing or lung sounds. |
| TensorFlow `oneDNN custom operations` notices | Normal CPU optimization message from TensorFlow | These informational logs can be silenced by setting environment variable `TF_ENABLE_ONEDNN_OPTS=0`. |

---

## ⚠️ 9. Medical & Educational Disclaimer

> [!WARNING]
> **This software is an educational and academic research prototype.**
> 
> - **Not a Medical Device:** This application has NOT been cleared, certified, or evaluated by the FDA, CE, or any health regulatory authority.
> - **No Medical Advice:** Predictions generated by the CNN-BiLSTM model are experimental and cannot diagnose, treat, prevent, or rule out respiratory conditions or diseases.
> - **Closed-World Classifier:** The model is constrained to distribute probability strictly among its five trained classes (Asthma, Bronchial, COPD, Healthy, Pneumonia). Unrelated sounds, speech, ambient noise, or other lung conditions will still be mapped to one of these five classes.
> - **Always Consult a Doctor:** If you or anyone you know experiences shortness of breath, persistent cough, chest pain, wheezing, or other medical symptoms, seek immediate evaluation from a licensed physician or emergency medical care.
