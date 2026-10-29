import os
import sys
import pickle
import numpy as np
import librosa
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset"
)

FEATURE_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "features"
)

BASELINE_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "cnn_bilstm_best.keras"
)

AUGMENTED_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models_noise_augmented",
    "cnn_bilstm_noise_augmented_best.keras"
)

BASELINE_SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "feature_scaler.pkl"
)

AUGMENTED_SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models_noise_augmented",
    "feature_scaler_noise_augmented.pkl"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "noise_augmented"
)

os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

TARGET_SR = 4000
TARGET_DURATION = 6
TARGET_SAMPLES = TARGET_SR * TARGET_DURATION

N_MFCC = 13
N_MELS = 128

RANDOM_STATE = 42

CLASS_NAMES = [
    "Asthma",
    "Bronchial",
    "COPD",
    "Healthy",
    "Pneumonia"
]

DATASET_CLASS_NAMES = [
    "asthma",
    "Bronchial",
    "copd",
    "healthy",
    "pneumonia"
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# LOAD ORIGINAL FEATURES
# ============================================================

print("\n========================================")
print("LOADING ORIGINAL FEATURES")
print("========================================")

mfcc = np.load(
    os.path.join(
        FEATURE_PATH,
        "mfcc_features.npy"
    )
)

mel = np.load(
    os.path.join(
        FEATURE_PATH,
        "mel_features.npy"
    )
)

labels = np.load(
    os.path.join(
        FEATURE_PATH,
        "labels.npy"
    )
)

with open(
    os.path.join(
        FEATURE_PATH,
        "file_names.pkl"
    ),
    "rb"
) as f:
    file_names = pickle.load(f)


print("MFCC:", mfcc.shape)
print("Mel:", mel.shape)
print("Labels:", labels.shape)
print("Files:", len(file_names))


# ============================================================
# CREATE SAME TRAIN / VALIDATION / TEST SPLIT
# ============================================================

indices = np.arange(
    len(labels)
)

train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=labels
)

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=labels[temp_indices]
)


print("\n========================================")
print("ORIGINAL SPLIT")
print("========================================")

print("Training:", len(train_indices))
print("Validation:", len(val_indices))
print("Testing:", len(test_indices))


# ============================================================
# HELPER: FIND WAV FILE
# ============================================================

def get_wav_path(file_name):

    base_name = os.path.basename(
        str(file_name).replace("\\", "/")
    )

    for class_name in DATASET_CLASS_NAMES:

        candidate = os.path.join(
            DATASET_PATH,
            class_name,
            base_name
        )

        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"WAV file not found: {file_name}"
    )


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(file_path):

    audio, _ = librosa.load(
        file_path,
        sr=TARGET_SR,
        mono=True
    )

    if len(audio) < TARGET_SAMPLES:

        audio = np.pad(
            audio,
            (
                0,
                TARGET_SAMPLES - len(audio)
            ),
            mode="constant"
        )

    else:

        audio = audio[:TARGET_SAMPLES]

    max_value = np.max(
        np.abs(audio)
    )

    if max_value > 0:

        audio = (
            audio /
            max_value
        )

    return audio.astype(
        np.float32
    )


# ============================================================
# ADD DETERMINISTIC WHITE NOISE
# ============================================================

def add_white_noise(
    audio,
    snr_db,
    rng
):

    signal_power = np.mean(
        audio ** 2
    )

    if signal_power <= 1e-12:

        return audio.copy()

    noise_power = (
        signal_power /
        (10 ** (snr_db / 10))
    )

    noise = rng.normal(
        0,
        np.sqrt(noise_power),
        size=audio.shape
    )

    noisy_audio = (
        audio +
        noise
    )

    # Normalize exactly as the
    # augmentation experiment did.

    max_value = np.max(
        np.abs(noisy_audio)
    )

    if max_value > 0:

        noisy_audio = (
            noisy_audio /
            max_value
        )

    return noisy_audio.astype(
        np.float32
    )


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_features(audio):

    mfcc_feature = librosa.feature.mfcc(
        y=audio,
        sr=TARGET_SR,
        n_mfcc=N_MFCC
    )

    mel_feature = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SR,
        n_mels=N_MELS
    )

    mel_feature = librosa.power_to_db(
        mel_feature,
        ref=np.max
    )

    mfcc_feature = mfcc_feature.T
    mel_feature = mel_feature.T

    combined = np.concatenate(
        [
            mfcc_feature,
            mel_feature
        ],
        axis=1
    )

    return combined.astype(
        np.float32
    )


# ============================================================
# LOAD MODELS
# ============================================================

print("\n========================================")
print("LOADING MODELS")
print("========================================")

baseline_model = tf.keras.models.load_model(
    BASELINE_MODEL_PATH
)

augmented_model = tf.keras.models.load_model(
    AUGMENTED_MODEL_PATH
)


print("Baseline model loaded.")
print("Noise-augmented model loaded.")


# ============================================================
# LOAD SCALERS
# ============================================================

print("\n========================================")
print("LOADING SCALERS")
print("========================================")

import joblib

baseline_scaler = joblib.load(
    BASELINE_SCALER_PATH
)

augmented_scaler = joblib.load(
    AUGMENTED_SCALER_PATH
)


print("Baseline scaler loaded.")
print("Noise-augmented scaler loaded.")


# ============================================================
# BUILD NOISY TEST SET
# ============================================================

def build_noisy_test_set(
    snr_db
):

    noisy_features = []
    test_labels = []

    rng = np.random.default_rng(
        RANDOM_STATE + snr_db
    )

    print(
        f"\nCreating {snr_db} dB noisy test set..."
    )

    for count, index in enumerate(
        test_indices,
        start=1
    ):

        wav_path = get_wav_path(
            file_names[index]
        )

        audio = load_audio(
            wav_path
        )

        noisy_audio = add_white_noise(
            audio,
            snr_db,
            rng
        )

        features = extract_features(
            noisy_audio
        )

        noisy_features.append(
            features
        )

        test_labels.append(
            labels[index]
        )

        if count % 50 == 0:

            print(
                f"Processed "
                f"{count}/"
                f"{len(test_indices)}"
            )

    return (
        np.array(
            noisy_features,
            dtype=np.float32
        ),
        np.array(
            test_labels,
            dtype=np.int64
        )
    )


# ============================================================
# SCALE FEATURES
# ============================================================

def scale_features(
    features,
    scaler
):

    samples = features.shape[0]
    time_steps = features.shape[1]
    num_features = features.shape[2]

    features_2d = features.reshape(
        -1,
        num_features
    )

    scaled = scaler.transform(
        features_2d
    )

    return scaled.reshape(
        samples,
        time_steps,
        num_features
    )


# ============================================================
# EVALUATE ONE SNR LEVEL
# ============================================================

def evaluate_snr(
    snr_db
):

    noisy_X, noisy_y = (
        build_noisy_test_set(
            snr_db
        )
    )

    print(
        "\nNoisy feature shape:",
        noisy_X.shape
    )

    # Each model uses its own scaler
    baseline_X = scale_features(
        noisy_X,
        baseline_scaler
    )

    augmented_X = scale_features(
        noisy_X,
        augmented_scaler
    )

    print(
        f"\nEvaluating baseline at "
        f"{snr_db} dB..."
    )

    baseline_probabilities = (
        baseline_model.predict(
            baseline_X,
            verbose=0
        )
    )

    baseline_predictions = np.argmax(
        baseline_probabilities,
        axis=1
    )

    baseline_accuracy = (
        accuracy_score(
            noisy_y,
            baseline_predictions
        )
    )

    print(
        "Baseline accuracy:",
        baseline_accuracy
    )

    print(
        f"\nEvaluating noise-augmented "
        f"model at {snr_db} dB..."
    )

    augmented_probabilities = (
        augmented_model.predict(
            augmented_X,
            verbose=0
        )
    )

    augmented_predictions = np.argmax(
        augmented_probabilities,
        axis=1
    )

    augmented_accuracy = (
        accuracy_score(
            noisy_y,
            augmented_predictions
        )
    )

    print(
        "Noise-augmented accuracy:",
        augmented_accuracy
    )

    # Classification reports
    baseline_report = classification_report(
        noisy_y,
        baseline_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    augmented_report = classification_report(
        noisy_y,
        augmented_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    return {
        "snr_db": snr_db,
        "baseline_accuracy": baseline_accuracy,
        "augmented_accuracy": augmented_accuracy,
        "baseline_report": baseline_report,
        "augmented_report": augmented_report
    }


# ============================================================
# RUN EXPERIMENT
# ============================================================

results = []

for snr in [20, 10]:

    result = evaluate_snr(
        snr
    )

    results.append(
        result
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    OUTPUT_PATH,
    "noise_robustness_evaluation.txt"
)


with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "NOISE ROBUSTNESS EVALUATION\n"
    )

    f.write(
        "========================================\n\n"
    )

    f.write(
        "Dataset: 5 original classes\n"
    )

    f.write(
        "Test samples per SNR: 182\n"
    )

    f.write(
        "Random state: 42\n\n"
    )

    for result in results:

        f.write(
            f"\nSNR: "
            f"{result['snr_db']} dB\n"
        )

        f.write(
            "----------------------------------------\n"
        )

        f.write(
            f"Baseline accuracy: "
            f"{result['baseline_accuracy']:.4f}\n"
        )

        f.write(
            f"Noise-augmented accuracy: "
            f"{result['augmented_accuracy']:.4f}\n"
        )

        f.write(
            "\nBaseline classification report:\n"
        )

        f.write(
            result["baseline_report"]
        )

        f.write(
            "\nNoise-augmented classification report:\n"
        )

        f.write(
            result["augmented_report"]
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("NOISE ROBUSTNESS SUMMARY")
print("========================================")

for result in results:

    print(
        f"\nSNR: {result['snr_db']} dB"
    )

    print(
        f"Baseline: "
        f"{result['baseline_accuracy'] * 100:.2f}%"
    )

    print(
        f"Noise-augmented: "
        f"{result['augmented_accuracy'] * 100:.2f}%"
    )

    difference = (
        result["augmented_accuracy"]
        -
        result["baseline_accuracy"]
    )

    print(
        f"Difference: "
        f"{difference * 100:+.2f} percentage points"
    )


print(
    "\nResults saved to:"
)

print(
    results_file
)
