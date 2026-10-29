import os
import sys
import pickle
import numpy as np
import librosa
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
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

BASELINE_SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "feature_scaler.pkl"
)

AUGMENTED_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models_noise_augmented",
    "cnn_bilstm_noise_augmented_best.keras"
)

AUGMENTED_SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models_noise_augmented",
    "feature_scaler_noise_augmented.pkl"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "noise_augmented",
    "final_comparison"
)

os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

TARGET_SR = 4000
TARGET_SAMPLES = 4000 * 6

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

DATASET_CLASSES = [
    "asthma",
    "Bronchial",
    "copd",
    "healthy",
    "pneumonia"
]


# ============================================================
# LOAD FEATURE FILE NAMES
# ============================================================

print("\n========================================")
print("LOADING DATA")
print("========================================")

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


# Recreate EXACT same split
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

test_labels = labels[test_indices]

print("Total files:", len(labels))
print("Test files:", len(test_indices))


# ============================================================
# FIND WAV
# ============================================================

def find_wav(file_name):

    base_name = os.path.basename(
        str(file_name).replace("\\", "/")
    )

    for class_name in DATASET_CLASSES:

        path = os.path.join(
            DATASET_PATH,
            class_name,
            base_name
        )

        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Could not locate: {file_name}"
    )


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(path):

    audio, _ = librosa.load(
        path,
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
# ADD NOISE
# ============================================================

def add_noise(
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

    noisy = audio + noise

    max_value = np.max(
        np.abs(noisy)
    )

    if max_value > 0:

        noisy = (
            noisy /
            max_value
        )

    return noisy.astype(
        np.float32
    )


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_features(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=TARGET_SR,
        n_mfcc=N_MFCC
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SR,
        n_mels=N_MELS
    )

    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mfcc = mfcc.T
    mel = mel.T

    combined = np.concatenate(
        [
            mfcc,
            mel
        ],
        axis=1
    )

    return combined.astype(
        np.float32
    )


# ============================================================
# BUILD TEST DATA
# ============================================================

def build_test_features(
    snr_db=None
):

    features = []

    if snr_db is not None:

        rng = np.random.default_rng(
            RANDOM_STATE + snr_db
        )

    for count, index in enumerate(
        test_indices,
        start=1
    ):

        path = find_wav(
            file_names[index]
        )

        audio = load_audio(
            path
        )

        if snr_db is not None:

            audio = add_noise(
                audio,
                snr_db,
                rng
            )

        feature = extract_features(
            audio
        )

        features.append(
            feature
        )

        if count % 50 == 0:

            print(
                f"Processed "
                f"{count}/"
                f"{len(test_indices)}"
            )

    return np.array(
        features,
        dtype=np.float32
    )


# ============================================================
# SCALE
# ============================================================

def scale(
    X,
    scaler
):

    samples = X.shape[0]
    time_steps = X.shape[1]
    features = X.shape[2]

    X2 = X.reshape(
        -1,
        features
    )

    X2 = scaler.transform(
        X2
    )

    return X2.reshape(
        samples,
        time_steps,
        features
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

baseline_scaler = joblib.load(
    BASELINE_SCALER_PATH
)

augmented_scaler = joblib.load(
    AUGMENTED_SCALER_PATH
)

print("Both models loaded.")
print("Both scalers loaded.")


# ============================================================
# EVALUATE CONDITION
# ============================================================

def evaluate_condition(
    name,
    X
):

    print("\n========================================")
    print(name)
    print("========================================")

    baseline_X = scale(
        X,
        baseline_scaler
    )

    augmented_X = scale(
        X,
        augmented_scaler
    )

    baseline_prob = (
        baseline_model.predict(
            baseline_X,
            verbose=0
        )
    )

    augmented_prob = (
        augmented_model.predict(
            augmented_X,
            verbose=0
        )
    )

    baseline_pred = np.argmax(
        baseline_prob,
        axis=1
    )

    augmented_pred = np.argmax(
        augmented_prob,
        axis=1
    )

    baseline_accuracy = accuracy_score(
        test_labels,
        baseline_pred
    )

    augmented_accuracy = accuracy_score(
        test_labels,
        augmented_pred
    )

    print(
        f"Baseline accuracy: "
        f"{baseline_accuracy * 100:.2f}%"
    )

    print(
        f"Noise-augmented accuracy: "
        f"{augmented_accuracy * 100:.2f}%"
    )

    difference = (
        augmented_accuracy
        -
        baseline_accuracy
    )

    print(
        f"Difference: "
        f"{difference * 100:+.2f} "
        f"percentage points"
    )

    # Reports
    baseline_report = classification_report(
        test_labels,
        baseline_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    augmented_report = classification_report(
        test_labels,
        augmented_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    print("\nBASELINE REPORT")
    print(baseline_report)

    print("\nNOISE-AUGMENTED REPORT")
    print(augmented_report)

    # Confusion matrices
    baseline_cm = confusion_matrix(
        test_labels,
        baseline_pred
    )

    augmented_cm = confusion_matrix(
        test_labels,
        augmented_pred
    )

    return {
        "name": name,
        "baseline_accuracy": baseline_accuracy,
        "augmented_accuracy": augmented_accuracy,
        "difference": difference,
        "baseline_report": baseline_report,
        "augmented_report": augmented_report,
        "baseline_cm": baseline_cm,
        "augmented_cm": augmented_cm
    }


# ============================================================
# CLEAN TEST
# ============================================================

print("\n========================================")
print("BUILDING CLEAN TEST SET")
print("========================================")

X_clean = build_test_features()


clean_result = evaluate_condition(
    "CLEAN TEST",
    X_clean
)


# ============================================================
# 20 dB TEST
# ============================================================

print("\n========================================")
print("BUILDING 20 dB TEST SET")
print("========================================")

X_20db = build_test_features(
    snr_db=20
)

result_20db = evaluate_condition(
    "20 dB TEST",
    X_20db
)


# ============================================================
# 10 dB TEST
# ============================================================

print("\n========================================")
print("BUILDING 10 dB TEST SET")
print("========================================")

X_10db = build_test_features(
    snr_db=10
)

result_10db = evaluate_condition(
    "10 dB TEST",
    X_10db
)


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

import csv

csv_path = os.path.join(
    OUTPUT_PATH,
    "model_comparison.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "Condition",
            "Baseline Accuracy",
            "Noise Augmented Accuracy",
            "Difference Percentage Points"
        ]
    )

    for result in [
        clean_result,
        result_20db,
        result_10db
    ]:

        writer.writerow(
            [
                result["name"],
                result["baseline_accuracy"],
                result["augmented_accuracy"],
                result["difference"] * 100
            ]
        )


# ============================================================
# SAVE CONFUSION MATRICES
# ============================================================

np.save(
    os.path.join(
        OUTPUT_PATH,
        "clean_baseline_confusion_matrix.npy"
    ),
    clean_result["baseline_cm"]
)

np.save(
    os.path.join(
        OUTPUT_PATH,
        "clean_augmented_confusion_matrix.npy"
    ),
    clean_result["augmented_cm"]
)

np.save(
    os.path.join(
        OUTPUT_PATH,
        "20db_baseline_confusion_matrix.npy"
    ),
    result_20db["baseline_cm"]
)

np.save(
    os.path.join(
        OUTPUT_PATH,
        "20db_augmented_confusion_matrix.npy"
    ),
    result_20db["augmented_cm"]
)

np.save(
    os.path.join(
        OUTPUT_PATH,
        "10db_baseline_confusion_matrix.npy"
    ),
    result_10db["baseline_cm"]
)

np.save(
    os.path.join(
        OUTPUT_PATH,
        "10db_augmented_confusion_matrix.npy"
    ),
    result_10db["augmented_cm"]
)


# ============================================================
# SAVE FULL REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_PATH,
    "final_noise_comparison_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FINAL NOISE ROBUSTNESS COMPARISON\n"
    )

    f.write(
        "========================================\n\n"
    )

    for result in [
        clean_result,
        result_20db,
        result_10db
    ]:

        f.write(
            f"\n{result['name']}\n"
        )

        f.write(
            "----------------------------------------\n"
        )

        f.write(
            f"Baseline accuracy: "
            f"{result['baseline_accuracy'] * 100:.2f}%\n"
        )

        f.write(
            f"Noise-augmented accuracy: "
            f"{result['augmented_accuracy'] * 100:.2f}%\n"
        )

        f.write(
            f"Difference: "
            f"{result['difference'] * 100:+.2f} "
            f"percentage points\n\n"
        )

        f.write(
            "Baseline classification report:\n"
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

        f.write(
            "\n\nBaseline confusion matrix:\n"
        )

        f.write(
            str(
                result["baseline_cm"]
            )
        )

        f.write(
            "\n\nNoise-augmented confusion matrix:\n"
        )

        f.write(
            str(
                result["augmented_cm"]
            )
        )

        f.write(
            "\n\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n\n========================================")
print("FINAL COMPARISON")
print("========================================")

for result in [
    clean_result,
    result_20db,
    result_10db
]:

    print(
        f"\n{result['name']}"
    )

    print(
        f"Baseline: "
        f"{result['baseline_accuracy'] * 100:.2f}%"
    )

    print(
        f"Noise-Augmented: "
        f"{result['augmented_accuracy'] * 100:.2f}%"
    )

    print(
        f"Difference: "
        f"{result['difference'] * 100:+.2f} pp"
    )


print("\nFiles saved to:")
print(OUTPUT_PATH)