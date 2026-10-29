import os
import sys
import pickle
import numpy as np
import librosa
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


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

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models_noise_augmented"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "noise_augmented"
)

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)


# ============================================================
# IMPORT MODEL
# ============================================================

SRC_PATH = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from models.cnn_bilstm import build_cnn_bilstm_model


# ============================================================
# AUDIO SETTINGS
# ============================================================

TARGET_SR = 4000
TARGET_DURATION = 6
TARGET_SAMPLES = TARGET_SR * TARGET_DURATION

N_MFCC = 13
N_MELS = 128

CLASS_NAMES = [
    "asthma",
    "Bronchial",
    "copd",
    "healthy",
    "pneumonia"
]


# ============================================================
# RANDOM SEEDS
# ============================================================

RANDOM_STATE = 42

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
print("File names:", len(file_names))


# ============================================================
# HELPER: BUILD FULL WAV PATH
# ============================================================

def get_wav_path(file_name):
    """
    Convert a stored file name into the actual dataset WAV path.

    The feature extractor stores file names while processing
    class folders, so this function searches the five allowed
    class directories.
    """

    # If the stored name already contains a class folder
    normalized = str(file_name).replace("\\", "/")

    for class_name in CLASS_NAMES:

        candidate = os.path.join(
            DATASET_PATH,
            class_name,
            os.path.basename(normalized)
        )

        if os.path.exists(candidate):
            return candidate

    # Fallback: search all five folders
    base_name = os.path.basename(normalized)

    for class_name in CLASS_NAMES:

        class_folder = os.path.join(
            DATASET_PATH,
            class_name
        )

        candidate = os.path.join(
            class_folder,
            base_name
        )

        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find WAV file: {file_name}"
    )


# ============================================================
# ORIGINAL AUDIO PREPROCESSING
# ============================================================

def load_original_audio(file_path):
    """
    Load audio exactly according to the original baseline
    preprocessing:

        WAV
        ↓
        mono
        ↓
        resample to 4000 Hz
        ↓
        6 seconds
        ↓
        normalize
    """

    audio, _ = librosa.load(
        file_path,
        sr=TARGET_SR,
        mono=True
    )

    # Force exactly 6 seconds
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

    # Normalize
    max_value = np.max(
        np.abs(audio)
    )

    if max_value > 0:
        audio = audio / max_value

    return audio.astype(np.float32)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features_from_audio(audio):

    # MFCC
    mfcc_feature = librosa.feature.mfcc(
        y=audio,
        sr=TARGET_SR,
        n_mfcc=N_MFCC
    )

    # Mel spectrogram
    mel_feature = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SR,
        n_mels=N_MELS
    )

    mel_feature = librosa.power_to_db(
        mel_feature,
        ref=np.max
    )

    return (
        mfcc_feature.astype(np.float32),
        mel_feature.astype(np.float32)
    )


# ============================================================
# AUDIO NOISE AUGMENTATION
# ============================================================

def add_white_noise(audio, snr_db):

    signal_power = np.mean(
        audio ** 2
    )

    if signal_power <= 1e-12:
        return audio.copy()

    noise_power = (
        signal_power /
        (10 ** (snr_db / 10))
    )

    noise = np.random.normal(
        0,
        np.sqrt(noise_power),
        size=audio.shape
    )

    noisy_audio = audio + noise

    return noisy_audio.astype(np.float32)


# ============================================================
# PROCESS ONE AUDIO FILE WITH NOISE
# ============================================================

def create_noisy_features(file_path, snr_db):

    # Load original audio
    audio = load_original_audio(
        file_path
    )

    # Add artificial external noise
    noisy_audio = add_white_noise(
        audio,
        snr_db
    )

    # Normalize after noise addition
    max_value = np.max(
        np.abs(noisy_audio)
    )

    if max_value > 0:
        noisy_audio = (
            noisy_audio /
            max_value
        )

    return extract_features_from_audio(
        noisy_audio
    )


# ============================================================
# TRANSPOSE + COMBINE FEATURES
# ============================================================

def combine_features(
    mfcc_array,
    mel_array
):

    mfcc_array = np.transpose(
        mfcc_array,
        (0, 2, 1)
    )

    mel_array = np.transpose(
        mel_array,
        (0, 2, 1)
    )

    return np.concatenate(
        [
            mfcc_array,
            mel_array
        ],
        axis=2
    )


# ============================================================
# ORIGINAL CLEAN FEATURES
# ============================================================

X_original = combine_features(
    mfcc,
    mel
)

y = labels


print("\nOriginal combined features:")
print("X:", X_original.shape)
print("y:", y.shape)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

indices = np.arange(
    len(y)
)

train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y[temp_indices]
)


print("\n========================================")
print("DATASET SPLIT")
print("========================================")

print(
    "Training:",
    len(train_indices)
)

print(
    "Validation:",
    len(val_indices)
)

print(
    "Testing:",
    len(test_indices)
)


# ============================================================
# CLEAN TRAINING DATA
# ============================================================

X_train_clean = X_original[
    train_indices
]

y_train_clean = y[
    train_indices
]


# ============================================================
# CREATE NOISY TRAINING DATA
# ============================================================

print("\n========================================")
print("CREATING NOISY TRAINING DATA")
print("========================================")

noisy_mfcc = []
noisy_mel = []
noisy_labels = []

for count, index in enumerate(
    train_indices,
    start=1
):

    file_name = file_names[index]

    wav_path = get_wav_path(
        file_name
    )

    # Alternate between 20 dB and 10 dB
    # for deterministic augmentation.
    if count % 2 == 0:
        snr_db = 20
    else:
        snr_db = 10

    mfcc_noisy, mel_noisy = (
        create_noisy_features(
            wav_path,
            snr_db
        )
    )

    noisy_mfcc.append(
        mfcc_noisy
    )

    noisy_mel.append(
        mel_noisy
    )

    noisy_labels.append(
        y[index]
    )

    if count % 100 == 0:

        print(
            f"Processed {count}/"
            f"{len(train_indices)} "
            f"training files"
        )


noisy_mfcc = np.array(
    noisy_mfcc,
    dtype=np.float32
)

noisy_mel = np.array(
    noisy_mel,
    dtype=np.float32
)

noisy_labels = np.array(
    noisy_labels,
    dtype=np.int64
)


X_train_noisy = combine_features(
    noisy_mfcc,
    noisy_mel
)


print("\nNoisy training features:")
print(
    "X_train_noisy:",
    X_train_noisy.shape
)


# ============================================================
# COMBINE CLEAN + NOISY TRAINING DATA
# ============================================================

X_train = np.concatenate(
    [
        X_train_clean,
        X_train_noisy
    ],
    axis=0
)

y_train = np.concatenate(
    [
        y_train_clean,
        noisy_labels
    ],
    axis=0
)


print("\nAugmented training dataset:")
print(
    "X_train:",
    X_train.shape
)

print(
    "y_train:",
    y_train.shape
)


# ============================================================
# CLEAN VALIDATION / TEST DATA
# ============================================================

X_val = X_original[
    val_indices
]

y_val = y[
    val_indices
]

X_test = X_original[
    test_indices
]

y_test = y[
    test_indices
]


print("\nValidation:")
print(X_val.shape)

print("\nClean test:")
print(X_test.shape)


# ============================================================
# FEATURE SCALING
# ============================================================

print("\n========================================")
print("FEATURE SCALING")
print("========================================")

num_features = X_train.shape[2]

scaler = StandardScaler()

X_train_2d = X_train.reshape(
    -1,
    num_features
)

X_val_2d = X_val.reshape(
    -1,
    num_features
)

X_test_2d = X_test.reshape(
    -1,
    num_features
)


# IMPORTANT:
# Fit scaler ONLY on training data.

scaler.fit(
    X_train_2d
)


X_train = scaler.transform(
    X_train_2d
).reshape(
    X_train.shape
)

X_val = scaler.transform(
    X_val_2d
).reshape(
    X_val.shape
)

X_test = scaler.transform(
    X_test_2d
).reshape(
    X_test.shape
)


print("Feature scaling completed.")


# ============================================================
# SAVE EXPERIMENT SCALER
# ============================================================

scaler_path = os.path.join(
    MODEL_PATH,
    "feature_scaler_noise_augmented.pkl"
)

import joblib

joblib.dump(
    scaler,
    scaler_path
)

print(
    "Scaler saved:",
    scaler_path
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.unique(
    y_train
)

class_weights_array = (
    compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight
    in zip(
        classes,
        class_weights_array
    )
}


print("\nClass weights:")

print(
    class_weights
)


# ============================================================
# BUILD MODEL
# ============================================================

print("\n========================================")
print("BUILDING CNN-BiLSTM")
print("========================================")

model = build_cnn_bilstm_model()

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = (
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
)


model_checkpoint = (
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            MODEL_PATH,
            "cnn_bilstm_noise_augmented_best.keras"
        ),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
)


reduce_learning_rate = (
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
)


# ============================================================
# TRAIN
# ============================================================

print("\n========================================")
print("STARTING NOISE-AUGMENTED TRAINING")
print("========================================")

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=50,

    batch_size=32,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        model_checkpoint,
        reduce_learning_rate
    ],

    verbose=1
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_PATH,
    "cnn_bilstm_noise_augmented_final.keras"
)

model.save(
    final_model_path
)


# ============================================================
# SAVE HISTORY
# ============================================================

history_path = os.path.join(
    OUTPUT_PATH,
    "training_history_noise_augmented.npy"
)

np.save(
    history_path,
    history.history,
    allow_pickle=True
)


# ============================================================
# CLEAN TEST EVALUATION
# ============================================================

print("\n========================================")
print("CLEAN TEST EVALUATION")
print("========================================")

clean_test_loss, clean_test_accuracy = (
    model.evaluate(
        X_test,
        y_test,
        verbose=1
    )
)


print(
    "\nClean Test Loss:",
    clean_test_loss
)

print(
    "Clean Test Accuracy:",
    clean_test_accuracy
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = os.path.join(
    OUTPUT_PATH,
    "noise_augmentation_results.txt"
)

with open(
    results_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Noise Augmentation Experiment\n"
    )

    f.write(
        "================================\n\n"
    )

    f.write(
        f"Training samples before augmentation: "
        f"{len(train_indices)}\n"
    )

    f.write(
        f"Training samples after augmentation: "
        f"{len(y_train)}\n"
    )

    f.write(
        "Noise SNR levels: 10 dB and 20 dB\n"
    )

    f.write(
        f"Clean test loss: "
        f"{clean_test_loss}\n"
    )

    f.write(
        f"Clean test accuracy: "
        f"{clean_test_accuracy}\n"
    )


print("\n========================================")
print("EXPERIMENT COMPLETED")
print("========================================")

print(
    "Best model:",
    os.path.join(
        MODEL_PATH,
        "cnn_bilstm_noise_augmented_best.keras"
    )
)

print(
    "Final model:",
    final_model_path
)

print(
    "Results:",
    results_path
)