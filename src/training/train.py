import os
import sys
import numpy as np
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

FEATURE_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "features"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs"
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
# LOAD FEATURES
# ============================================================

print("Loading extracted features...")

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


print("\nOriginal feature shapes:")
print("MFCC:", mfcc.shape)
print("Mel:", mel.shape)
print("Labels:", labels.shape)


# ============================================================
# TRANSPOSE FEATURES
# ============================================================
#
# MFCC:
# (samples, 13, 47)
#        ↓
# (samples, 47, 13)
#
# Mel:
# (samples, 128, 47)
#        ↓
# (samples, 47, 128)
#
# Then combine:
# (samples, 47, 141)
#
# ============================================================

mfcc = np.transpose(
    mfcc,
    (0, 2, 1)
)

mel = np.transpose(
    mel,
    (0, 2, 1)
)


# ============================================================
# COMBINE MFCC + MEL
# ============================================================

X = np.concatenate(
    [mfcc, mel],
    axis=2
)

y = labels


print("\nCombined feature shape:")
print("X:", X.shape)
print("y:", y.shape)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First split:
# 70% training
# 30% temporary

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# Second split:
# Half of 30% = 15% validation
# Half of 30% = 15% test

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("\nDataset split:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# ============================================================
# FEATURE SCALING
# ============================================================
#
# The scaler is fitted ONLY on training data.
# This prevents information leakage from validation/test data.
#
# ============================================================

num_train_samples = X_train.shape[0]
num_time_steps = X_train.shape[1]
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


# Fit scaler only using training data

scaler.fit(
    X_train_2d
)


# Transform all sets

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


print("\nFeature scaling completed.")


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.unique(y_train)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight
    in zip(classes, class_weights_array)
}


print("\nClass weights:")
print(class_weights)


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding CNN-BiLSTM model...")

model = build_cnn_bilstm_model()


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)


model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(
        MODEL_PATH,
        "cnn_bilstm_best.keras"
    ),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n========================================")
print("STARTING MODEL TRAINING")
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
    "cnn_bilstm_final.keras"
)

model.save(
    final_model_path
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    OUTPUT_PATH,
    "training_history.npy"
)

np.save(
    history_path,
    history.history,
    allow_pickle=True
)


# ============================================================
# FINAL EVALUATION ON TEST SET
# ============================================================

print("\n========================================")
print("EVALUATING ON TEST SET")
print("========================================")


test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)


print("\n========================================")
print("TRAINING COMPLETED")
print("========================================")

print(
    "Test Loss:",
    test_loss
)

print(
    "Test Accuracy:",
    test_accuracy
)

print(
    "\nBest model saved at:"
)

print(
    os.path.join(
        MODEL_PATH,
        "cnn_bilstm_best.keras"
    )
)

print(
    "\nFinal model saved at:"
)

print(
    final_model_path
)