import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. PROJECT PATHS
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

FIGURE_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "figures"
)

PREDICTION_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "predictions"
)


os.makedirs(FIGURE_PATH, exist_ok=True)
os.makedirs(PREDICTION_PATH, exist_ok=True)


# ============================================================
# 2. CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Asthma",
    "Bronchial",
    "COPD",
    "Healthy",
    "Pneumonia"
]


# ============================================================
# 3. LOAD FEATURES
# ============================================================

print("========================================")
print("LOADING FEATURES")
print("========================================")

mfcc = np.load(
    os.path.join(FEATURE_PATH, "mfcc_features.npy")
)

mel = np.load(
    os.path.join(FEATURE_PATH, "mel_features.npy")
)

labels = np.load(
    os.path.join(FEATURE_PATH, "labels.npy")
)

with open(
    os.path.join(FEATURE_PATH, "file_names.pkl"),
    "rb"
) as file:
    file_names = pickle.load(file)


print("MFCC shape:", mfcc.shape)
print("Mel Spectrogram shape:", mel.shape)
print("Labels shape:", labels.shape)


# ============================================================
# 4. PREPARE FEATURES
# ============================================================

print("\n========================================")
print("PREPARING FEATURES")
print("========================================")

# MFCC:
# (samples, 13, 47)
# becomes:
# (samples, 47, 13)

mfcc = np.transpose(
    mfcc,
    (0, 2, 1)
)

# Mel:
# (samples, 128, 47)
# becomes:
# (samples, 47, 128)

mel = np.transpose(
    mel,
    (0, 2, 1)
)

# Combine MFCC + Mel Spectrogram
#
# 13 MFCC features
# +
# 128 Mel features
# =
# 141 features

X = np.concatenate(
    [mfcc, mel],
    axis=2
)

y = labels


print("Combined feature shape:", X.shape)
print("Labels shape:", y.shape)


# ============================================================
# 5. RECREATE THE SAME TRAIN/VALIDATION/TEST SPLIT
# ============================================================

print("\n========================================")
print("RECREATING DATASET SPLIT")
print("========================================")

# Create indices so we can also recover filenames
indices = np.arange(len(y))


# First split:
# 70% training
# 30% temporary

train_indices, temp_indices, y_train, y_temp = train_test_split(
    indices,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# Second split:
# 15% validation
# 15% testing

val_indices, test_indices, y_val, y_test = train_test_split(
    temp_indices,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


X_train = X[train_indices]
X_val = X[val_indices]
X_test = X[test_indices]


print("Training samples:", X_train.shape)
print("Validation samples:", X_val.shape)
print("Testing samples:", X_test.shape)


# ============================================================
# 6. APPLY THE SAME FEATURE SCALING
# ============================================================

print("\n========================================")
print("APPLYING FEATURE SCALING")
print("========================================")

num_features = X_train.shape[2]

scaler = StandardScaler()


# Convert 3D data into 2D
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
# Fit scaler ONLY on training data

scaler.fit(X_train_2d)


X_train = scaler.transform(
    X_train_2d
).reshape(X_train.shape)


X_val = scaler.transform(
    X_val_2d
).reshape(X_val.shape)


X_test = scaler.transform(
    X_test_2d
).reshape(X_test.shape)


print("Feature scaling completed.")


# ============================================================
# 7. SAVE SCALER FOR FUTURE STREAMLIT APP
# ============================================================

scaler_path = os.path.join(
    MODEL_PATH,
    "feature_scaler.pkl"
)

with open(
    scaler_path,
    "wb"
) as file:
    pickle.dump(scaler, file)


print("Scaler saved at:")
print(scaler_path)


# ============================================================
# 8. LOAD BEST TRAINED MODEL
# ============================================================

print("\n========================================")
print("LOADING TRAINED MODEL")
print("========================================")

best_model_path = os.path.join(
    MODEL_PATH,
    "cnn_bilstm_best.keras"
)


if not os.path.exists(best_model_path):
    raise FileNotFoundError(
        f"Best model not found:\n{best_model_path}"
    )


model = tf.keras.models.load_model(
    best_model_path
)


print("Model loaded successfully.")
print("Model path:")
print(best_model_path)


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

print("\n========================================")
print("MAKING TEST PREDICTIONS")
print("========================================")

probabilities = model.predict(
    X_test,
    verbose=1
)


# Get class with highest probability

y_pred = np.argmax(
    probabilities,
    axis=1
)


# Confidence = highest softmax probability

confidence = np.max(
    probabilities,
    axis=1
)


print("Predictions generated successfully.")


# ============================================================
# 10. CALCULATE ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n========================================")
print("TEST ACCURACY")
print("========================================")

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

report = classification_report(
    y_test,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print(report)


# Save classification report

report_path = os.path.join(
    PREDICTION_PATH,
    "classification_report.txt"
)

with open(
    report_path,
    "w"
) as file:

    file.write(
        "AI-Based Multi-Class Respiratory Disease Detection\n"
    )

    file.write(
        "CNN-BiLSTM Evaluation Report\n\n"
    )

    file.write(
        f"Test Accuracy: {accuracy * 100:.2f}%\n\n"
    )

    file.write(report)


print("Classification report saved at:")
print(report_path)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

print("\n========================================")
print("CREATING CONFUSION MATRIX")
print("========================================")

cm = confusion_matrix(
    y_test,
    y_pred
)


print("Confusion Matrix:")
print(cm)


fig, ax = plt.subplots(
    figsize=(8, 6)
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)


disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)


plt.title(
    "Confusion Matrix - CNN-BiLSTM"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

plt.tight_layout()


confusion_matrix_path = os.path.join(
    FIGURE_PATH,
    "confusion_matrix.png"
)


plt.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Confusion matrix saved at:")
print(confusion_matrix_path)


# ============================================================
# 13. SAVE INDIVIDUAL PREDICTIONS
# ============================================================

print("\n========================================")
print("SAVING TEST PREDICTIONS")
print("========================================")


test_file_names = [
    file_names[index]
    for index in test_indices
]


prediction_data = []


for i in range(len(y_test)):

    row = {
        "file_name": test_file_names[i],
        "true_class": CLASS_NAMES[y_test[i]],
        "predicted_class": CLASS_NAMES[y_pred[i]],
        "confidence": float(confidence[i])
    }

    # Add probability for each class

    for class_index, class_name in enumerate(CLASS_NAMES):

        row[
            f"probability_{class_name.lower()}"
        ] = float(
            probabilities[i][class_index]
        )

    prediction_data.append(row)


prediction_df = pd.DataFrame(
    prediction_data
)


prediction_csv_path = os.path.join(
    PREDICTION_PATH,
    "test_predictions.csv"
)


prediction_df.to_csv(
    prediction_csv_path,
    index=False
)


print("Prediction file saved at:")
print(prediction_csv_path)


# ============================================================
# 14. SAVE SUMMARY METRICS
# ============================================================

metrics = {
    "Test Accuracy": accuracy
}


metrics_df = pd.DataFrame(
    [metrics]
)


metrics_path = os.path.join(
    PREDICTION_PATH,
    "evaluation_metrics.csv"
)


metrics_df.to_csv(
    metrics_path,
    index=False
)


print("Evaluation metrics saved at:")
print(metrics_path)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n========================================")
print("EVALUATION COMPLETED")
print("========================================")

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)

print("\nGenerated files:")

print(
    "1. Confusion Matrix:"
)

print(
    confusion_matrix_path
)

print(
    "\n2. Classification Report:"
)

print(
    report_path
)

print(
    "\n3. Test Predictions:"
)

print(
    prediction_csv_path
)

print(
    "\n4. Evaluation Metrics:"
)

print(
    metrics_path
)

print(
    "\n5. Feature Scaler:"
)

print(
    scaler_path
)

print("\nEvaluation finished successfully!")