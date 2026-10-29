import os
import sys
import numpy as np
import pickle


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

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "features"
)


# ============================================================
# CLASS INFORMATION
# ============================================================

CLASSES = [
    "asthma",
    "Bronchial",
    "copd",
    "healthy",
    "pneumonia"
]


CLASS_TO_INDEX = {
    "asthma": 0,
    "Bronchial": 1,
    "copd": 2,
    "healthy": 3,
    "pneumonia": 4
}


# ============================================================
# IMPORT PREPROCESSING FUNCTIONS
# ============================================================

SRC_PATH = os.path.dirname(
    os.path.dirname(__file__)
)

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from preprocessing.audio_preprocessing import preprocess_audio


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)


# ============================================================
# EXTRACT FEATURES FROM DATASET
# ============================================================

def extract_dataset_features():

    mfcc_features = []
    mel_features = []
    labels = []
    file_names = []

    print("Starting feature extraction...")
    print("----------------------------------------")

    total_files = 0

    for class_name in CLASSES:

        class_path = os.path.join(
            DATASET_PATH,
            class_name
        )

        if not os.path.exists(class_path):

            print(
                f"WARNING: Class folder not found: {class_name}"
            )

            continue

        files = [
            file for file in os.listdir(class_path)
            if file.lower().endswith(".wav")
        ]

        print(
            f"{class_name}: {len(files)} audio files"
        )

        for file_name in files:

            file_path = os.path.join(
                class_path,
                file_name
            )

            try:

                audio, mfcc, mel = preprocess_audio(
                    file_path
                )

                mfcc_features.append(mfcc)
                mel_features.append(mel)

                labels.append(
                    CLASS_TO_INDEX[class_name]
                )

                file_names.append(
                    file_name
                )

                total_files += 1

            except Exception as error:

                print(
                    f"Error processing {file_name}: {error}"
                )

    # --------------------------------------------------------
    # CONVERT LISTS TO NUMPY ARRAYS
    # --------------------------------------------------------

    mfcc_features = np.array(
        mfcc_features,
        dtype=np.float32
    )

    mel_features = np.array(
        mel_features,
        dtype=np.float32
    )

    labels = np.array(
        labels,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # SAVE FEATURES
    # --------------------------------------------------------

    np.save(
        os.path.join(
            OUTPUT_PATH,
            "mfcc_features.npy"
        ),
        mfcc_features
    )

    np.save(
        os.path.join(
            OUTPUT_PATH,
            "mel_features.npy"
        ),
        mel_features
    )

    np.save(
        os.path.join(
            OUTPUT_PATH,
            "labels.npy"
        ),
        labels
    )

    # --------------------------------------------------------
    # SAVE FILE NAMES
    # --------------------------------------------------------

    with open(
        os.path.join(
            OUTPUT_PATH,
            "file_names.pkl"
        ),
        "wb"
    ) as file:

        pickle.dump(
            file_names,
            file
        )

    # --------------------------------------------------------
    # DISPLAY SUMMARY
    # --------------------------------------------------------

    print("\n========================================")
    print("FEATURE EXTRACTION COMPLETED")
    print("========================================")

    print(
        "Total files processed:",
        total_files
    )

    print(
        "MFCC feature shape:",
        mfcc_features.shape
    )

    print(
        "Mel Spectrogram shape:",
        mel_features.shape
    )

    print(
        "Labels shape:",
        labels.shape
    )

    print(
        "Unique labels:",
        np.unique(labels)
    )

    print(
        "\nFeatures saved to:"
    )

    print(OUTPUT_PATH)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    extract_dataset_features()