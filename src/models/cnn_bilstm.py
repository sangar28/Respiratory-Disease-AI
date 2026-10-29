import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)


# ============================================================
# MODEL SETTINGS
# ============================================================

TIME_STEPS = 47
FEATURES = 141
NUM_CLASSES = 5


# ============================================================
# BUILD CNN-BILSTM MODEL
# ============================================================

def build_cnn_bilstm_model():

    model = Sequential([
        
        # ----------------------------------------------------
        # INPUT LAYER
        # ----------------------------------------------------
        Input(
            shape=(TIME_STEPS, FEATURES)
        ),

        # ----------------------------------------------------
        # CNN BLOCK
        # ----------------------------------------------------
        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling1D(
            pool_size=2
        ),

        # ----------------------------------------------------
        # SECOND CNN BLOCK
        # ----------------------------------------------------
        Conv1D(
            filters=128,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling1D(
            pool_size=2
        ),

        # ----------------------------------------------------
        # BI-LSTM
        # ----------------------------------------------------
        Bidirectional(
            LSTM(
                64,
                return_sequences=False
            )
        ),

        # ----------------------------------------------------
        # FULLY CONNECTED LAYER
        # ----------------------------------------------------
        Dense(
            64,
            activation="relu"
        ),

        Dropout(
            0.5
        ),

        # ----------------------------------------------------
        # OUTPUT LAYER
        # ----------------------------------------------------
        Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    # ========================================================
    # COMPILE MODEL
    # ========================================================

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    model = build_cnn_bilstm_model()

    model.summary()