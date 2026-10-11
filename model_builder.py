import tensorflow as tf
import keras
from keras import layers, models, regularizers

def build_model(
    input_shape:   tuple = (13, 63, 1),
    l2_lambda:     float = 1e-4,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:


    # he_normal: correct variance scaling for ReLU networks.
    # Prevents vanishing/exploding activations and reduces
    # run-to-run variance caused by unlucky random initialisation.
    init = "he_normal"

    inputs = layers.Input(shape=input_shape, name="mfcc_input")


    x = layers.Conv2D(
        32, (3, 3), padding="same",
        kernel_initializer=init,
        kernel_regularizer=regularizers.l2(l2_lambda),
        name="conv1"
    )(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)   # → (6, 31, 32)
    x = layers.Dropout(0.25, name="drop1")(x)


    x = layers.Conv2D(
        64, (3, 3), padding="same",
        kernel_initializer=init,
        kernel_regularizer=regularizers.l2(l2_lambda),
        name="conv2"
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)   # → (3, 15, 64)
    x = layers.Dropout(0.25, name="drop2")(x)


    # (1,3) kernel sweeps along time axis only — efficient for temporal cues.
    x = layers.Conv2D(
        128, (1, 3), padding="same",
        kernel_initializer=init,
        kernel_regularizer=regularizers.l2(l2_lambda),
        name="conv3"
    )(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.Dropout(0.3, name="drop3")(x)


    # (3, 15, 128) → (128,)  — far fewer parameters than Flatten + Dense
    x = layers.GlobalAveragePooling2D(name="gap")(x)


    x = layers.Dense(
        128,
        kernel_initializer=init,
        kernel_regularizer=regularizers.l2(l2_lambda),
        name="dense1"
    )(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.Dropout(0.4, name="drop4")(x)

    x = layers.Dense(
        64,
        kernel_initializer=init,
        kernel_regularizer=regularizers.l2(l2_lambda),
        name="dense2"
    )(x)
    x = layers.Activation("relu", name="relu5")(x)
    x = layers.Dropout(0.3, name="drop5")(x)


    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="FordEngineCNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )
    return model


if __name__ == "__main__":
    m = build_model()
    m.summary()
    print(f"\nTotal trainable parameters: {m.count_params():,}")
