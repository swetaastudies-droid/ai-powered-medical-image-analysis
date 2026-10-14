"""Frozen MobileNetV2 transfer learning model."""
import tensorflow as tf
from tensorflow.keras import layers

def build_model():
    inputs = layers.Input(shape=(224, 224, 3), name="xray_input")
    # Input is already normalized by src.preprocessing.
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    backbone.trainable = False
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pneumonia_score")(x)
    model = tf.keras.Model(inputs, outputs, name="xray_label_classifier")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")])
    return model
