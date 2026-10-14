
"""Grad-CAM utilities for the MobileNetV2 research model."""

import cv2
import numpy as np
import tensorflow as tf


def make_gradcam_heatmap(model, image_batch):
    """
    Create a normalized Grad-CAM heatmap.

    image_batch must have shape (1, 224, 224, 3) and already
    be preprocessed using MobileNetV2 preprocess_input().
    """
    backbone = model.get_layer("mobilenetv2_1.00_224")

    # Build a model that returns the backbone feature maps and
    # the original classifier prediction.
    feature_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            backbone.get_layer("out_relu").output,
            backbone.output,
        ],
    )

    # Obtain feature maps from the backbone.
    with tf.GradientTape() as tape:
        # Run the backbone directly on the model's input tensor.
        feature_maps = feature_model(image_batch, training=False)[0]

        # Pass feature maps through the classifier layers.
        x = model.get_layer("global_average_pooling2d")(feature_maps)
        x = model.get_layer("dropout")(x, training=False)
        predictions = model.get_layer("pneumonia_score")(x)

        # Class 1 corresponds to PNEUMONIA in this project.
        score = predictions[:, 0]

    gradients = tape.gradient(score, feature_maps)

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients are unavailable. Check the model architecture."
        )

    # Average gradients spatially to obtain channel weights.
    weights = tf.reduce_mean(gradients, axis=(1, 2))

    # Combine feature maps using the gradient-derived weights.
    heatmap = tf.reduce_sum(
        feature_maps * weights[:, tf.newaxis, tf.newaxis, :],
        axis=-1,
    )

    heatmap = tf.nn.relu(heatmap)[0]
    maximum = tf.reduce_max(heatmap)

    heatmap = tf.where(
        maximum > 0,
        heatmap / (maximum + tf.keras.backend.epsilon()),
        tf.zeros_like(heatmap),
    )

    return heatmap.numpy()


def create_gradcam_overlay(original_image, heatmap, alpha=0.4):
    """Blend a normalized heatmap with an RGB image."""
    original_image = np.asarray(original_image, dtype=np.uint8)
    height, width = original_image.shape[:2]

    resized = cv2.resize(heatmap, (width, height))
    heatmap_uint8 = np.uint8(255 * np.clip(resized, 0, 1))

    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        original_image, 1 - alpha, colored, alpha, 0
    )

    return overlay
