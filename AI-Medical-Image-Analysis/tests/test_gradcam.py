
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config import DATA_ROOT
from src.preprocessing import preprocess_single_image
from src.explainability import make_gradcam_heatmap


def test_gradcam_generates_valid_heatmap():
    model_path = Path("models/best_model.keras")
    assert model_path.exists(), "Trained model not found."

    image_path = DATA_ROOT / "test" / "NORMAL" / "IM-0001-0001.jpeg"
    assert image_path.exists(), f"Image not found: {image_path}"

    model = tf.keras.models.load_model(model_path)
    image_batch = preprocess_single_image(image_path)

    heatmap = make_gradcam_heatmap(model, image_batch)

    assert heatmap.ndim == 2
    assert heatmap.size > 0
    assert np.isfinite(heatmap).all()
    assert heatmap.min() >= 0
    assert heatmap.max() <= 1
