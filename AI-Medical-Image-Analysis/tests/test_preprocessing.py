import numpy as np
import tensorflow as tf
from src.preprocessing import preprocess_single_image

def test_preprocess_single_image(tmp_path):
    image = tf.ones((30, 40, 3), dtype=tf.uint8) * 127
    path = tmp_path / "test.png"
    tf.io.write_file(str(path), tf.io.encode_png(image))
    result = preprocess_single_image(path).numpy()
    assert result.shape == (1, 224, 224, 3)
    assert np.isfinite(result).all()
    assert result.min() >= -1.01 and result.max() <= 1.01
