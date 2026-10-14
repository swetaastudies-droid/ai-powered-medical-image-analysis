"""Shared MobileNetV2 image preprocessing (exactly once)."""
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from src.config import IMAGE_SIZE

def load_image(path, label):
    data = tf.io.read_file(path)
    image = tf.io.decode_image(data, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    image = tf.image.resize(image, IMAGE_SIZE)
    image = preprocess_input(image)  # raw 0..255 -> approx -1..1
    return image, tf.reshape(tf.cast(label, tf.float32), (1,))

def preprocess_single_image(path):
    image = tf.keras.utils.load_img(str(path), target_size=IMAGE_SIZE, color_mode="rgb")
    image = tf.keras.utils.img_to_array(image)
    return tf.expand_dims(preprocess_input(image), axis=0)
