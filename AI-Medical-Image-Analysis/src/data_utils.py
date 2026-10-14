"""Read image paths and create reproducible datasets."""
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.config import CLASS_NAMES, BATCH_SIZE, SEED
from src.preprocessing import load_image

EXTENSIONS = {".jpeg", ".jpg", ".png"}

def collect_image_paths(folder):
    folder = Path(folder)
    paths, labels = [], []
    for index, label in enumerate(CLASS_NAMES):
        class_dir = folder / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing folder: {class_dir}. See data/README.md")
        found = sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not found:
            raise ValueError(f"No JPG/JPEG/PNG images found in {class_dir}")
        paths.extend(map(str, found))
        labels.extend([index] * len(found))
    return np.array(paths), np.array(labels, dtype=np.int32)

def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    return ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def create_datasets(root):
    root = Path(root)
    train_paths, train_labels = collect_image_paths(root / "train")
    test_paths, test_labels = collect_image_paths(root / "test")
    x_train, x_val, y_train, y_val = train_test_split(
        train_paths, train_labels, test_size=0.2, stratify=train_labels, random_state=SEED
    )
    print(f"Training: {len(x_train)} | Validation: {len(x_val)} | Test: {len(test_paths)}")
    print(f"Class mapping: {CLASS_NAMES}")
    return (make_dataset(x_train, y_train, True), make_dataset(x_val, y_val),
            make_dataset(test_paths, test_labels), test_paths, test_labels)
