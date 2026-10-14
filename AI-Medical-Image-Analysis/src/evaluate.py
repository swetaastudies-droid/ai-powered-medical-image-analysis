"""Evaluate saved model on the untouched test folder."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from src.config import DATA_ROOT, MODEL_DIR, METRICS_DIR, FIGURES_DIR, CLASS_NAMES
from src.data_utils import collect_image_paths, make_dataset

def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "best_model.keras"
    if not model_path.is_file():
        raise FileNotFoundError("First run: python -m src.train")
    model = tf.keras.models.load_model(model_path)
    paths, labels = collect_image_paths(DATA_ROOT / "test")
    scores = model.predict(make_dataset(paths, labels), verbose=1).reshape(-1)
    predictions = (scores >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_pneumonia": float(precision_score(labels, predictions, zero_division=0)),
        "recall_pneumonia": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }
    (METRICS_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = classification_report(labels, predictions, labels=[0,1], target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    (METRICS_DIR / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame({"image_path": paths, "actual_label": [CLASS_NAMES[i] for i in labels],
                  "predicted_label": [CLASS_NAMES[i] for i in predictions],
                  "pneumonia_score": scores}).to_csv(METRICS_DIR / "test_predictions.csv", index=False)
    cm = confusion_matrix(labels, predictions, labels=[0,1])
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(values_format="d")
    plt.title("Test confusion matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    count = min(12, len(paths))
    fig, axes = plt.subplots(4, 3, figsize=(12, 14))
    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= count:
            continue
        with Image.open(paths[idx]) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"Actual: {CLASS_NAMES[labels[idx]]}\nPredicted: {CLASS_NAMES[predictions[idx]]}\nScore: {scores[idx]:.3f}", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "predicted_vs_actual.png", dpi=150)
    plt.close()
    print("Test results:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("Confusion matrix:\n", cm)

if __name__ == "__main__":
    main()
