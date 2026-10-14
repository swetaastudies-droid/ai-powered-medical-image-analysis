"""Train and save best checkpoint plus learning curves."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from src.config import DATA_ROOT, MODEL_DIR, METRICS_DIR, FIGURES_DIR, CLASS_NAMES, SEED
from src.data_utils import create_datasets
from src.model import build_model

def save_curves(history):
    for metric, filename in (("accuracy", "accuracy_curve.png"), ("loss", "loss_curve.png")):
        plt.figure(figsize=(8, 5))
        plt.plot(history.history[metric], label="Training")
        plt.plot(history.history["val_" + metric], label="Validation")
        plt.title(metric.capitalize() + " over epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / filename, dpi=150)
        plt.close()

def main():
    tf.keras.utils.set_random_seed(SEED)
    for folder in (MODEL_DIR, METRICS_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, _, _, _ = create_datasets(DATA_ROOT)
    model = build_model()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_DIR / "best_model.keras"), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(METRICS_DIR / "training_log.csv")),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks)
    (MODEL_DIR / "class_names.json").write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")
    (METRICS_DIR / "history.json").write_text(
        json.dumps({k: [float(x) for x in v] for k, v in history.history.items()}, indent=2), encoding="utf-8"
    )
    save_curves(history)
    print("Training complete. Model saved to models/best_model.keras")

if __name__ == "__main__":
    main()
