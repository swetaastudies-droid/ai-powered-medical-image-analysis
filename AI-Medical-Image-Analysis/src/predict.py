"""Run an individual image prediction using the trained model."""
import argparse
import json
from pathlib import Path
import tensorflow as tf
from src.config import MODEL_DIR, CLASS_NAMES
from src.preprocessing import preprocess_single_image

class MedicalImagePredictor:
    def __init__(self):
        model_path = MODEL_DIR / "best_model.keras"
        labels_path = MODEL_DIR / "class_names.json"
        if not model_path.exists() or not labels_path.exists():
            raise FileNotFoundError("Model files missing. Run python -m src.train first.")
        self.model = tf.keras.models.load_model(model_path)
        self.classes = json.loads(labels_path.read_text(encoding="utf-8"))
        if self.classes != CLASS_NAMES:
            raise ValueError("Class order mismatch")

    def predict(self, image_path):
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        score = float(self.model(preprocess_single_image(path), training=False).numpy()[0, 0])
        return {"image": path.name,
                "predicted_label": self.classes[int(score >= 0.5)],
                "scores": {"NORMAL": 1 - score, "PNEUMONIA": score}}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to JPG/JPEG/PNG image")
    args = parser.parse_args()
    print(json.dumps(MedicalImagePredictor().predict(args.image_path), indent=2))
    print("Research prototype only. Not for medical diagnosis.")

if __name__ == "__main__":
    main()
