# Architecture
Public chest X-ray dataset → `src/data_utils.py` (image-level stratified validation split) → `src/preprocessing.py` (224×224, RGB, MobileNetV2 scaling) → `src/model.py` (frozen ImageNet MobileNetV2 + sigmoid head) → `src/train.py` (checkpoint) → `src/evaluate.py` (untouched test set) → `src/predict.py` → `app.py` (Streamlit demonstration).
