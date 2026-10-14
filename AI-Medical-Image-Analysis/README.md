# AI-Powered Medical Image Analysis System

**Educational/research demonstration only. Not for medical diagnosis or clinical decisions.**

A beginner-friendly binary chest X-ray **dataset-label classifier** (NORMAL vs PNEUMONIA) using TensorFlow/Keras, pretrained MobileNetV2, scikit-learn, and Streamlit.

## 1. Prerequisites
- Windows/macOS/Linux, Python 3.13 (64-bit) recommended for this starter; TensorFlow compatibility depends on your platform.
- VS Code, internet for installing packages and downloading pretrained ImageNet weights.

## 2. Create and activate environment

Windows PowerShell:
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
macOS/Linux (if Python 3.13 installed):
```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Download data
Download the dataset yourself from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia and extract to `data/raw/chest_xray/` so you have:
```
data/raw/chest_xray/train/NORMAL/
data/raw/chest_xray/train/PNEUMONIA/
data/raw/chest_xray/test/NORMAL/
data/raw/chest_xray/test/PNEUMONIA/
```
The original `val` folder is optional: our code makes a stratified validation split from `train` and keeps `test` separate. **Image-level splits do not guarantee patient-level separation.** Check dataset terms and attribution before use.

## 4. Run
From the repository root with `.venv` activated:
```bash
python -m src.check_dataset
python -m src.train
python -m src.evaluate
python -m src.predict "data/raw/chest_xray/test/NORMAL/REPLACE_WITH_REAL_FILENAME.jpeg"
python -m streamlit run app.py
python -m pytest -q
```
`main.py` also supports `python main.py check`, `python main.py train`, and `python main.py evaluate`.

## 5. Outputs
- `models/best_model.keras`, `models/class_names.json` after training
- `outputs/figures/accuracy_curve.png`, `loss_curve.png`
- `outputs/metrics/test_metrics.json`, `classification_report.json`, `test_predictions.csv`
- `outputs/figures/confusion_matrix.png`, `predicted_vs_actual.png`

## Important limitations
The model learns associations with **dataset labels**, not independently verified patient diagnoses. This dataset is largely pediatric and not representative of general clinical populations. The scores are **not calibrated clinical probabilities**. The `0.5` threshold is an experimental default. No patient history, independent radiologist adjudication, external validation, or clinical safety study is included. Do not use for triage or treatment.

See `docs/` for architecture, dataset guide, model card and experiment log.
