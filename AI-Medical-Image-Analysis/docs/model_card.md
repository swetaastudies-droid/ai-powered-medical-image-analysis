# Model card (template)
- **Task:** Binary dataset-label classification (NORMAL/PNEUMONIA) from chest X-ray images.
- **Architecture:** ImageNet-pretrained MobileNetV2, frozen backbone, global average pooling, dropout, sigmoid classifier.
- **Inputs:** RGB resized 224×224, MobileNetV2 preprocessing applied once.
- **Outputs:** Experimental score for PNEUMONIA label; 0.5 threshold for class assignment.
- **Dataset:** Public chest X-ray pneumonia dataset; largely pediatric.
- **Evaluation:** Fill in actual test accuracy, precision, recall, F1 and ROC-AUC after running `python -m src.evaluate`.
- **Limitations:** No clinical validation, patient-level independence not assured, no external-site validation, no probability calibration, possible dataset shortcuts/biases.
- **Intended use:** Education and research demonstration only; NOT for diagnosis, triage, or treatment.
