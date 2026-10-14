
import streamlit as st
from PIL import Image
from pathlib import Path
import tempfile
import numpy as np

from src.predict import MedicalImagePredictor
from src.preprocessing import preprocess_single_image
from src.explainability import (
    make_gradcam_heatmap,
    create_gradcam_overlay,
)

st.set_page_config(
    page_title="AI Medical Image Analysis",
    page_icon="🩻",
    layout="wide",
)

st.title("AI-Powered Medical Image Analysis System")

st.write(
    "An educational chest X-ray classification "
    "prototype using TensorFlow and MobileNetV2."
)

st.warning(
    "Research and educational use only. "
    "Not intended for medical diagnosis or treatment."
)


@st.cache_resource
def load_predictor():
    return MedicalImagePredictor()


try:
    predictor = load_predictor()
    st.success("Trained model loaded successfully.")
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()


uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded X-ray")
            st.image(image, use_container_width=True)

        if st.button("Analyze X-ray", type="primary"):
            suffix = Path(uploaded_file.name).suffix or ".jpg"

            # Save the uploaded image temporarily.
            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = Path(temp_file.name)

            try:
                # Run the trained model.
                result = predictor.predict(temp_path)

                # Generate Grad-CAM explanation.
                image_batch = preprocess_single_image(temp_path)

                heatmap = make_gradcam_heatmap(
                    predictor.model,
                    image_batch,
                )

                original_image = np.array(image)

                overlay = create_gradcam_overlay(
                    original_image,
                    heatmap,
                    alpha=0.4,
                )

                # Display prediction results.
                with col2:
                    st.subheader("Model output")

                    st.metric(
                        "Predicted label",
                        result["predicted_label"],
                    )

                    for label, score in result["scores"].items():
                        st.write(label)
                        st.progress(float(score))
                        st.write(f"Model score: {score:.2%}")

                    st.caption(
                        "Model scores are experimental and "
                        "must not guide clinical decisions."
                    )

                    st.divider()

                    st.subheader("Grad-CAM visualization")

                    st.write(
                        "The highlighted regions indicate areas "
                        "that influenced the model's output. "
                        "This visualization is experimental and "
                        "does not establish the presence of disease."
                    )

                    st.image(
                        overlay,
                        caption="Experimental Grad-CAM overlay",
                        use_container_width=True,
                    )

            except Exception as exc:
                st.error(f"Unable to process this image: {exc}")

            finally:
                # Always remove the temporary image file.
                temp_path.unlink(missing_ok=True)

    except Exception as exc:
        st.error(f"Unable to open the uploaded image: {exc}")
