import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
from report_generator import generate_pdf_report

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Radiology System",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# PROFESSIONAL DARK HOSPITAL THEME
# --------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0b1c2d;
}
.stApp {
    background-color: #0b1c2d;
}
h1, h2, h3 {
    color: #00e5ff;
}
.css-1d391kg {
    background-color: #102c44;
}
.stButton>button {
    background-color: #00e5ff;
    color: black;
    border-radius: 8px;
    height: 3em;
}
.stProgress > div > div > div > div {
    background-color: #00e5ff;
}
</style>
""", unsafe_allow_html=True)

st.title("🏥 AI-Powered Brain Tumor Detection System")
st.markdown("### Deep Learning Based MRI Classification")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "brain_tumor_model.h5")
    return tf.keras.models.load_model(model_path)

model = load_model()
class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# --------------------------------------------------
# MODEL SUMMARY (Expandable)
# --------------------------------------------------
with st.expander("📚 View Model Architecture"):
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    summary = "\n".join(stringlist)
    st.text(summary)

# --------------------------------------------------
# PREPROCESS IMAGE
# --------------------------------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    return image

# --------------------------------------------------
# SIDEBAR – PATIENT INFO
# --------------------------------------------------
st.sidebar.header("👤 Patient Information")
patient_name = st.sidebar.text_input("Name")
patient_age = st.sidebar.number_input("Age", 1, 120)
scan_id = st.sidebar.text_input("Scan ID")

uploaded_file = st.sidebar.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# MAIN INTERFACE
# --------------------------------------------------
if uploaded_file is not None:

    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file)

    with col1:
        st.subheader("🖼 Uploaded MRI Scan")
        st.image(image, use_column_width=True)

    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]

    pred_index = np.argmax(prediction)
    pred_class = class_names[pred_index]
    confidence = float(prediction[pred_index])

    with col2:
        st.subheader("📊 AI Diagnosis")

        if pred_class == "No Tumor":
            st.success(f"Diagnosis: {pred_class}")
        else:
            st.error(f"Diagnosis: {pred_class}")

        st.write(f"Confidence: {confidence*100:.2f}%")
        st.progress(int(confidence * 100))

        # Probability Bar Chart
        st.subheader("🔬 Prediction Probability Distribution")

        prob_df = pd.DataFrame({
            "Tumor Type": class_names,
            "Probability": prediction
        })

        st.bar_chart(prob_df.set_index("Tumor Type"))

    st.markdown("---")

    # PDF REPORT
    if st.button("📄 Generate Diagnostic Report"):
        pdf_path = generate_pdf_report(
            patient_name,
            patient_age,
            scan_id,
            pred_class,
            confidence
        )

        with open(pdf_path, "rb") as f:
            st.download_button(
                "⬇ Download Report",
                f,
                file_name="Brain_Tumor_Report.pdf"
            )

else:
    st.info("Upload MRI image to begin AI analysis.")
