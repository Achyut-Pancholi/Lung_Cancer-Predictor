import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Constants
IMAGE_SIZE = 256
MODEL_PATH = 'LungCancerPrediction.h5'
CLASS_NAMES = ['Benign', 'Malignant', 'Normal']  # Update based on your model

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Streamlit UI
st.set_page_config(page_title="Lung Cancer Prediction", layout="centered")
st.title("Lung Cancer Detection from Chest X-ray")
st.markdown("Upload a chest X-ray image to predict the condition (Benign, Malignant, Normal).")

# File uploader
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

# Image preprocessing function
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Prediction logic
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)

    if st.button("Predict"):
        try:
            input_data = preprocess_image(image)
            predictions = model.predict(input_data)
            predicted_index = np.argmax(predictions[0])
            predicted_label = CLASS_NAMES[predicted_index]
            confidence = float(predictions[0][predicted_index])

            st.subheader(f"Prediction: **{predicted_label}**")
            st.write(f"Confidence: {confidence:.2f}")

            # Optional: Show all class probabilities
            st.markdown("### Class Probabilities:")
            for i, class_name in enumerate(CLASS_NAMES):
                st.write(f"- **{class_name}**: {predictions[0][i]:.2f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
