import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

IMAGE_SIZE = 256
MODEL_PATH = 'LungCancerPrediction.h5'

model = tf.keras.models.load_model(MODEL_PATH)

st.set_page_config(page_title="Lung Cancer Prediction", layout="centered")
st.title("Lung Cancer Detection from Chest X-ray")
st.markdown("Upload a chest X-ray image to check for Lung Cancer.")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

    if st.button("Predict"):
        try:
            input_data = preprocess_image(image)
            prediction = model.predict(input_data)
            confidence = float(prediction[0][0])

            if confidence > 0.5:
                st.error(f"Lung Cancer Detected (Confidence: {confidence:.2f})")
            else:
                st.success(f"No Lung Cancer Detected (Confidence: {1 - confidence:.2f})")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
