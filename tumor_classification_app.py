import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load model and label encoder
model = load_model('model/brain_tumor_Classifier.h5')
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Image preprocessing function
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB')                  # Convert to RGB
    image = image.resize((128, 128))              # Resize to match model input
    image_array = np.array(image) / 255.0         # Normalize
    return np.expand_dims(image_array, axis=0)    # Add batch dimension

# Streamlit UI
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor Classification with CNN")
st.write("Upload an MRI image to classify the tumor type:")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Predict on image
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Show prediction
    st.success(f"🧪 Prediction: **{predicted_class.upper()}**")
    st.info(f"Confidence: {confidence * 100:.2f}%")
