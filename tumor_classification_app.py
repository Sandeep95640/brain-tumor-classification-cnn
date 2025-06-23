import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# -------------------------------
# Load your trained model
# -------------------------------
model = load_model('model/brain_tumor_classifier.h5')

# Define class labels
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# -------------------------------
# Image preprocessing function
# -------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB')                  # Ensure 3 channels
    image = image.resize((128, 128))              # Resize to model input shape
    image_array = np.array(image) / 255.0         # Normalize pixel values
    return np.expand_dims(image_array, axis=0)    # Add batch dimension

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor Classification with CNN")
st.write("Upload an MRI image to classify the tumor type:")

# Upload file
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

# When user uploads a file
if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Show result
    st.success(f"🧪 Prediction: **{predicted_class.upper()}**")
    st.info(f"📊 Confidence: {confidence * 100:.2f}%")
