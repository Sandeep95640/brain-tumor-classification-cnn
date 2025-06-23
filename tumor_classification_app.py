import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# -------------------------------
# Load the trained CNN model
# -------------------------------
model = load_model('model/brain_tumor_classifier.h5')

# Define class labels
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# -------------------------------
# Image preprocessing
# -------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB')                  # Ensure 3 color channels
    image = image.resize((128, 128))              # Resize for model input
    image_array = np.array(image) / 255.0         # Normalize pixel values
    return np.expand_dims(image_array, axis=0)    # Add batch dimension

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor Classification with CNN")
st.write("Upload an MRI image to classify the tumor type.")

# Upload MRI file
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and display the image
        image = Image.open(uploaded_file)

        # Optional: Restrict file size to 5MB
        if uploaded_file.size > 5 * 1024 * 1024:
            st.warning("⚠️ File too large. Please upload an image smaller than 5MB.")
        else:
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Preprocess and predict
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)

            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

            # Show result
            st.success(f"🧪 Prediction: **{predicted_class.upper()}**")
            st.info(f"📊 Confidence: {confidence * 100:.2f}%")

    except Exception as e:
        st.error(f"❌ Could not process the image.\n\nError: {str(e)}")
