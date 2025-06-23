import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('model/brain_tumor_Classifier.h5')
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Preprocessing function
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB')              # Convert to RGB format
    image = image.resize((128, 128))          # Resize for model input
    image_array = np.array(image) / 255.0     # Normalize pixel values
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Streamlit UI configuration
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor Classification with CNN")
st.write("Upload an MRI image to classify the tumor type.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        # Optional: Warn if file size exceeds limit (you can remove this check)
        if uploaded_file.size > 5 * 1024 * 1024:
            st.warning("⚠️ File too large. Please upload an image smaller than 5MB.")
        else:
            st.image(image, caption="Uploaded Image")  # ✅ Compatible line

            # Preprocess and predict
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)

            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

            # Display prediction
            st.success(f"🧪 Prediction: **{predicted_class.upper()}**")
            st.info(f"📊 Confidence: {confidence * 100:.2f}%")

    except Exception as e:
        st.error(f"❌ Could not process the image.\n\nError: {str(e)}")
