# 🧠 Brain Tumor Classification using CNN

This project uses a Convolutional Neural Network (CNN) to classify brain MRI images into one of four categories:
- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

It also includes a **Streamlit web app** where users can upload an MRI image and get a prediction.

---

## 📁 Project Contents

BrainTumorClassificationUsingCNN/
├── Training/ # Training images
├── Testing/ # Testing images
├── model/ # Contains the trained model (.h5)
├── BrainTumorClassification.ipynb # Notebook for training the CNN
├── tumor_classification_app.py # Streamlit app
├── requirements.txt # Required libraries


---

## ✅ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/BrainTumorClassificationUsingCNN.git
cd BrainTumorClassificationUsingCNN


2. Install dependencies
pip install -r requirements.txt


3. Launch the app
streamlit run tumor_classification_app.py
