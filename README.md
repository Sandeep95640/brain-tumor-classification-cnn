# 🧠 Brain Tumor Classification using CNN

This project uses a Convolutional Neural Network (CNN) to classify brain MRI images into one of four categories:
- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

It also includes a **Streamlit web app** where users can upload an MRI image and get a prediction.

---

## 📁 Project Contents

```
BrainTumorClassificationUsingCNN/
├── Training/ # Folder containing training MRI images organized by class
├── Testing/ # Folder containing testing MRI images organized by class
├── model/
│ └── brain_tumor_Classifier.h5 # Trained CNN model saved in HDF5 format
├── BrainTumorClassification.ipynb # Jupyter Notebook for loading data, building, training, and evaluating the CNN
├── tumor_classification_app.py # Streamlit app to upload MRI images and get predictions
├── requirements.txt # List of all required Python libraries with versions

```

---

## ✅ How to Run

```bash
1. Clone the repository
git clone https://github.com/your-username/BrainTumorClassificationUsingCNN.git
cd BrainTumorClassificationUsingCNN


2. Install dependencies
pip install -r requirements.txt


3. Launch the app
streamlit run tumor_classification_app.py
