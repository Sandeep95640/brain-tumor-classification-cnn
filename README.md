# 🧠 Brain Tumor Classification using CNN

An end-to-end Deep Learning project that classifies brain MRI images into four categories using a Convolutional Neural Network (CNN).  
The project also includes a deployed **Streamlit web application** for real-time tumor prediction.

🔗 **Live App:**  
https://brain-tumor-classification-cnn.streamlit.app/

---

## 📌 Problem Statement

Brain tumors are abnormal cell growths inside the brain that can be life-threatening. Early and accurate classification of tumors is crucial for:

- Effective treatment planning  
- Improving patient survival rates  
- Reducing the risk of permanent neurological damage  

Manual MRI interpretation requires expert radiologists and can be time-consuming.

👉 This project builds an automated deep learning system that classifies MRI images into different tumor types.

---

## 🎯 Project Objective

To develop a CNN-based model that classifies brain MRI scans into one of the following four categories:

- **Glioma** – Tumor originating from glial cells  
- **Meningioma** – Tumor in the meninges  
- **Pituitary** – Tumor in the pituitary gland  
- **No Tumor** – Normal brain scan  

---

## 📊 Dataset Overview

The dataset was downloaded from **Kaggle**.

The Kaggle dataset itself is a combination of publicly available MRI datasets originally sourced from:

- Figshare  
- SARTAJ  
- Br35H  

### 📦 Dataset Details

- Total Images: **7023**
- Image Size: Resized to **128 × 128**
- Classes: **4**
- Training Images: **5712**
- Testing Images: **1311**

Images are organized into class-specific folders.

---

## 🧠 Model Architecture

The CNN model is built using **Keras Sequential API** and includes:

- 3 Convolutional Layers (32, 64, 128 filters)
- MaxPooling Layers
- Batch Normalization
- Fully Connected Dense Layer
- Dropout (0.5)
- Softmax Output Layer (4 classes)

### 🏗️ Architecture Flow

```
Input Image (128x128x3)
        ↓
Conv2D (32) + ReLU
        ↓
MaxPooling
        ↓
BatchNormalization
        ↓
Conv2D (64) + ReLU
        ↓
MaxPooling
        ↓
BatchNormalization
        ↓
Conv2D (128) + ReLU
        ↓
MaxPooling
        ↓
BatchNormalization
        ↓
Flatten
        ↓
Dense (258) + ReLU
        ↓
Dropout (0.5)
        ↓
Dense (4) + Softmax
```

---

## 🛠️ Training Details

- Optimizer: **Adam**
- Loss Function: **Categorical Crossentropy**
- Metric: **Accuracy**
- Early Stopping: Enabled (patience = 3)
- Epochs: Up to 20

### 📈 Performance

- Validation Accuracy: ~**96%**
- EarlyStopping used to prevent overfitting
- Model saved in `.h5` format

---

## 🧹 Data Preprocessing

- Images resized to 128×128
- Pixel normalization (scaled from [0–255] to [0–1])
- Label encoding
- One-hot encoding
- Dataset shuffled before training

---

## 🌐 Streamlit Web Application

The project includes a user-friendly web interface built using **Streamlit**.

### 🚀 Features

- Upload MRI image
- Real-time tumor classification
- Displays predicted class
- Clean and simple UI
- Deployable on Streamlit Cloud

---

## 📁 Project Structure

```
BrainTumorClassificationUsingCNN/
│
├── Training/                      # Training MRI images (class-wise folders)
├── Testing/                       # Testing MRI images (class-wise folders)
│
├── model/
│   └── brain_tumor_Classifier.h5  # Saved trained model
│
├── BrainTumorClassification.ipynb # Model development notebook
├── tumor_classification_app.py    # Streamlit app
├── requirements.txt               # Dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Sandeep95640/BrainTumorClassificationUsingCNN.git
cd BrainTumorClassificationUsingCNN
```

### 2️⃣ (Optional) Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Streamlit App

```bash
streamlit run tumor_classification_app.py
```

---

## 💾 Model Saving

The trained CNN model is saved using:

```python
model.save("model/brain_tumor_Classifier.h5")
```

This allows:
- Reuse without retraining
- Easy deployment
- Integration into applications

---

## 🔮 Future Improvements

- Use Transfer Learning (ResNet, VGG16)
- Add Data Augmentation
- Implement Grad-CAM for explainability
- Improve dataset balancing
- Deploy using FastAPI + Docker
- Add MRI segmentation support


---

# ⭐ If you found this project useful, consider giving it a star!
