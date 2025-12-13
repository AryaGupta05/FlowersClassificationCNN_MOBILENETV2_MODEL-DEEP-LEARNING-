# 🌸 Flower Classification using MobileNetV2  
A deep-learning based flower classifier trained on 5 categories using **MobileNetV2 Transfer Learning**.  
Designed for **college submission**, **portfolio showcase**, and **production-grade ML workflows**.

---

## 🖼️ Main Banner

<p align="center">
  <img src="https://i.imgur.com/32NtpAZ.jpeg" width="95%" style="border-radius:12px"/>
</p>

---

## 🖼️ Secondary Aesthetic Panel

<p align="center">
  <img src="https://i.imgur.com/8bN6eOL.jpeg" width="90%" style="border-radius:12px"/>
</p>

---

## 🚀 Technologies Used

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white&style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white&style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white&style=for-the-badge"/>
  <img src="https://img.shields.io/badge/MobileNetV2-4285F4?logo=google&logoColor=white&style=for-the-badge"/>
</p>

---

# 📌 Project Overview

This project uses **MobileNetV2**, pre-trained on ImageNet, as a frozen backbone.  
On top of it, a lightweight classifier is trained to distinguish **5 flower species**:

- 🌼 Daisy  
- 🌾 Dandelion  
- 🌹 Rose  
- 🌻 Sunflower  
- 🌷 Tulip  

Key Highlights:
- 87% validation accuracy  
- GPU-powered training  
- Clean architecture  
- Predict any custom flower image  
- College-ready visualizations + code  

---

# 🧠 Model Architecture (Detailed)
# 🧠 Model Architecture (MobileNetV2 Transfer Learning)

The model uses **MobileNetV2** as a frozen feature extractor with a custom classification head.
┌──────────────────────────────┐
│        Input Image           │
│       (180 × 180 × 3)        │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│   MobileNetV2 (Frozen Base)  │
│  Pretrained on ImageNet      │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│  GlobalAveragePooling2D      │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│  Dense (128, ReLU)           │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│        Dropout (0.3)         │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│   Dense (5, Softmax Output)  │
└──────────────────────────────┘


**Explanation:**
- **MobileNetV2** extracts high-level image features  
- **GlobalAveragePooling** reduces feature maps efficiently  
- **Dense + Dropout** improves learning and prevents overfitting  
- **Softmax output** predicts one of the 5 flower classes  

project_root/
 ├── Flower_Classification_CNN.ipynb
 ├── mobilenet_flower_model.h5
 ├── dataset/
 │    └── flowers/
 │         ├── daisy/
 │         ├── dandelion/
 │         ├── rose/
 │         ├── sunflower/
 │         └── tulip/
 ├── results/
 │    ├── accuracy.png
 │    ├── loss.png
 │    └── confusion_matrix.png
 └── README.md



