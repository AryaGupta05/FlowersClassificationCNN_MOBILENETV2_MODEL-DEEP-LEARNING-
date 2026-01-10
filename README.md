# 🌸 Flower Classification using MobileNetV2

A deep-learning based flower classifier trained on 5 flower categories using
MobileNetV2 Transfer Learning.
Designed for college submission, portfolio showcase, and production-grade ML workflows.

Repository:
https://github.com/AryaGupta05/FlowersClassificationCNN_MOBILENETV2_MODEL-DEEP-LEARNING-

---
LIVE DEMO:
You can acess the live demo of this project in real time here.
https://flowerclassificationcnndeeplearning.streamlit.app/
## 

<p align="center">
  <img src="https://i.imgur.com/32NtpAZ.jpeg" width="95%" style="border-radius:12px"/>
</p>

---

## 

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

## 📌 Project Overview

This project uses MobileNetV2, pre-trained on ImageNet, as a frozen backbone.
A lightweight custom classification head is trained to classify 5 flower species:

- Daisy
- Dandelion
- Rose
- Sunflower
- Tulip

Key Highlights:
- ~87% validation accuracy
- GPU-powered training using Google Colab
- Clean and efficient transfer learning architecture
- Supports prediction on any custom flower image
- College-ready visualizations and reproducible code

---

## 🧠 Model Architecture (MobileNetV2 Transfer Learning)

The model follows a transfer learning pipeline where MobileNetV2 acts as a feature extractor
and only the classifier head is trained.

    Input Image (180 × 180 × 3)
            |
            v
    MobileNetV2 Backbone
    (Pretrained on ImageNet, Frozen Layers)
            |
            v
    GlobalAveragePooling2D
            |
            v
    Dense Layer (128 units, ReLU)
            |
            v
    Dropout (0.3)
            |
            v
    Output Layer (5 units, Softmax)

Architecture Explanation:
- MobileNetV2 extracts high-level visual features from images
- GlobalAveragePooling2D reduces spatial dimensions efficiently
- Dense layer learns task-specific patterns
- Dropout prevents overfitting
- Softmax layer predicts one of the five flower classes

---

## 📈 Training Results

Accuracy Curve:
results/accuracy.png

Loss Curve:
results/loss.png

Confusion Matrix:
results/confusion_matrix.png

---

## 🧪 Testing the Model

To test the trained model on any custom image, use:

    predict_image("your_image.jpg")

Sample Output:
Prediction: dandelion

---

## 🗂 Project Structure

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

---

## 🔧 Setup Instructions

1. Clone the repository

    git clone https://github.com/AryaGupta05/FlowersClassificationCNN_MOBILENETV2_MODEL-DEEP-LEARNING-
    cd FlowersClassificationCNN_MOBILENETV2_MODEL-DEEP-LEARNING-

2. Install dependencies

    pip install tensorflow numpy matplotlib pillow scikit-learn

3. Run the notebook

    Flower_Classification_CNN.ipynb

---

## 🔮 Future Enhancements

- Fine-tune MobileNetV2 for 90%+ validation accuracy
- Deploy the model using Streamlit or Gradio
- Convert the model to TensorFlow Lite for mobile deployment
- Extend the dataset to include more flower categories

---

## 👤 Author

Arya Gupta  
B.Tech CSE(AI&ML)

GitHub: https://github.com/AryaGupta05

If you like this project, please star the repository.
