import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="FlowerVision AI – Technical Dashboard",
    page_icon="🌸",
    layout="wide"
)

# ================= CSS =================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
}
@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.card {
    background: rgba(255,255,255,0.12);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(18px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
}
h1,h2,h3 {color:white;}
p,li {color:#e0e0e0;}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenet_flower_model.h5")

model = load_model()
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# ================= SIDEBAR – THEORY =================
st.sidebar.markdown("## 📘 Technical Overview")
st.sidebar.markdown("""
### 🔬 Model Architecture
- CNN with **MobileNetV2**
- Transfer Learning (ImageNet)
- Softmax classifier

### ⚙️ Training Details
- Input size: 224×224  
- Optimizer: Adam  
- Loss: Categorical Crossentropy  
- Epochs: 10  

### 📊 Performance
- Training Accuracy: **~92%**
- Validation Accuracy: **~88%**
- Lightweight model (~11MB)

### 🚀 Why MobileNetV2?
- Depthwise separable convolutions  
- Faster inference  
- Low memory usage  
- Ideal for deployment
""")

# ================= HERO =================
st.markdown("<h1>🌸 FlowerVision AI</h1>", unsafe_allow_html=True)
st.markdown("<h3>Deep Learning Flower Classification – Technical Dashboard</h3>", unsafe_allow_html=True)

# ================= DATASET INFO =================
st.markdown("## 📂 Dataset Information")
st.markdown("""
- 5 flower categories  
- Daisy, Dandelion, Rose, Sunflower, Tulip  
- Image augmentation applied  
- Balanced dataset for classification
""")

# ================= UPLOAD =================
uploaded_file = st.file_uploader("📤 Upload Flower Image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    c1, c2, c3 = st.columns([1.2, 1, 1])

    # IMAGE
    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # PREDICTION
    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        img = image.resize((224,224))
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        start = time.time()
        preds = model.predict(img_array)[0]
        inference_time = (time.time() - start) * 1000

        top3 = preds.argsort()[-3:][::-1]
        st.markdown(f"### 🌼 Prediction: **{class_names[top3[0]].capitalize()}**")

        st.markdown("### 🔢 Confidence Scores")
        for i in top3:
            score = preds[i]*100
            st.write(f"{class_names[i].capitalize()} — {score:.2f}%")
            st.progress(int(score))

        st.markdown(f"⏱️ Inference Time: **{inference_time:.2f} ms**")
        st.markdown("</div>", unsafe_allow_html=True)

    # METRICS
    with c3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Model Metrics")
        st.metric("Training Accuracy", "92%")
        st.metric("Validation Accuracy", "88%")
        st.metric("Model Size", "≈11 MB")
        st.metric("Classes", "5")
        st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("""
<p style="text-align:center;opacity:0.6;">
End-to-End CNN System • Model Training • Deployment • UI Engineering
</p>
""", unsafe_allow_html=True)
