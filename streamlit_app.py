import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="FlowerVision AI",
    page_icon="🌸",
    layout="wide"
)

# ================= ULTRA UI CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif;
}

/* Animated gradient background */
body {
    background: linear-gradient(270deg, #ff512f, #dd2476, #24c6dc, #514a9d);
    background-size: 800% 800%;
    animation: gradientMove 18s ease infinite;
}

@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Glass card */
.glass {
    background: rgba(255,255,255,0.18);
    border-radius: 24px;
    padding: 24px;
    backdrop-filter: blur(20px);
    box-shadow: 0 0 25px rgba(255,255,255,0.25);
    transition: transform 0.3s ease;
}

.glass:hover {
    transform: scale(1.02);
}

/* Title */
.hero-title {
    font-size: 52px;
    font-weight: 700;
    text-align: center;
    color: white;
    text-shadow: 0 4px 20px rgba(0,0,0,0.4);
}

.hero-sub {
    text-align: center;
    color: #f1f1f1;
    margin-bottom: 30px;
}

/* Progress label */
.label {
    color: white;
    font-size: 14px;
}

/* Footer */
.footer {
    text-align: center;
    color: white;
    opacity: 0.6;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenet_flower_model.h5")

model = load_model()
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# ================= SIDEBAR =================
st.sidebar.markdown("## 🚀 Live AI Workflow")
st.sidebar.markdown("""
🟢 **Step 1:** Upload Flower Image  
🟢 **Step 2:** Image Resized & Normalized  
🟢 **Step 3:** CNN Feature Extraction  
🟢 **Step 4:** Softmax Classification  
🟢 **Step 5:** Confidence Visualization  

---
### 🧠 Model
• MobileNetV2  
• CNN + Transfer Learning  

### 🌍 Uses
• Agriculture  
• Botany  
• Education  

**Built by Arya Gupta**
""")

# ================= HERO =================
st.markdown("<div class='hero-title'>🌸 FlowerVision AI</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Next-Gen Flower Classification using Deep Learning</div>", unsafe_allow_html=True)

# ================= SAMPLE CARDS =================
st.markdown("### 🌼 Flower Categories")
cols = st.columns(5)
for col, name in zip(cols, class_names):
    with col:
        st.markdown(f"<div class='glass' style='text-align:center;color:white;'><b>{name.capitalize()}</b></div>", unsafe_allow_html=True)

# ================= UPLOAD =================
uploaded_file = st.file_uploader("📤 Upload a flower image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)

        img = image.resize((224,224))
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("⚡ AI Processing..."):
            time.sleep(1)
            preds = model.predict(img_array)[0]

        top = preds.argsort()[-3:][::-1]
        st.markdown(f"<h2 style='color:white;'>🌼 {class_names[top[0]].capitalize()}</h2>", unsafe_allow_html=True)

        for i in top:
            score = preds[i]*100
            st.markdown(f"<div class='label'>{class_names[i].capitalize()} — {score:.2f}%</div>", unsafe_allow_html=True)
            st.progress(int(score))

        st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("<div class='footer'>⚡ Production-grade AI • Live Deployment</div>", unsafe_allow_html=True)
