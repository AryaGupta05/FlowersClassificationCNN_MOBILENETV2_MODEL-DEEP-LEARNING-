import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="FlowerVision AI",
    page_icon="🌸",
    layout="wide"
)

# ================== CUSTOM CSS ==================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

body {
    background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
}

.hero {
    padding: 4rem 2rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.02));
    backdrop-filter: blur(20px);
    text-align: center;
    margin-bottom: 2rem;
}

.card {
    background: rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.3);
}

.label {
    font-size: 14px;
    opacity: 0.7;
}

.big {
    font-size: 32px;
    font-weight: 700;
}

.footer {
    text-align: center;
    opacity: 0.6;
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenet_flower_model.h5")

model = load_model()

class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# ================== SIDEBAR ==================
st.sidebar.markdown("## 🌸 FlowerVision AI")
st.sidebar.markdown("""
**AI Image Classification Platform**

• CNN (MobileNetV2)  
• Transfer Learning  
• Real-time inference  
• Production deployment  

**Use cases**
• Agriculture  
• Botany  
• Education  
• Mobile apps  

Built by **Arya Gupta**
""")

# ================== HERO SECTION ==================
st.markdown("""
<div class="hero">
    <h1>🌸 FlowerVision AI</h1>
    <p>Professional flower classification system powered by deep learning</p>
</div>
""", unsafe_allow_html=True)

# ================== UPLOAD ==================
uploaded_file = st.file_uploader(
    "Upload a flower image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1.1, 1])

    # ---------- IMAGE ----------
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- PREDICTION ----------
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyzing image with deep neural network..."):
            preds = model.predict(img_array)[0]

        top_indices = preds.argsort()[-3:][::-1]

        st.markdown("<p class='label'>Primary Prediction</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p class='big'>{class_names[top_indices[0]].capitalize()}</p>",
            unsafe_allow_html=True
        )

        st.markdown("### Confidence Distribution")

        for idx in top_indices:
            label = class_names[idx].capitalize()
            score = preds[idx] * 100
            st.write(f"**{label}** — {score:.2f}%")
            st.progress(int(score))

        st.markdown("</div>", unsafe_allow_html=True)

# ================== FOOTER ==================
st.markdown("""
<div class="footer">
    🚀 End-to-end Deep Learning System | Live Deployment
</div>
""", unsafe_allow_html=True)
