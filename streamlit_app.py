import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="FlowerVision AI 🌸",
    page_icon="🌸",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

body {
    background: linear-gradient(-45deg, #141e30, #243b55, #141e30);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.glass {
    background: rgba(255,255,255,0.12);
    border-radius: 22px;
    padding: 2rem;
    backdrop-filter: blur(18px);
    box-shadow: 0 25px 50px rgba(0,0,0,0.35);
}

.fade {
    animation: fadeIn 1.2s ease-in-out;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

.progress-label {
    font-size: 14px;
    opacity: 0.75;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenet_flower_model.h5")

model = load_model()
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# ================= SIDEBAR (LIVE WORKING STEPS) =================
st.sidebar.markdown("## 🌸 How This AI Works")
st.sidebar.markdown("""
**Live Prediction Pipeline**

**1️⃣ Image Upload**  
User uploads a flower image.

**2️⃣ Preprocessing**  
Image resized to **224×224**, normalized.

**3️⃣ CNN Feature Extraction**  
MobileNetV2 extracts deep visual features.

**4️⃣ Classification Layer**  
Softmax layer predicts flower class.

**5️⃣ Confidence Score**  
Probability distribution shown in real-time.

---

### 🔬 Model Info
- CNN + Transfer Learning  
- MobileNetV2 backbone  
- Lightweight & fast inference  

---

### 🌍 Use Cases
- Agriculture & farming  
- Botanical research  
- Education tools  
- Mobile AI apps  

**Built by Arya Gupta**
""")

# ================= HERO SECTION =================
st.markdown("""
<div class="glass fade" style="text-align:center; margin-bottom:30px;">
    <h1>🌸 FlowerVision AI</h1>
    <p>Professional flower classification system with live deep learning inference</p>
</div>
""", unsafe_allow_html=True)

# ================= SAMPLE IMAGES =================
st.markdown("### 🌼 Sample Flower Categories")
sample_cols = st.columns(5)
samples = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]

for col, name in zip(sample_cols, samples):
    with col:
        st.markdown(f"""
        <div class="glass fade" style="text-align:center; padding:10px;">
            <p>{name}</p>
        </div>
        """, unsafe_allow_html=True)

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader(
    "📤 Upload a flower image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1.2, 1])

    # -------- IMAGE CARD --------
    with col1:
        st.markdown("<div class='glass fade'>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # -------- PREDICTION CARD --------
    with col2:
        st.markdown("<div class='glass fade'>", unsafe_allow_html=True)

        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("🧠 AI is analyzing the image..."):
            time.sleep(1)
            preds = model.predict(img_array)[0]

        top_indices = preds.argsort()[-3:][::-1]

        st.markdown("### 🌼 Prediction Result")
        st.markdown(
            f"<h2>{class_names[top_indices[0]].capitalize()}</h2>",
            unsafe_allow_html=True
        )

        st.markdown("### 📊 Confidence Distribution")
        for idx in top_indices:
            label = class_names[idx].capitalize()
            score = preds[idx] * 100
            st.markdown(f"<span class='progress-label'>{label} — {score:.2f}%</span>", unsafe_allow_html=True)
            st.progress(int(score))

        st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("""
<div style="text-align:center; opacity:0.6; margin-top:40px;">
🚀 Live AI System • CNN • Streamlit Deployment
</div>
""", unsafe_allow_html=True)
