import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import plotly.graph_objects as go

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="FlowerVision AI 3D",
    page_icon="🌸",
    layout="wide"
)

# ================= ADVANCED CSS =================
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #0f2027, #000000);
}
.neon-card {
    background: rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 20px;
    backdrop-filter: blur(18px);
    box-shadow: 0 0 25px rgba(0,255,255,0.4);
}
.neon-title {
    color: #00fff7;
    text-shadow: 0 0 20px #00fff7;
}
.step {
    padding: 12px;
    margin-bottom: 8px;
    border-left: 4px solid #00fff7;
    color: #eaeaea;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenet_flower_model.h5")

model = load_model()
class_names = ['daisy','dandelion','rose','sunflower','tulip']

# ================= HEADER =================
st.markdown("<h1 class='neon-title'>🌸 FlowerVision AI</h1>", unsafe_allow_html=True)
st.markdown("### Interactive Deep Learning Dashboard")

# ================= LAYOUT =================
left, center, right = st.columns([1.2, 1.6, 1.2])

# ================= LEFT PANEL =================
with left:
    st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
    st.markdown("## 🧠 AI Pipeline")

    steps = [
        "Image Upload",
        "Resize & Normalize",
        "CNN Feature Extraction",
        "Softmax Probability",
        "Prediction Output"
    ]

    for s in steps:
        st.markdown(f"<div class='step'>✔ {s}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ================= CENTER =================
with center:
    st.markdown("<div class='neon-card'>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload Flower Image", type=["jpg","png","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)

        img = image.resize((224,224))
        arr = np.expand_dims(np.array(img)/255.0, axis=0)

        if st.button("🚀 Run AI Prediction"):
            with st.spinner("⚡ Deep Neural Network Processing..."):
                start = time.time()
                preds = model.predict(arr)[0]
                infer_time = (time.time() - start) * 1000

            top = preds.argsort()[-3:][::-1]
            st.success(f"🌼 Prediction: {class_names[top[0]].capitalize()}")
            st.info(f"⏱ Inference Time: {infer_time:.2f} ms")

            st.session_state["preds"] = preds
            st.session_state["top"] = top

    st.markdown("</div>", unsafe_allow_html=True)

# ================= RIGHT PANEL =================
with right:
    st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
    st.markdown("## 📊 Confidence Analysis")

    if "preds" in st.session_state:
        preds = st.session_state["preds"]
        top = st.session_state["top"]

        for i in top:
            st.write(f"{class_names[i].capitalize()} — {preds[i]*100:.2f}%")
            st.progress(int(preds[i]*100))

        # 3D-like interactive chart (Plotly)
        fig = go.Figure(
            data=[go.Bar(
                x=[class_names[i].capitalize() for i in top],
                y=[preds[i]*100 for i in top],
                marker_color=["#00fff7","#ff00ff","#ffaa00"]
            )]
        )
        fig.update_layout(
            title="Prediction Confidence",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
