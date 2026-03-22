import streamlit as st
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="centered"
)

# ─────────────────────────────────────────────
# CSS (เหลือเฉพาะที่จำเป็น)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* พื้นหลังเขียว */
html, body, .stApp {
    background: linear-gradient(135deg, #a7f3d0, #34d399) !important;
}

/* ปรับ container */
.block-container {
    max-width: 600px;
}

/* ปุ่ม */
.stButton > button {
    background: #059669 !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 12px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))

    paths = [
        os.path.join(base, "model.pkl"),
        os.path.join(base, "model_artifacts", "model.pkl")
    ]

    for path in paths:
        if os.path.exists(path):
            model = joblib.load(path)
            if hasattr(model, "predict"):
                return model

    raise FileNotFoundError("ไม่พบ model.pkl")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("💳 Credit Risk Predictor")

student = st.selectbox(
    "สถานะ",
    ["ไม่ใช่นักเรียน / นักศึกษา", "นักเรียน / นักศึกษา"]
)
student_val = 1 if student == "นักเรียน / นักศึกษา" else 0

balance = st.slider("ยอดหนี้", 0, 10000, 2000)
income  = st.slider("รายได้", 0, 10000, 5000)

debt_ratio = (balance / income * 100) if income > 0 else 100

# ─────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────
if st.button("🔍 ประเมินความเสี่ยงสินเชื่อ"):

    input_data = np.array([[student_val, balance, income]])

    if model_loaded:
        try:
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else 0.5
        except:
            model_loaded = False

    if not model_loaded:
        logit = -6.0 + (-0.5 * student_val) + (0.0004 * balance) + (-0.0001 * income) + (0.04 * debt_ratio)
        prob = 1 / (1 + np.exp(-logit))
        pred = 1 if prob >= 0.5 else 0

    # Result
    if pred == 0:
        st.success(f"✅ ความเสี่ยงต่ำ ({(1-prob)*100:.1f}%)")
    else:
        st.error(f"⚠️ ความเสี่ยงสูง ({prob*100:.1f}%)")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.caption("Model: Logistic Regression")
