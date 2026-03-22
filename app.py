"""
app.py — Credit Risk Predictor (Light Green Theme)
"""

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
# Green Light Theme 💚
# ─────────────────────────────────────────────
st.markdown("""
<style>

/* 🌿 Background */
.stApp {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
}

/* Text */
html, body, [class*="css"]  {
    color: #064e3b;
}

/* Title */
h1 {
    color: #059669 !important;
}

/* Caption */
.stCaption {
    color: #065f46 !important;
}

/* Input */
.stSelectbox > div, .stSlider > div {
    background-color: white !important;
    border: 1px solid #a7f3d0 !important;
    border-radius: 10px !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #34d399, #6ee7b7) !important;
    color: #064e3b !important;
    font-weight: bold !important;
    border-radius: 12px !important;
    height: 3em;
    border: none;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #10b981, #34d399) !important;
}

/* Result */
.stSuccess {
    background-color: #d1fae5 !important;
    color: #065f46 !important;
}

.stError {
    background-color: #fee2e2 !important;
    color: #991b1b !important;
}

.stInfo {
    background-color: #ecfdf5 !important;
    color: #047857 !important;
}

/* Footer */
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "model.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    print("ERROR:", e)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("💳 Credit Risk Predictor")
st.caption("ระบบประเมินความเสี่ยงการผิดนัดชำระหนี้ ด้วย Machine Learning")

if not model_loaded:
    st.warning("⚠️ ไม่พบไฟล์โมเดล กำลังใช้โมเดลจำลองแทน")

# ─────────────────────────────────────────────
# Input Section
# ─────────────────────────────────────────────
st.subheader("👤 สถานะผู้ขอสินเชื่อ")

student = st.selectbox(
    "สถานะปัจจุบัน",
    ["ไม่ใช่นักเรียน / นักศึกษา", "นักเรียน / นักศึกษา"]
)

student_val = 1 if "นักเรียน" in student and "ไม่ใช่" not in student else 0

# ─────────────────────────────────────────────
# Financial Input
# ─────────────────────────────────────────────
st.subheader("💰 ข้อมูลการเงิน")

balance = st.slider("ยอดหนี้คงค้าง (บาท)", 0, 10000, 2000, step=100)
income = st.slider("รายได้ต่อเดือน (บาท)", 0, 10000, 5000, step=100)

# Debt Ratio
debt_ratio = (balance / income * 100) if income > 0 else 100
st.info(f"Debt-to-Income Ratio: {debt_ratio:.2f}%")

# ─────────────────────────────────────────────
# Predict Button
# ─────────────────────────────────────────────
if st.button("🔍 ประเมินความเสี่ยง", type="primary", use_container_width=True):

    input_data = np.array([[student_val, balance, income]])

    if model_loaded:
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
    else:
        # Mock model
        logit = -8.5 + (-0.6 * student_val) + (0.0055 * balance) + (-0.00008 * income)
        prob = 1 / (1 + np.exp(-logit))
        pred = 1 if prob >= 0.5 else 0

    # Result
    st.subheader("📊 ผลการประเมิน")

    if pred == 0:
        st.success(f"✅ ความเสี่ยงต่ำ ({(1-prob)*100:.2f}%)")
        st.write("✔ ลูกค้ามีแนวโน้มชำระหนี้ได้ดี")
    else:
        st.error(f"⚠️ ความเสี่ยงสูง ({prob*100:.2f}%)")
        st.write("✗ มีโอกาสผิดนัดชำระหนี้")

    st.write("---")
    st.write(f"💰 ยอดหนี้: {balance} บาท")
    st.write(f"💵 รายได้: {income} บาท")
    st.write(f"📊 Debt Ratio: {debt_ratio:.2f}%")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.caption("Model: Logistic Regression | For Educational Use Only")
