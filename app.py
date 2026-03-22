"""
app.py — Credit Risk Predictor (Clean Green UI)
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
# Clean Green Theme 💚 (อ่านง่าย)
# ─────────────────────────────────────────────
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #f0fdf4;
}

/* Text */
html, body, [class*="css"]  {
    color: #111827;
}

/* Title */
h1 {
    color: #059669 !important;
    font-weight: 700;
}

/* Subtitle */
.stCaption {
    color: #4b5563 !important;
}

/* Input */
.stSelectbox > div, .stSlider > div {
    background-color: #ffffff !important;
    border: 1px solid #d1fae5 !important;
    border-radius: 10px !important;
}

/* Button */
.stButton > button {
    background: #10b981 !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 10px !important;
    height: 3em;
    border: none;
}

.stButton > button:hover {
    background: #059669 !important;
}

/* Info */
.stInfo {
    background-color: #d1fae5 !important;
    color: #065f46 !important;
}

/* Success */
.stSuccess {
    background-color: #bbf7d0 !important;
    color: #065f46 !important;
}

/* Error */
.stError {
    background-color: #fecaca !important;
    color: #7f1d1d !important;
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

if debt_ratio < 30:
    st.success(f"Debt Ratio: {debt_ratio:.2f}% (ปลอดภัย)")
elif debt_ratio < 50:
    st.info(f"Debt Ratio: {debt_ratio:.2f}% (ควรระวัง)")
else:
    st.error(f"Debt Ratio: {debt_ratio:.2f}% (เสี่ยงสูง)")

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
