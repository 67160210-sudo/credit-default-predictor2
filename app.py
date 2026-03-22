"""
app.py — Credit Risk Predictor (Green Professional UI)
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
# Green Theme UI 💚
# ─────────────────────────────────────────────
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #02140f, #052e26);
}

/* Title */
h1 {
    color: #34d399 !important;
    font-weight: 700;
}

/* Card */
.block-container {
    padding-top: 2rem;
}

/* Input */
.stSelectbox > div, .stSlider > div {
    background-color: rgba(255,255,255,0.05) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(52,211,153,0.2) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #059669, #34d399) !important;
    color: white !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    height: 3em;
    width: 100%;
    border: none;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #047857, #10b981) !important;
}

/* Result */
.stSuccess {
    background-color: rgba(16,185,129,0.15) !important;
    color: #6ee7b7 !important;
}

.stError {
    background-color: rgba(220,38,38,0.15) !important;
    color: #fca5a5 !important;
}

.stInfo {
    background-color: rgba(16,185,129,0.1) !important;
    color: #34d399 !important;
}

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
st.markdown("""
<h1>💳 Credit Risk Predictor</h1>
<p style='color:#9ca3af'>
ระบบประเมินความเสี่ยงการผิดนัดชำระหนี้<br>
<span style='color:#34d399'>Machine Learning · Logistic Regression</span>
</p>
""", unsafe_allow_html=True)

if not model_loaded:
    st.warning("⚠️ ไม่พบไฟล์โมเดล กำลังใช้โมเดลจำลองแทน")

# ─────────────────────────────────────────────
# Input
# ─────────────────────────────────────────────
st.subheader("👤 สถานะผู้ขอสินเชื่อ")

student = st.selectbox(
    "สถานะปัจจุบัน",
    ["ไม่ใช่นักเรียน / นักศึกษา", "นักเรียน / นักศึกษา"]
)

student_val = 1 if "นักเรียน" in student and "ไม่ใช่" not in student else 0

# ─────────────────────────────────────────────
# Financial
# ─────────────────────────────────────────────
st.subheader("💰 ข้อมูลการเงิน")

balance = st.slider("ยอดหนี้คงค้าง (บาท)", 0, 10000, 2000, step=100)
income = st.slider("รายได้ต่อเดือน (บาท)", 0, 10000, 5000, step=100)

# Debt Ratio
debt_ratio = (balance / income * 100) if income > 0 else 100

if debt_ratio < 30:
    st.success(f"Debt Ratio: {debt_ratio:.2f}% (ปลอดภัย)")
elif debt_ratio < 50:
    st.warning(f"Debt Ratio: {debt_ratio:.2f}% (ควรระวัง)")
else:
    st.error(f"Debt Ratio: {debt_ratio:.2f}% (เสี่ยงสูง)")

# ─────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────
if st.button("🔍 ประเมินความเสี่ยง", type="primary"):

    input_data = np.array([[student_val, balance, income]])

    if model_loaded:
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
    else:
        # fallback model
        logit = -8.5 + (-0.6 * student_val) + (0.0055 * balance) + (-0.00008 * income)
        prob = 1 / (1 + np.exp(-logit))
        pred = 1 if prob >= 0.5 else 0

    st.subheader("📊 ผลลัพธ์")

    if pred == 0:
        st.success(f"✅ ความเสี่ยงต่ำ ({(1-prob)*100:.2f}%)")
        st.write("✔ ลูกค้ามีแนวโน้มชำระหนี้ได้ดี")
    else:
        st.error(f"⚠️ ความเสี่ยงสูง ({prob*100:.2f}%)")
        st.write("✗ มีโอกาสผิดนัดชำระหนี้")

    st.write("---")
    st.write(f"💰 หนี้: {balance} บาท")
    st.write(f"💵 รายได้: {income} บาท")
    st.write(f"📊 Debt Ratio: {debt_ratio:.2f}%")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.caption("Model: Logistic Regression | For Educational Use Only")
