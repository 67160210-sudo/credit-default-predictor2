"""
app.py — Credit Risk Predictor (Fixed Version)
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
# Load Model (FIXED ✅)
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
# UI Header
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
if st.button("🔍 ประเมินความเสี่ยง", type="primary"):

    input_data = np.array([[student_val, balance, income]])

    if model_loaded:
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
    else:
        # Mock model (fallback)
        logit = -8.5 + (-0.6 * student_val) + (0.0055 * balance) + (-0.00008 * income)
        prob = 1 / (1 + np.exp(-logit))
        pred = 1 if prob >= 0.5 else 0

    # ─────────────────────────────────────────
    # Result
    # ─────────────────────────────────────────
    st.subheader("📊 ผลการประเมิน")

    if pred == 0:
        st.success(f"✅ ความเสี่ยงต่ำ ({(1-prob)*100:.2f}%)")
        st.write("✔ สามารถพิจารณาอนุมัติสินเชื่อได้")
    else:
        st.error(f"⚠️ ความเสี่ยงสูง ({prob*100:.2f}%)")
        st.write("✗ ควรตรวจสอบเพิ่มเติมก่อนอนุมัติ")

    st.write("---")
    st.write(f"ยอดหนี้: {balance} บาท")
    st.write(f"รายได้: {income} บาท")
    st.write(f"Debt Ratio: {debt_ratio:.2f}%")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.caption("Model: Logistic Regression | For Education Use Only")
