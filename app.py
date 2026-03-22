"""
app.py — Credit Risk Predictor (FINAL GREEN VERSION)
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
# CSS (Green Clean UI)
# ─────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}
.stApp { background: #f0fdf4; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 650px !important; }

/* Card */
.card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #d1fae5;
    margin-bottom: 12px;
}

/* Ratio Box */
.ratio-box {
    margin-top: 16px;
    padding: 14px;
    background: #f9fafb;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}
.ratio-top {
    display:flex;
    justify-content:space-between;
    font-weight:600;
}
.ratio-track {
    height:8px;
    background:#e5e7eb;
    border-radius:10px;
    margin-top:8px;
}
.fill-safe { background:#10b981; height:100%; border-radius:10px;}
.fill-warn { background:#f59e0b; height:100%; border-radius:10px;}
.fill-risk { background:#ef4444; height:100%; border-radius:10px;}

/* Button */
.stButton > button {
    background:#10b981 !important;
    color:white !important;
    border-radius:10px !important;
    height:3em;
    font-weight:bold;
}
.stButton > button:hover {
    background:#059669 !important;
}

/* Result */
.res {
    padding:20px;
    border-radius:14px;
    margin-top:10px;
}
.safe { background:#dcfce7; }
.risk { background:#fee2e2; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    return joblib.load(os.path.join(base, "model.pkl"))

try:
    model = load_model()
    model_loaded = True
except:
    model_loaded = False

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("💳 Credit Risk Predictor")
st.caption("ระบบประเมินความเสี่ยงการผิดนัดชำระหนี้")

if not model_loaded:
    st.warning("⚠️ ไม่พบโมเดล ใช้โหมดจำลอง")

# ─────────────────────────────────────────────
# Section A
# ─────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)

student = st.selectbox(
    "สถานะ",
    ["ไม่ใช่นักเรียน / นักศึกษา", "นักเรียน / นักศึกษา"]
)

student_val = 1 if "นักเรียน" in student and "ไม่ใช่" not in student else 0

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Section B
# ─────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)

balance = st.slider("ยอดหนี้", 0, 10000, 2000)
income = st.slider("รายได้", 0, 10000, 5000)

debt_ratio = (balance / income * 100) if income > 0 else 100
bar_w = min(debt_ratio, 100)

if debt_ratio < 30:
    fill = "fill-safe"
    status = "✓ ปลอดภัย"
elif debt_ratio < 50:
    fill = "fill-warn"
    status = "⚠ ระวัง"
else:
    fill = "fill-risk"
    status = "✗ เสี่ยง"

st.markdown(f"""
<div class="ratio-box">
    <div class="ratio-top">
        <span>Debt Ratio</span>
        <span>{debt_ratio:.1f}% - {status}</span>
    </div>
    <div class="ratio-track">
        <div class="{fill}" style="width:{bar_w}%"></div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


