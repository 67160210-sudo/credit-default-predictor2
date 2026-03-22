"""
app.py — Credit Risk Predictor (Beautiful Edition)
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
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', 'Noto Sans Thai', sans-serif;
}

.stApp { background: #f0fdf4; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 620px !important; padding: 0 1.5rem 4rem !important; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%);
    border-radius: 0 0 24px 24px;
    padding: 32px 32px 28px;
    margin: -1rem -1.5rem 24px;
    box-shadow: 0 8px 32px rgba(6,78,59,0.25);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
}
.hero-tag {
    display: inline-block;
    font-size: 10px; letter-spacing: 2px; text-transform: uppercase; font-weight: 600;
    color: rgba(255,255,255,0.65);
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 100px; padding: 3px 12px; margin-bottom: 12px;
}
.hero-title {
    font-size: 28px; font-weight: 700; color: #fff; margin: 0 0 6px; line-height: 1.2;
}
.hero-title span { color: #6ee7b7; }
.hero-sub { font-size: 13px; color: rgba(255,255,255,0.6); margin: 0; }

/* ── Cards ── */
.card {
    background: #ffffff;
    border: 1px solid #d1fae5;
    border-radius: 16px;
    padding: 22px 24px 18px;
    margin-bottom: 14px;
    box-shadow: 0 2px 12px rgba(6,78,59,0.06);
}
.card-head {
    font-size: 10px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #6b7280;
    margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px;
}
.card-head::after { content: ''; flex: 1; height: 1px; background: #f0fdf4; }

/* ── Widget labels ── */
.stSelectbox label, .stSlider label {
    font-size: 13px !important; font-weight: 600 !important; color: #374151 !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background: #f0fdf4 !important;
    border: 1.5px solid #a7f3d0 !important;
    border-radius: 10px !important;
    color: #065f46 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}

/* ── Debt ratio box ── */
.ratio-box {
    margin-top: 16px;
    padding: 14px 16px;
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
}
.ratio-top {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 8px;
}
.ratio-label { font-size: 12px; font-weight: 600; color: #6b7280; }
.ratio-val { font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 700; }
.ratio-safe { color: #059669; }
.ratio-warn { color: #d97706; }
.ratio-risk { color: #dc2626; }
.ratio-track {
    height: 7px; background: #e5e7eb; border-radius: 100px; overflow: hidden; margin-bottom: 5px;
}
.fill-safe { height:100%; border-radius:100px; background: linear-gradient(90deg,#059669,#34d399); }
.fill-warn { height:100%; border-radius:100px; background: linear-gradient(90deg,#d97706,#fbbf24); }
.fill-risk { height:100%; border-radius:100px; background: linear-gradient(90deg,#dc2626,#f87171); }
.ratio-hint { font-size: 10px; color: #9ca3af; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 14px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 15px !important; font-weight: 600 !important;
    box-shadow: 0 4px 18px rgba(16,185,129,0.3) !important;
    width: 100% !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 24px rgba(16,185,129,0.45) !important;
    transform: translateY(-2px) !important;
}

/* ── Result ── */
.res {
    border-radius: 16px; padding: 26px 26px 20px; margin-top: 4px;
}
.res-safe { background: linear-gradient(135deg,#f0fdf9,#dcfce7); border: 1.5px solid #86efac; }
.res-risk { background: linear-gradient(135deg,#fff7ed,#fee2e2); border: 1.5px solid #fca5a5; }

.res-verdict { font-size: 18px; font-weight: 700; margin-bottom: 4px; }
.res-safe .res-verdict { color: #065f46; }
.res-risk .res-verdict { color: #991b1b; }

.res-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 46px; font-weight: 700; line-height: 1; margin: 6px 0 2px;
}
.res-safe .res-pct { color: #059669; }
.res-risk .res-pct { color: #dc2626; }

.res-sub {
    font-size: 10px; text-transform: uppercase; letter-spacing: 1.5px;
    font-weight: 700; margin-bottom: 16px;
}
.res-safe .res-sub { color: #34d399; }
.res-risk .res-sub { color: #f87171; }

.pbar { background: rgba(0,0,0,0.07); border-radius:100px; height:8px; overflow:hidden; margin-bottom:6px; }
.pbar-s { height:100%; border-radius:100px; background: linear-gradient(90deg,#059669,#34d399); }
.pbar-r { height:100%; border-radius:100px; background: linear-gradient(90deg,#dc2626,#f87171); }

.pbar-lbl {
    display:flex; justify-content:space-between;
    font-size:10px; font-weight:600; letter-spacing:1px;
    text-transform:uppercase; margin-bottom:16px; color:#9ca3af;
}

.mini { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }
.mini-b {
    background: rgba(255,255,255,0.6); border-radius:10px;
    padding:10px 8px 8px; text-align:center;
    border: 1px solid rgba(0,0,0,0.05);
}
.mini-v { font-family:'JetBrains Mono',monospace; font-size:13px; font-weight:700; color:#1e293b; }
.mini-k { font-size:9px; font-weight:600; text-transform:uppercase; letter-spacing:1px; color:#9ca3af; margin-top:2px; }

.adv {
    margin-top:14px; padding:12px 14px; border-radius:10px;
    font-size:12px; line-height:1.6; font-weight:500;
}
.res-safe .adv { background:rgba(5,150,105,0.08); color:#065f46; }
.res-risk .adv { background:rgba(220,38,38,0.08); color:#991b1b; }

.foot { text-align:center; font-size:11px; color:#d1d5db; margin-top:28px; letter-spacing:.5px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "model_artifacts", "model.pkl")
    return joblib.load(path)

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    print("Load error:", e)

# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">💳 AI Credit Analysis</div>
    <div class="hero-title">Credit <span>Risk</span> Predictor</div>
    <p class="hero-sub">ระบบประเมินความเสี่ยงการผิดนัดชำระหนี้ ด้วย Machine Learning</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.warning("⚠️ ไม่พบไฟล์โมเดล กำลังใช้โมเดลจำลองแทน")

# ─────────────────────────────────────────────
# Section A: สถานะผู้ขอสินเชื่อ
# ─────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-head">👤 สถานะผู้ขอสินเชื่อ</div>', unsafe_allow_html=True)

student = st.selectbox(
    "สถานะปัจจุบัน",
    ["ไม่ใช่นักเรียน / นักศึกษา", "นักเรียน / นักศึกษา"],
    help="นักเรียน/นักศึกษามักมีรายได้ไม่แน่นอน"
)
student_val = 1 if student == "นักเรียน / นักศึกษา" else 0

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Section B: ข้อมูลการเงิน
# ─────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-head">💰 ข้อมูลการเงิน</div>', unsafe_allow_html=True)

balance = st.slider("ยอดหนี้คงค้าง (บาท)", 0, 10000, 2000, step=100, format="฿%d")
income  = st.slider("รายได้ต่อเดือน (บาท)", 0, 10000, 5000, step=100, format="฿%d")

# Debt ratio
debt_ratio = (balance / income * 100) if income > 0 else 100
bar_w = min(debt_ratio, 100)

if debt_ratio < 30:
    r_cls, fill_cls, r_status = "ratio-safe", "fill-safe", "✓ ปลอดภัย"
elif debt_ratio < 50:
    r_cls, fill_cls, r_status = "ratio-warn", "fill-warn", "⚠ ควรระวัง"
else:
    r_cls, fill_cls, r_status = "ratio-risk", "fill-risk", "✗ เสี่ยงสูง"

st.markdown(f"""
<div class="ratio-box">
    <div class="ratio-top">
        <span class="ratio-label">Debt-to-Income Ratio</span>
        <span class="ratio-val {r_cls}">{debt_ratio:.1f}% — {r_status}</span>
    </div>
    <div class="ratio-track">
        <div class="{fill_cls}" style="width:{bar_w:.1f}%"></div>
    </div>
    <div class="ratio-hint">น้อยกว่า 30% = ปลอดภัย &nbsp;|&nbsp; 30–50% = ควรระวัง &nbsp;|&nbsp; มากกว่า 50% = เสี่ยง</div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────
predict_btn = st.button("🔍 ประเมินความเสี่ยงสินเชื่อ")

if predict_btn:
    input_data = np.array([[student_val, balance, income]])

    if model_loaded:
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
    else:
        logit = -6.0 + (-0.5 * student_val) + (0.0004 * balance) + (-0.0001 * income) + (0.04 * debt_ratio)
        prob = 1 / (1 + np.exp(-logit))
        if debt_ratio >= 80:
            prob = max(prob, 0.75)
        pred = 1 if prob >= 0.5 else 0

    student_label = "นักเรียน/นักศึกษา" if student_val == 1 else "บุคคลทั่วไป"

    # แสดง input ที่ส่งเข้าโมเดล (debug)
    with st.expander("🔎 ดูข้อมูลที่ส่งเข้าโมเดล"):
        st.write({
            "student": student_val,
            "balance": balance,
            "income": income,
            "debt_ratio": round(debt_ratio, 2),
            "prob_default": round(float(prob), 4),
            "prediction": "เสี่ยง" if pred == 1 else "ไม่เสี่ยง"
        })

    if pred == 0:
        safe_num = (1 - prob) * 100
        safe_str = f"{safe_num:.1f}%"
        st.markdown(f"""
<div class="res res-safe">
    <div class="res-verdict">✅ ความเสี่ยงต่ำ</div>
    <div class="res-pct">{safe_str}</div>
    <div class="res-sub">โอกาสชำระหนี้ได้ปกติ</div>
    <div class="pbar"><div class="pbar-s" style="width:{safe_num:.1f}%"></div></div>
    <div class="pbar-lbl"><span>ปลอดภัย</span><span>เสี่ยงสูง</span></div>
    <div class="mini">
        <div class="mini-b"><div class="mini-v">฿{balance:,}</div><div class="mini-k">ยอดหนี้</div></div>
        <div class="mini-b"><div class="mini-v">฿{income:,}</div><div class="mini-k">รายได้</div></div>
        <div class="mini-b"><div class="mini-v">{debt_ratio:.1f}%</div><div class="mini-k">Debt Ratio</div></div>
    </div>
    <div class="adv">✔ ผู้ขอสินเชื่ออยู่ในเกณฑ์น่าเชื่อถือ สามารถพิจารณาอนุมัติสินเชื่อได้ตามปกติ</div>
</div>
        """, unsafe_allow_html=True)
    else:
        risk_num = prob * 100
        risk_str = f"{risk_num:.1f}%"
        st.markdown(f"""
<div class="res res-risk">
    <div class="res-verdict">⚠️ ตรวจพบความเสี่ยงสูง</div>
    <div class="res-pct">{risk_str}</div>
    <div class="res-sub">โอกาสผิดนัดชำระหนี้</div>
    <div class="pbar"><div class="pbar-r" style="width:{risk_num:.1f}%"></div></div>
    <div class="pbar-lbl"><span>ปลอดภัย</span><span>เสี่ยงสูง</span></div>
    <div class="mini">
        <div class="mini-b"><div class="mini-v">฿{balance:,}</div><div class="mini-k">ยอดหนี้</div></div>
        <div class="mini-b"><div class="mini-v">฿{income:,}</div><div class="mini-k">รายได้</div></div>
        <div class="mini-b"><div class="mini-v">{debt_ratio:.1f}%</div><div class="mini-k">Debt Ratio</div></div>
    </div>
    <div class="adv">✗ แนะนำให้ตรวจสอบประวัติการชำระหนี้และขอเอกสารรายได้เพิ่มเติมก่อนพิจารณาอนุมัติ</div>
</div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div class="foot">Model: Logistic Regression &nbsp;·&nbsp; For Educational Purposes Only</div>
""", unsafe_allow_html=True)
