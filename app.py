"""
app.py — Credit Risk Predictor (Beautiful Edition)
ระบบทำนายความเสี่ยงการผิดนัดชำระหนี้
"""

import streamlit as st
import numpy as np
import joblib

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="centered"
)

# ─────────────────────────────────────────────
# CSS — Dark Financial Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@300;400;500;600;700&family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', 'Noto Sans Thai', sans-serif;
}

.stApp {
    background: linear-gradient(145deg, #0a0e1a 0%, #0f1724 60%, #0a1628 100%);
}

#MainMenu, footer { visibility: hidden; }

.block-container {
    max-width: 600px !important;
    padding: 0 1.5rem 4rem !important;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #0f2444 0%, #1a3a6b 50%, #0d2d5e 100%);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 0 0 24px 24px;
    padding: 32px 32px 28px;
    margin: -1rem -1.5rem 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
}
.hero-tag {
    display: inline-block;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 600;
    color: rgba(255,255,255,0.6);
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 100px;
    padding: 3px 12px;
    margin-bottom: 14px;
}
.hero-title {
    font-size: 28px;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 8px;
    line-height: 1.2;
}
.hero-title span { color: #63b3ed; }
.hero-sub {
    font-size: 13px;
    color: rgba(255,255,255,0.55);
    margin: 0;
    line-height: 1.6;
}

/* Section card */
.sec-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 22px 24px 18px;
    margin-bottom: 14px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.2);
}
.sec-head {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4a5568;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec-head::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.05);
}

/* Helper text */
.helper {
    display: inline-block;
    font-size: 11px;
    color: #718096;
    background: rgba(255,255,255,0.04);
    border-radius: 6px;
    padding: 2px 8px;
    margin-bottom: 4px;
    border: 1px solid rgba(255,255,255,0.06);
}

/* Widget overrides */
.stSelectbox label,
.stSlider label,
.stNumberInput label {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #a0aec0 !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1.5px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 14px !important;
}
.stSelectbox [data-baseweb="select"] > div:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}

/* Slider value display */
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {
    color: #4a5568 !important;
    font-size: 11px !important;
}

/* Debt ratio bar */
.ratio-wrap {
    margin-top: 14px;
    padding: 14px 16px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
}
.ratio-label {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #718096;
    font-weight: 600;
    margin-bottom: 8px;
}
.ratio-val { color: #e2e8f0; font-family: 'JetBrains Mono', monospace; }
.ratio-track {
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    overflow: hidden;
    margin-bottom: 6px;
}
.ratio-fill-safe { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #059669, #34d399); }
.ratio-fill-warn { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #d97706, #fbbf24); }
.ratio-fill-risk { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #dc2626, #f87171); }
.ratio-hint { font-size: 11px; color: #4a5568; }

/* Predict button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    color: white !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.3) !important;
    letter-spacing: 0.3px !important;
    width: 100% !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 28px rgba(59,130,246,0.5) !important;
    transform: translateY(-2px) !important;
}

/* Result boxes */
.res-box {
    border-radius: 16px;
    padding: 26px 26px 20px;
    margin-top: 4px;
}
.res-safe {
    background: linear-gradient(135deg, rgba(6,78,59,0.3) 0%, rgba(4,120,87,0.15) 100%);
    border: 1.5px solid rgba(52,211,153,0.3);
}
.res-risk {
    background: linear-gradient(135deg, rgba(127,29,29,0.3) 0%, rgba(185,28,28,0.15) 100%);
    border: 1.5px solid rgba(248,113,113,0.3);
}
.res-verdict {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 4px;
}
.res-safe .res-verdict { color: #6ee7b7; }
.res-risk .res-verdict { color: #fca5a5; }

.res-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 44px;
    font-weight: 700;
    line-height: 1;
    margin: 6px 0 2px;
}
.res-safe .res-pct { color: #34d399; }
.res-risk .res-pct { color: #f87171; }

.res-sub {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    margin-bottom: 16px;
}
.res-safe .res-sub { color: #059669; }
.res-risk .res-sub { color: #dc2626; }

.pbar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
    margin-bottom: 6px;
}
.pbar-safe { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #059669, #34d399); }
.pbar-risk { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #dc2626, #f87171); }

.pbar-labels {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 16px;
    color: #4a5568;
}

.mini-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
}
.mini-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 10px 8px 8px;
    text-align: center;
}
.mini-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    color: #e2e8f0;
}
.mini-key {
    font-size: 9px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #4a5568;
    margin-top: 2px;
}

.advice {
    margin-top: 14px;
    padding: 12px 14px;
    border-radius: 10px;
    font-size: 12px;
    line-height: 1.6;
    font-weight: 500;
}
.res-safe .advice { background: rgba(5,150,105,0.1); color: #6ee7b7; }
.res-risk .advice { background: rgba(220,38,38,0.1); color: #fca5a5; }

.foot {
    text-align: center;
    font-size: 11px;
    color: #2d3748;
    margin-top: 28px;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model_artifacts/model.pkl")

try:
    model = load_model()
    model_loaded = True
except Exception:
    model_loaded = False

# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">💳 AI Credit Analysis</div>
    <div class="hero-title">Credit <span>Risk</span> Predictor</div>
    <p class="hero-sub">ระบบประเมินความเสี่ยงการผิดนัดชำระหนี้<br/>ด้วย Machine Learning · Logistic Regression</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.warning("⚠️ ไม่พบไฟล์โมเดล กำลังใช้โมเดลจำลองแทน")

# ─────────────────────────────────────────────
# Section A: สถานะผู้ขอสินเชื่อ
# ─────────────────────────────────────────────
st.markdown('<div class="sec-card"><div class="sec-head">👤 สถานะผู้ขอสินเชื่อ</div>', unsafe_allow_html=True)

student = st.selectbox(
    "สถานะปัจจุบัน",
    ["ไม่ใช่นักเรียน / นักศึกษา", "นักเรียน / นักศึกษา"],
    help="นักเรียน/นักศึกษามักมีรายได้ไม่แน่นอน"
)
student_val = 1 if "นักเรียน" in student and "ไม่ใช่" not in student else 0

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Section B: ยอดหนี้และรายได้
# ─────────────────────────────────────────────
st.markdown('<div class="sec-card"><div class="sec-head">💰 ข้อมูลการเงิน</div>', unsafe_allow_html=True)

st.markdown('<span class="helper">สูงสุด ฿10,000</span>', unsafe_allow_html=True)
balance = st.slider("ยอดหนี้คงค้าง (บาท)", min_value=0, max_value=10000, value=2000, step=100, format="฿%d")

st.markdown('<span class="helper">สูงสุด ฿10,000</span>', unsafe_allow_html=True)
income = st.slider("รายได้ต่อเดือน (บาท)", min_value=0, max_value=10000, value=5000, step=100, format="฿%d")

# Debt ratio indicator
debt_ratio = (balance / income * 100) if income > 0 else 100

if debt_ratio < 30:
    ratio_cls = "ratio-fill-safe"
    ratio_status = "✓ อยู่ในเกณฑ์ดี"
elif debt_ratio < 50:
    ratio_cls = "ratio-fill-warn"
    ratio_status = "⚠ ควรระวัง"
else:
    ratio_cls = "ratio-fill-risk"
    ratio_status = "✗ เสี่ยงสูง"

bar_ratio_w = min(debt_ratio, 100)

st.markdown(f"""
<div class="ratio-wrap">
    <div class="ratio-label">
        <span>Debt-to-Income Ratio</span>
        <span class="ratio-val">{debt_ratio:.1f}% — {ratio_status}</span>
    </div>
    <div class="ratio-track">
        <div class="{ratio_cls}" style="width:{bar_ratio_w}%"></div>
    </div>
    <div class="ratio-hint">ยอดหนี้ ÷ รายได้ | ต่ำกว่า 30% = ปลอดภัย / 30-50% = ควรระวัง / มากกว่า 50% = เสี่ยง</div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Predict Button
# ─────────────────────────────────────────────
predict_btn = st.button("🔍 ประเมินความเสี่ยงสินเชื่อ", type="primary")

# ─────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────
if predict_btn:
    input_data = np.array([[student_val, balance, income]])

    if model_loaded:
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
    else:
        # จำลอง Logistic Regression
        logit = -8.5 + (-0.6 * student_val) + (0.0055 * balance) + (-0.00008 * income)
        prob = 1 / (1 + np.exp(-logit))
        pred = 1 if prob >= 0.5 else 0

    pct = prob * 100

    if pred == 0:
        css_cls = "res-safe"
        verdict = "✅ ความเสี่ยงต่ำ"
        display_pct = f"{(1-prob)*100:.1f}%"
        pct_label = "โอกาสชำระหนี้ได้ปกติ"
        pbar_cls = "pbar-safe"
        bar_w = f"{(1-prob)*100:.1f}%"
        advice = "✔ ผู้ขอสินเชื่ออยู่ในเกณฑ์น่าเชื่อถือ สามารถพิจารณาอนุมัติสินเชื่อได้ตามปกติ"
    else:
        css_cls = "res-risk"
        verdict = "⚠️ มีความเสี่ยงสูง"
        display_pct = f"{pct:.1f}%"
        pct_label = "โอกาสผิดนัดชำระหนี้"
        pbar_cls = "pbar-risk"
        bar_w = f"{pct:.1f}%"
        advice = "✗ ตรวจพบความเสี่ยงสูง แนะนำให้ตรวจสอบประวัติการชำระหนี้และขอเอกสารรายได้เพิ่มเติม"

    student_label = "นักเรียน" if student_val == 1 else "ทั่วไป"

    st.markdown(f"""
<div class="res-box {css_cls}">
    <div class="res-verdict">{verdict}</div>
    <div class="res-pct">{display_pct}</div>
    <div class="res-sub">{pct_label}</div>
    <div class="pbar-wrap">
        <div class="{pbar_cls}" style="width:{bar_w}"></div>
    </div>
    <div class="pbar-labels"><span>ปลอดภัย</span><span>เสี่ยงสูง</span></div>
    <div class="mini-grid">
        <div class="mini-box">
            <div class="mini-val">฿{balance:,}</div>
            <div class="mini-key">ยอดหนี้</div>
        </div>
        <div class="mini-box">
            <div class="mini-val">฿{income:,}</div>
            <div class="mini-key">รายได้</div>
        </div>
        <div class="mini-box">
            <div class="mini-val">{debt_ratio:.1f}%</div>
            <div class="mini-key">Debt Ratio</div>
        </div>
    </div>
    <div class="advice">{advice}</div>
</div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div class="foot">
    Model: Logistic Regression &nbsp;·&nbsp; For Educational Purposes Only
</div>
""", unsafe_allow_html=True)