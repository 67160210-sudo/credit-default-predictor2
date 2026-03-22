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
# CSS (ของเดิม 100% แก้ syntax แล้ว)
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

    path1 = os.path.join(base, "model.pkl")
    path2 = os.path.join(base, "model_artifacts", "model.pkl")

    if os.path.exists(path1):
        path = path1
    elif os.path.exists(path2):
        path = path2
    else:
        raise FileNotFoundError(f"ไม่พบ model.pkl\n{path1}\n{path2}")

    model = joblib.load(path)

    if not hasattr(model, "predict"):
        raise ValueError("โมเดลไม่มี predict()")

    return model

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

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_data)[0][1]
            else:
                prob = 0.5

        except Exception as e:
            st.error(f"❌ โมเดลพังตอน predict: {e}")
            model_loaded = False

    if not model_loaded:
        # fallback model
        logit = -6.0 + (-0.5 * student_val) + (0.0004 * balance) + (-0.0001 * income) + (0.04 * debt_ratio)
        prob = 1 / (1 + np.exp(-logit))
        pred = 1 if prob >= 0.5 else 0

    # debug
    with st.expander("🔎 Debug"):
        st.write({
            "student": student_val,
            "balance": balance,
            "income": income,
            "debt_ratio": round(debt_ratio, 2),
            "prob": round(float(prob), 4),
            "prediction": int(pred)
        })

    # Result
    if pred == 0:
        st.success(f"✅ ความเสี่ยงต่ำ ({(1-prob)*100:.1f}%)")
    else:
        st.error(f"⚠️ ความเสี่ยงสูง ({prob*100:.1f}%)")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.caption("Model: Logistic Regression · Demo")
