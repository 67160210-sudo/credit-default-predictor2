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
# CSS (แก้แล้ว)
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

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 14px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    width: 100% !important;
}
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
