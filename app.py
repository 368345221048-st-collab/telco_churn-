"""
ระบบทำนายการยกเลิกบริการของลูกค้า (Telco Churn Prediction)
รันด้วย:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- config
APP_TITLE = "ระบบทำนายการยกเลิกบริการลูกค้า"
APP_SUBTITLE = "Telco Churn Prediction"
DEVELOPERS = ["นายกฤษกร พยอมหอม", "นายณนชกร เปรมทอง"]

BASE_DIR = Path(__file__).parent
SCALER_PATH = BASE_DIR / "telco_churn_scaler.joblib"
MODEL_PATH = BASE_DIR / "telco_churn_model.joblib"   # ไฟล์โมเดลที่เทรนแล้ว (ต้องวางเพิ่ม)
METRICS_PATH = BASE_DIR / "metrics.json"             # เก็บค่า accuracy จริง

# ฟีเจอร์ที่เป็น 0/1 ให้แสดงเป็นตัวเลือก (ชื่อคอลัมน์ : (ป้ายกำกับ, ความหมาย 0, ความหมาย 1))
BINARY_FEATURES = {"Sex_female": ("เพศ", "ชาย", "หญิง")}

st.set_page_config(page_title=APP_TITLE, page_icon="🌿", layout="centered")

# ---------------------------------------------------------------- style
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Prompt', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 760px; padding-top: 1.5rem; }
.hero { text-align:center; padding: 8px 0 18px; }
.hero h1 { color:#1F6B41; font-size:1.9rem; margin:6px 0 0; font-weight:600; }
.hero p  { color:#5C8A6E; margin:2px 0 0; letter-spacing:.08em; font-size:.95rem; }
.mascot { width:150px; animation: float 3s ease-in-out infinite; }
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
.card { background:#fff; border:1px solid #D5EBDC; border-radius:18px;
        padding:18px 22px; margin:10px 0; box-shadow:0 2px 10px rgba(46,158,91,.07); }
.result-ok   { background:#E8F7EE; border:1px solid #9BD7B1; }
.result-warn { background:#FFF4E5; border:1px solid #F3C98B; }
.bar { height:12px; border-radius:99px; background:#E3F4E8; overflow:hidden; margin-top:8px; }
.bar > div { height:100%; background:linear-gradient(90deg,#7BD39A,#2E9E5B); }
.stButton > button, .stFormSubmitButton > button {
    background:#2E9E5B; color:#fff; border:0; border-radius:99px;
    padding:.55rem 2rem; font-weight:600; width:100%; }
.stButton > button:hover, .stFormSubmitButton > button:hover { background:#237A46; color:#fff; }
.footer { text-align:center; margin-top:26px; padding:18px; border-top:1px dashed #B7DDC4; }
.footer .acc { font-size:2.2rem; font-weight:600; color:#1F6B41; line-height:1.1; }
.footer .lbl { color:#5C8A6E; font-size:.9rem; }
.footer .dev { color:#3E6B50; margin-top:12px; font-size:.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

MASCOT_SVG = """
<svg class="mascot" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
<ellipse cx="100" cy="188" rx="46" ry="7" fill="#CDE9D6"/>
<line x1="100" y1="34" x2="100" y2="16" stroke="#2E9E5B" stroke-width="5" stroke-linecap="round"/>
<circle cx="100" cy="13" r="8" fill="#F6C453"/>
<rect x="38" y="34" width="124" height="112" rx="52" fill="#8EDBA9" stroke="#2E9E5B" stroke-width="5"/>
<rect x="24" y="76" width="16" height="32" rx="8" fill="#2E9E5B"/>
<rect x="160" y="76" width="16" height="32" rx="8" fill="#2E9E5B"/>
<path d="M32 84 C32 30 168 30 168 84" fill="none" stroke="#2E9E5B" stroke-width="5"/>
<path d="M168 100 Q168 128 138 130" fill="none" stroke="#2E9E5B" stroke-width="4" stroke-linecap="round"/>
<circle cx="136" cy="130" r="5" fill="#2E9E5B"/>
<circle cx="78" cy="86" r="11" fill="#1F3A2B"/><circle cx="122" cy="86" r="11" fill="#1F3A2B"/>
<circle cx="81" cy="82" r="4" fill="#fff"/><circle cx="125" cy="82" r="4" fill="#fff"/>
<circle cx="62" cy="106" r="8" fill="#FFB7B7" opacity=".8"/><circle cx="138" cy="106" r="8" fill="#FFB7B7" opacity=".8"/>
<path d="M86 108 Q100 124 114 108" fill="none" stroke="#1F3A2B" stroke-width="5" stroke-linecap="round"/>
<rect x="66" y="146" width="68" height="34" rx="16" fill="#8EDBA9" stroke="#2E9E5B" stroke-width="5"/>
<path d="M100 154 c-6-7-16 1-8 9 l8 8 l8-8 c8-8-2-16-8-9z" fill="#fff"/>
</svg>
"""


# ---------------------------------------------------------------- loaders
@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def load_accuracy():
    try:
        value = json.loads(METRICS_PATH.read_text(encoding="utf-8")).get("accuracy")
        return float(value) if value is not None else None
    except Exception:
        return None


scaler = load_scaler()
model = load_model()
features = list(getattr(scaler, "feature_names_in_", [f"x{i}" for i in range(scaler.n_features_in_)]))

# ---------------------------------------------------------------- header
st.markdown(
    f'<div class="hero">{MASCOT_SVG}<h1>{APP_TITLE}</h1><p>{APP_SUBTITLE}</p></div>',
    unsafe_allow_html=True,
)

if model is None:
    st.info(
        "ยังไม่พบไฟล์โมเดล `telco_churn_model.joblib` — ตอนนี้ระบบจะแสดงเฉพาะค่าที่ผ่านการปรับสเกลแล้ว "
        "เมื่อวางไฟล์โมเดลไว้โฟลเดอร์เดียวกับ app.py ระบบจะทำนายให้อัตโนมัติ",
        icon="🌱",
    )

# ---------------------------------------------------------------- form
st.markdown("#### 📝 กรอกข้อมูลลูกค้า")
values = {}
with st.form("customer_form"):
    cols = st.columns(2)
    for i, name in enumerate(features):
        col = cols[i % 2]
        mean = float(scaler.mean_[i])
        if name in BINARY_FEATURES:
            label, zero, one = BINARY_FEATURES[name]
            choice = col.selectbox(label, [zero, one], index=int(mean >= 0.5))
            values[name] = float(choice == one)
        else:
            values[name] = col.number_input(name, value=round(mean, 3), step=0.01, format="%.3f")
    submitted = st.form_submit_button("🔍 ทำนายผล")

# ---------------------------------------------------------------- predict
if submitted:
    X = pd.DataFrame([values], columns=features)
    X_scaled = scaler.transform(X)

    if model is None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**ค่าที่ผ่านการปรับสเกลแล้ว (StandardScaler)**")
        st.dataframe(pd.DataFrame(X_scaled, columns=features).round(4), hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X_scaled)[0][1])
        else:
            prob = float(model.predict(X_scaled)[0])
        churn = prob >= 0.5
        css, icon = ("result-warn", "⚠️") if churn else ("result-ok", "😊")
        text = "มีแนวโน้มจะยกเลิกบริการ" if churn else "มีแนวโน้มใช้บริการต่อ"
        st.markdown(
            f'<div class="card {css}"><div style="font-size:1.25rem;font-weight:600">{icon} {text}</div>'
            f'<div style="margin-top:4px">โอกาสยกเลิกบริการ <b>{prob*100:.1f}%</b></div>'
            f'<div class="bar"><div style="width:{prob*100:.1f}%"></div></div></div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------- footer
acc = load_accuracy()
acc_html = f"{acc*100:.2f}%" if acc is not None else "ยังไม่ได้ระบุ"
st.markdown(
    f"""
<div class="footer">
<div class="lbl">ค่าความแม่นยำของระบบ (Accuracy)</div>
<div class="acc">{acc_html}</div>
<div class="dev">พัฒนาโดย {" · ".join(DEVELOPERS)}</div>
</div>
""",
    unsafe_allow_html=True,
)
