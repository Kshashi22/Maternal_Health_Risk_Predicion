"""
Maternal Health Risk Prediction — Streamlit App
------------------------------------------------
Loads the model + scaler you trained/saved in your notebook
(maternal_risk_model.pkl, scaler.pkl) and lets a user enter
patient values through a form to get a live risk prediction.

Run with:
    streamlit run app.py

Make sure maternal_risk_model.pkl and scaler.pkl are in:
    model/maternal_risk_model.pkl
    model/scaler.pkl
(same folder as this app.py, or update the paths below).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================================================
# PAGE CONFIG
# ==============================================================
st.set_page_config(
    page_title="Maternal Health Risk Predictor",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================
# GLOBAL STYLE
# ==============================================================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        /* Hide default Streamlit chrome */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1150px;
        }

        /* Hero banner */
        .hero {
            background: linear-gradient(120deg, #6a5af9 0%, #d16ba5 100%);
            border-radius: 20px;
            padding: 2.2rem 2.5rem;
            margin-bottom: 1.8rem;
            box-shadow: 0 10px 30px rgba(106, 90, 249, 0.25);
            color: white;
        }
        .hero h1 {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 2.1rem;
            margin: 0 0 0.4rem 0;
            color: white;
        }
        .hero p {
            font-size: 1.02rem;
            opacity: 0.92;
            margin: 0;
            max-width: 700px;
        }

        /* Section card wrapper */
        .card {
            background: var(--background-color, #ffffff);
            border: 1px solid rgba(120,120,140,0.15);
            border-radius: 18px;
            padding: 1.6rem 1.8rem;
            box-shadow: 0 4px 18px rgba(30,30,60,0.06);
            margin-bottom: 1.4rem;
        }

        .section-title {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 1.15rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.9rem;
        }

        /* Inputs */
        div[data-testid="stNumberInput"] label, div[data-testid="stSelectbox"] label {
            font-weight: 500;
            font-size: 0.88rem;
            opacity: 0.85;
        }
        div[data-testid="stNumberInput"] input, div[data-baseweb="select"] {
            border-radius: 10px !important;
        }

        /* Submit button */
        div.stButton > button, button[kind="formSubmit"] {
            background: linear-gradient(120deg, #6a5af9 0%, #d16ba5 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.7rem 1rem;
            font-weight: 600;
            font-size: 1rem;
            letter-spacing: 0.2px;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            box-shadow: 0 6px 16px rgba(106, 90, 249, 0.3);
        }
        div.stButton > button:hover, button[kind="formSubmit"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 22px rgba(106, 90, 249, 0.4);
        }

        /* Result banner */
        .result-banner {
            border-radius: 16px;
            padding: 1.6rem 1.8rem;
            display: flex;
            align-items: center;
            gap: 1.2rem;
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
            margin-bottom: 1rem;
        }
        .result-emoji {
            font-size: 2.6rem;
            line-height: 1;
        }
        .result-title {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 1.5rem;
            margin: 0;
        }
        .result-sub {
            font-size: 0.92rem;
            opacity: 0.85;
            margin: 0.15rem 0 0 0;
        }

        /* Confidence pills */
        .conf-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
            font-size: 0.92rem;
        }
        .conf-bar-bg {
            background: rgba(120,120,140,0.15);
            border-radius: 8px;
            height: 10px;
            width: 100%;
            overflow: hidden;
            margin-bottom: 0.9rem;
        }
        .conf-bar-fill {
            height: 100%;
            border-radius: 8px;
        }

        .footer-note {
            text-align: center;
            font-size: 0.82rem;
            opacity: 0.65;
            margin-top: 1.5rem;
        }

        .sidebar-title {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 1.05rem;
            margin-bottom: 0.4rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================
# LOAD MODEL + SCALER (cached so it only loads once)
# ==============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/maternal_risk_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error(
        "Could not find `maternal_risk_model.pkl` or `scaler.pkl`. "
        "Place both files inside a `model/` folder next to app.py and restart."
    )
    st.stop()

# ==============================================================
# CONSTANTS — must match your notebook's encoding EXACTLY
# ==============================================================
ACTIVITY_MAP = {"Low": 0, "Moderate": 1, "High": 2}

# LabelEncoder sorted these alphabetically during training:
# high risk = 0, low risk = 1, mid risk = 2
RISK_LABELS = {0: "High Risk", 1: "Low Risk", 2: "Mid Risk"}

RISK_STYLE = {
    "High Risk": {"color": "#e63946", "bg": "#fdecee", "emoji": "🔴",
                  "sub": "Elevated risk detected — please consult a healthcare provider."},
    "Mid Risk":  {"color": "#f4a261", "bg": "#fef3e8", "emoji": "🟠",
                  "sub": "Moderate risk — regular monitoring is recommended."},
    "Low Risk":  {"color": "#2a9d8f", "bg": "#e9f7f5", "emoji": "🟢",
                  "sub": "Low risk based on current vitals."},
}

# Exact column order the scaler was fit on
FEATURE_ORDER = [
    "Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate",
    "BMI", "PreviousComplications", "GestationalAgeWeeks",
    "HemoglobinLevel", "PhysicalActivityLevel", "PulsePressure", "MAP"
]

# ==============================================================
# SIDEBAR
# ==============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">🤰 About this tool</div>', unsafe_allow_html=True)
    st.write(
        "This predictor uses a **Random Forest** model trained on maternal "
        "vitals to estimate pregnancy risk level as **Low**, **Mid**, or **High**."
    )
    st.markdown("---")
    st.markdown('<div class="sidebar-title">📊 Model snapshot</div>', unsafe_allow_html=True)
    st.metric("Test Accuracy", "~97.5%")
    st.metric("Algorithm", "Random Forest")
    st.markdown("---")
    st.caption(
        "⚕️ Educational/demo purposes only — not a substitute for professional "
        "medical advice."
    )

# ==============================================================
# HERO HEADER
# ==============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🤰 Maternal Health Risk Predictor</h1>
        <p>Enter a patient's vitals to instantly estimate their pregnancy risk
        level using a trained machine learning model — built to support
        earlier detection and better maternal care.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================
# INPUT FORM
# ==============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📝 Patient Vitals</div>', unsafe_allow_html=True)

with st.form("risk_form"):
    tab1, tab2, tab3 = st.tabs(["🫀 Vitals", "🧬 Body & Blood", "🤰 Pregnancy"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", min_value=10, max_value=70, value=28)
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=160, value=115)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=110, value=75)
        with c2:
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=60, max_value=110, value=76)
            body_temp = st.number_input("Body Temperature (°F)", min_value=96.0, max_value=103.0, value=98.0, step=0.1)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            bs = st.number_input("Blood Sugar (mmol/L)", min_value=6.0, max_value=19.0, value=7.5, step=0.1)
            bmi = st.number_input("BMI", min_value=15.0, max_value=45.0, value=24.0, step=0.1)
        with c2:
            hemoglobin = st.number_input("Hemoglobin Level (g/dL)", min_value=6.0, max_value=16.0, value=11.5, step=0.1)
            activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            gest_age = st.number_input("Gestational Age (weeks)", min_value=4, max_value=42, value=28)
        with c2:
            prev_complications = st.selectbox("Previous Complications?", ["No", "Yes"])

    st.write("")
    submitted = st.form_submit_button("🔍 Predict Risk Level", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================
# PREDICTION LOGIC (unchanged from the trained pipeline)
# ==============================================================
if submitted:
    # --- Build raw input row ---
    input_dict = {
        "Age": age,
        "SystolicBP": systolic_bp,
        "DiastolicBP": diastolic_bp,
        "BS": bs,
        "BodyTemp": body_temp,
        "HeartRate": heart_rate,
        "BMI": bmi,
        "PreviousComplications": 1 if prev_complications == "Yes" else 0,
        "GestationalAgeWeeks": gest_age,
        "HemoglobinLevel": hemoglobin,
        "PhysicalActivityLevel": ACTIVITY_MAP[activity],
    }

    # --- Feature engineering (same as training) ---
    input_dict["PulsePressure"] = systolic_bp - diastolic_bp
    input_dict["MAP"] = (systolic_bp + 2 * diastolic_bp) / 3

    # --- Build DataFrame in exact training column order ---
    input_df = pd.DataFrame([input_dict])[FEATURE_ORDER]

    # --- Scale using the SAME fitted scaler (no re-fit) ---
    input_scaled = scaler.transform(input_df)

    # --- Predict ---
    pred_class = model.predict(input_scaled)[0]
    pred_label = RISK_LABELS[pred_class]

    pred_proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0]
        pred_proba = {RISK_LABELS[i]: round(p * 100, 1) for i, p in enumerate(proba)}

    # ==============================================================
    # RESULT DISPLAY
    # ==============================================================
    style = RISK_STYLE[pred_label]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Prediction Result</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="result-banner" style="background:{style['bg']}; border-left: 6px solid {style['color']};">
            <div class="result-emoji">{style['emoji']}</div>
            <div>
                <p class="result-title" style="color:{style['color']};">{pred_label}</p>
                <p class="result-sub">{style['sub']}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if pred_proba:
        st.write("**Confidence Breakdown**")
        # Order High / Low / Mid consistently for readability
        ordered_labels = ["High Risk", "Mid Risk", "Low Risk"]
        for label in ordered_labels:
            pct = pred_proba.get(label, 0.0)
            color = RISK_STYLE[label]["color"]
            st.markdown(
                f"""
                <div class="conf-row">
                    <span>{RISK_STYLE[label]['emoji']} {label}</span>
                    <span><b>{pct}%</b></span>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{pct}%; background:{color};"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("🔬 View input values sent to the model"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<p class="footer-note">⚕️ This tool is for educational/demo purposes only '
    'and is not a substitute for professional medical advice.</p>',
    unsafe_allow_html=True,
)