"""
Maternal Health Risk Prediction — Streamlit App
------------------------------------------------
Loads the model + scaler you trained/saved in your notebook
(maternal_risk_model.pkl, scaler.pkl) and lets a user enter
patient values through a form to get a live risk prediction.

Run with:
    streamlit run app.py

Make sure maternal_risk_model.pkl and scaler.pkl are in the
SAME folder as this app.py (or update the paths below).
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
    layout="centered"
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
        "Place both files in the same folder as app.py and restart."
    )
    st.stop()

# ==============================================================
# CONSTANTS — must match your notebook's encoding EXACTLY
# ==============================================================
ACTIVITY_MAP = {"Low": 0, "Moderate": 1, "High": 2}

# LabelEncoder sorted these alphabetically during training:
# high risk = 0, low risk = 1, mid risk = 2
RISK_LABELS = {0: "High Risk", 1: "Low Risk", 2: "Mid Risk"}

RISK_COLOR = {
    "High Risk": "#e63946",
    "Mid Risk": "#f4a261",
    "Low Risk": "#2a9d8f"
}

# Exact column order the scaler was fit on
FEATURE_ORDER = [
    "Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate",
    "BMI", "PreviousComplications", "GestationalAgeWeeks",
    "HemoglobinLevel", "PhysicalActivityLevel", "PulsePressure", "MAP"
]

# ==============================================================
# HEADER
# ==============================================================
st.title("🤰 Maternal Health Risk Predictor")
st.markdown(
    "Enter the patient's vitals below to predict their pregnancy "
    "risk level using a trained Random Forest model."
)
st.divider()

# ==============================================================
# INPUT FORM
# ==============================================================
with st.form("risk_form"):
    st.subheader("Patient Vitals")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=10, max_value=70, value=28)
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=160, value=115)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=110, value=75)
        bs = st.number_input("Blood Sugar (mmol/L)", min_value=6.0, max_value=19.0, value=7.5, step=0.1)
        body_temp = st.number_input("Body Temperature (°F)", min_value=96.0, max_value=103.0, value=98.0, step=0.1)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=60, max_value=110, value=76)

    with col2:
        bmi = st.number_input("BMI", min_value=15.0, max_value=45.0, value=24.0, step=0.1)
        prev_complications = st.selectbox("Previous Complications?", ["No", "Yes"])
        gest_age = st.number_input("Gestational Age (weeks)", min_value=4, max_value=42, value=28)
        hemoglobin = st.number_input("Hemoglobin Level (g/dL)", min_value=6.0, max_value=16.0, value=11.5, step=0.1)
        activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])

    submitted = st.form_submit_button("Predict Risk Level", use_container_width=True)

# ==============================================================
# PREDICTION LOGIC
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
    st.divider()
    st.subheader("Prediction Result")

    color = RISK_COLOR[pred_label]
    st.markdown(
        f"""
        <div style="
            background-color:{color}20;
            border-left: 6px solid {color};
            padding: 20px;
            border-radius: 8px;
        ">
            <h3 style="color:{color}; margin:0;">Predicted Risk Level: {pred_label}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    if pred_proba:
        st.markdown("#### Confidence Breakdown")
        proba_df = pd.DataFrame(
            {"Risk Level": list(pred_proba.keys()), "Probability (%)": list(pred_proba.values())}
        ).sort_values("Probability (%)", ascending=False)
        st.bar_chart(proba_df.set_index("Risk Level"))

    with st.expander("View input values sent to the model"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))

    if pred_label == "High Risk":
        st.warning("⚠️ This prediction suggests elevated risk. Please consult a healthcare provider.")
    elif pred_label == "Mid Risk":
        st.info("ℹ️ Moderate risk detected. Regular monitoring is recommended.")
    else:
        st.success("✅ Low risk predicted based on current vitals.")

st.divider()
st.caption(
    "⚕️ This tool is for educational/demo purposes only and is not a substitute "
    "for professional medical advice."
)
