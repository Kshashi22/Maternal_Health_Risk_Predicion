"""
Maternal Health Risk Predictor — Vercel serverless API (Flask)
POST /api/predict  -> JSON risk prediction
GET  /api/health   -> simple health check
The static front end lives in public/index.html.
"""
import warnings
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
model = joblib.load(MODEL_DIR / "maternal_risk_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

ACTIVITY_MAP = {"Low": 0, "Moderate": 1, "High": 2}

# LabelEncoder sorted alphabetically: high risk=0, low risk=1, mid risk=2
RISK_LABELS = {0: "High Risk", 1: "Low Risk", 2: "Mid Risk"}

# Exact column order the scaler was fit on
FEATURE_ORDER = [
    "Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate",
    "BMI", "PreviousComplications", "GestationalAgeWeeks",
    "HemoglobinLevel", "PhysicalActivityLevel", "PulsePressure", "MAP",
]

# (min, max) accepted for each numeric input — same limits as the Streamlit form
LIMITS = {
    "Age": (10, 70), "SystolicBP": (70, 160), "DiastolicBP": (40, 110),
    "BS": (6.0, 19.0), "BodyTemp": (96.0, 103.0), "HeartRate": (60, 110),
    "BMI": (15.0, 45.0), "GestationalAgeWeeks": (4, 42),
    "HemoglobinLevel": (6.0, 16.0),
}


@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}

    row = {}
    for name, (lo, hi) in LIMITS.items():
        try:
            value = float(data[name])
        except (KeyError, TypeError, ValueError):
            return jsonify(error=f"'{name}' is missing or not a number."), 400
        if not lo <= value <= hi:
            return jsonify(error=f"'{name}' must be between {lo} and {hi}."), 400
        row[name] = value

    if data.get("PreviousComplications") not in ("Yes", "No"):
        return jsonify(error="'PreviousComplications' must be 'Yes' or 'No'."), 400
    if data.get("PhysicalActivityLevel") not in ACTIVITY_MAP:
        return jsonify(error="'PhysicalActivityLevel' must be Low, Moderate or High."), 400

    row["PreviousComplications"] = 1 if data["PreviousComplications"] == "Yes" else 0
    row["PhysicalActivityLevel"] = ACTIVITY_MAP[data["PhysicalActivityLevel"]]

    # Feature engineering (same as training)
    row["PulsePressure"] = row["SystolicBP"] - row["DiastolicBP"]
    row["MAP"] = (row["SystolicBP"] + 2 * row["DiastolicBP"]) / 3

    X = np.array([[row[c] for c in FEATURE_ORDER]], dtype=float)
    X_scaled = scaler.transform(X)

    pred = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0]

    return jsonify(
        label=RISK_LABELS[pred],
        probabilities={RISK_LABELS[int(c)]: round(float(p) * 100, 1)
                       for c, p in zip(model.classes_, proba)},
        inputs={c: round(float(row[c]), 2) for c in FEATURE_ORDER},
    )
