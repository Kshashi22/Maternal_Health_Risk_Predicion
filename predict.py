"""
Maternal Health Risk Predictor — Vercel Python serverless function.
URL:  /api/predict
  GET  -> health check (also reports model-loading errors as JSON)
  POST -> JSON risk prediction
"""
import json
import warnings
from http.server import BaseHTTPRequestHandler
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

ACTIVITY_MAP = {"Low": 0, "Moderate": 1, "High": 2}

# LabelEncoder sorted alphabetically: high risk=0, low risk=1, mid risk=2
RISK_LABELS = {0: "High Risk", 1: "Low Risk", 2: "Mid Risk"}

# Exact column order the scaler was fit on
FEATURE_ORDER = [
    "Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate",
    "BMI", "PreviousComplications", "GestationalAgeWeeks",
    "HemoglobinLevel", "PhysicalActivityLevel", "PulsePressure", "MAP",
]

LIMITS = {
    "Age": (10, 70), "SystolicBP": (70, 160), "DiastolicBP": (40, 110),
    "BS": (6.0, 19.0), "BodyTemp": (96.0, 103.0), "HeartRate": (60, 110),
    "BMI": (15.0, 45.0), "GestationalAgeWeeks": (4, 42),
    "HemoglobinLevel": (6.0, 16.0),
}

_cache = {}


def load_artifacts():
    """Load model + scaler once per warm instance."""
    if "model" not in _cache:
        import joblib  # imported lazily so a failure is reported as JSON, not a crash

        here = Path(__file__).resolve().parent
        for model_dir in (here.parent / "model", Path.cwd() / "model", here / "model"):
            if (model_dir / "maternal_risk_model.pkl").exists():
                _cache["model"] = joblib.load(model_dir / "maternal_risk_model.pkl")
                _cache["scaler"] = joblib.load(model_dir / "scaler.pkl")
                break
        else:
            raise FileNotFoundError("model/maternal_risk_model.pkl was not bundled with the function")
    return _cache["model"], _cache["scaler"]


def run_prediction(data):
    """Returns (status_code, body_dict)."""
    import numpy as np

    row = {}
    for name, (lo, hi) in LIMITS.items():
        try:
            value = float(data[name])
        except (KeyError, TypeError, ValueError):
            return 400, {"error": f"'{name}' is missing or not a number."}
        if not lo <= value <= hi:
            return 400, {"error": f"'{name}' must be between {lo} and {hi}."}
        row[name] = value

    if data.get("PreviousComplications") not in ("Yes", "No"):
        return 400, {"error": "'PreviousComplications' must be 'Yes' or 'No'."}
    if data.get("PhysicalActivityLevel") not in ACTIVITY_MAP:
        return 400, {"error": "'PhysicalActivityLevel' must be Low, Moderate or High."}

    row["PreviousComplications"] = 1 if data["PreviousComplications"] == "Yes" else 0
    row["PhysicalActivityLevel"] = ACTIVITY_MAP[data["PhysicalActivityLevel"]]

    # Feature engineering (same as training)
    row["PulsePressure"] = row["SystolicBP"] - row["DiastolicBP"]
    row["MAP"] = (row["SystolicBP"] + 2 * row["DiastolicBP"]) / 3

    model, scaler = load_artifacts()
    X = np.array([[row[c] for c in FEATURE_ORDER]], dtype=float)
    X_scaled = scaler.transform(X)

    pred = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0]

    return 200, {
        "label": RISK_LABELS[pred],
        "probabilities": {RISK_LABELS[int(c)]: round(float(p) * 100, 1)
                          for c, p in zip(model.classes_, proba)},
        "inputs": {c: round(float(row[c]), 2) for c in FEATURE_ORDER},
    }


class handler(BaseHTTPRequestHandler):
    def _send(self, status, body):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        try:
            load_artifacts()
            self._send(200, {"status": "ok"})
        except Exception as exc:  # surface the real reason instead of a blank 500
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(length) or b"{}")
            status, body = run_prediction(data)
            self._send(status, body)
        except json.JSONDecodeError:
            self._send(400, {"error": "Request body must be valid JSON."})
        except Exception as exc:
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
