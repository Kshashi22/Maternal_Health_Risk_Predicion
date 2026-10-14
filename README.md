# Maternal_Health_Risk_Predicion

Predicts a pregnant patient's health risk level (**Low / Mid / High**) from vital signs, using a Random Forest classifier, with a Streamlit web app for live predictions.

## Project Structure

```
maternal-health-risk-prediction/
│
├── data/
│   ├── training_dataset.csv     # 2,100 rows × 12 columns, used to train the model
│   └── testing_dataset.csv      # 525 rows, held-out data used to evaluate the model
│
├── model/
│   ├── maternal_risk_model.pkl  # trained Random Forest classifier
│   └── scaler.pkl               # fitted StandardScaler (must be used for any new input)
│
├── predictions.csv              # model's predictions on testing_dataset.csv
├── app.py                       # Streamlit UI for live single-patient predictions
├── requirements.txt
└── README.md
```

## Dataset

- **Columns**: Age, SystolicBP, DiastolicBP, BS (Blood Sugar), BodyTemp, HeartRate, BMI, PreviousComplications, GestationalAgeWeeks, HemoglobinLevel, PhysicalActivityLevel, RiskLevel
- **Target**: `RiskLevel` — Low Risk / Mid Risk / High Risk
- Part of the training data is based on real records; the remainder is synthetically generated to expand dataset size, following the statistical distributions of the original data. Not real clinical records — for prototyping/coursework use only.

## Model

- **Algorithm**: Random Forest Classifier (`n_estimators=200`, `max_depth=10`)
- **Feature engineering**: `PulsePressure = SystolicBP - DiastolicBP`, `MAP = (SystolicBP + 2×DiastolicBP) / 3`
- **Preprocessing**: `PhysicalActivityLevel` ordinally encoded (Low=0, Moderate=1, High=2), `RiskLevel` label-encoded (High=0, Low=1, Mid=2), all features scaled with `StandardScaler`
- **Test accuracy**: ~97.5% on the held-out `testing_dataset.csv`

## predictions.csv

Contains every row from `testing_dataset.csv` plus:
- `PredictedRiskLevel` — the model's prediction
- `Confidence_high risk`, `Confidence_low risk`, `Confidence_mid risk` — predicted probability (%) per class
- `Correct` — whether the prediction matched the actual label

## Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

Make sure `app.py` stays in the same folder as `model/maternal_risk_model.pkl` and `model/scaler.pkl` (or update the paths inside `app.py`).

## Disclaimer

This project is for educational/demo purposes only and is not a substitute for professional medical advice.
