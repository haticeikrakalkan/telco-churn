from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.features import clean_data

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"

app = FastAPI(title="Telco Churn API")
model = joblib.load(MODEL_PATH)


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])
    df = clean_data(df)
    proba = float(model.predict_proba(df)[0, 1])
    return {"churn_probability": round(proba, 4), "churn": proba >= 0.5}