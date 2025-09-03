# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# ------------------------
# Load models
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

blood_model = joblib.load(os.path.join(BASE_DIR, "blood_model.pkl"))
diabetes_model = joblib.load(os.path.join(BASE_DIR, "diabetes_model.pkl"))
heart_model = joblib.load(os.path.join(BASE_DIR, "heart_model.pkl"))

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(title="VitalAI - Healthcare Prediction API")

# ------------------------
# Input Schemas
# ------------------------
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

class HeartInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

class BloodSugarInput(BaseModel):
    PPG_Signal: float
    Heart_Rate: float
    Systolic_Peak: float
    Diastolic_Peak: float
    Pulse_Area: float
    Gender: int
    Age: int
    Height: float
    Weight: float
    BMI: float
    pulse_pressure: float
    systolic_diastolic_ratio: float
    hr_pulsearea: float

# ------------------------
# Endpoints
# ------------------------
@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    df = pd.DataFrame([data.dict()])
    pred = diabetes_model.predict(df)[0]
    return {"diabetes_prediction": int(pred)}

@app.post("/predict/heart")
def predict_heart(data: HeartInput):
    df = pd.DataFrame([data.dict()])
    pred = heart_model.predict(df)[0]
    return {"heart_disease_prediction": int(pred)}

@app.post("/predict/bloodsugar")
def predict_blood_sugar(data: BloodSugarInput):
    df = pd.DataFrame([data.dict()])
    pred = blood_model.predict(df)[0]
    return {"blood_sugar_prediction": float(pred)}
