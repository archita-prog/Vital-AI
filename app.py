from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# ------------------------
# 1️⃣ Load Models
# ------------------------
blood_model = joblib.load("blood_model.pkl")
diabetes_model = joblib.load("diabetes_model.pkl")
heart_model = joblib.load("heart_model.pkl")

# If you also saved preprocessors, load them too (optional)
# blood_preprocessor = joblib.load("blood_preprocessor.pkl")
# diabetes_preprocessor = joblib.load("diabetes_preprocessor.pkl")
# heart_preprocessor = joblib.load("heart_preprocessor.pkl")

# ------------------------
# 2️⃣ Create FastAPI app
# ------------------------
app = FastAPI(title="VitalAI - Healthcare Prediction API")

# ------------------------
# 3️⃣ Define Input Schemas
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
    Gender: int   # 0/1 encoding (you can map M/F)
    Age: int
    Height: float
    Weight: float
    BMI: float
    pulse_pressure: float
    systolic_diastolic_ratio: float
    hr_pulsearea: float

# ------------------------
# 4️⃣ Define Endpoints
# ------------------------
@app.get("/")
def home():
    return {"message": "Welcome to VitalAI Healthcare Prediction API 🚀"}

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
