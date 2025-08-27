from fastapi import FastAPI
import joblib
import pandas as pd

# Initialize FastAPI
app = FastAPI(title="Healthcare Prediction API")

# Load models
blood_model = joblib.load("blood_model.pkl")
diabetes_model = joblib.load("diabetes_model.pkl")
heart_model = joblib.load("heart_model.pkl")

# Endpoints
@app.get("/")
def home():
    return {"message": "Healthcare Prediction API is running!"}

@app.post("/predict/blood_sugar")
def predict_blood_sugar(data: dict):
    df = pd.DataFrame([data])
    prediction = blood_model.predict(df)[0]
    return {"blood_sugar_prediction": float(prediction)}

@app.post("/predict/diabetes")
def predict_diabetes(data: dict):
    df = pd.DataFrame([data])
    prediction = diabetes_model.predict(df)[0]
    return {"diabetes_prediction": int(prediction)}

@app.post("/predict/heart")
def predict_heart(data: dict):
    df = pd.DataFrame([data])
    prediction = heart_model.predict(df)[0]
    return {"heart_disease_prediction": int(prediction)}
