# train_and_save.py
import os
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

iris = load_iris()
X, y = iris.data, iris.target
model = LogisticRegression(max_iter=200)
model.fit(X, y)

joblib.dump(model, os.path.join(MODEL_DIR, "diabetes_model.pkl"))
joblib.dump(model, os.path.join(MODEL_DIR, "heart_model.pkl"))
joblib.dump(model, os.path.join(MODEL_DIR, "blood_model.pkl"))

print(f"✅ Models saved in {MODEL_DIR}")
