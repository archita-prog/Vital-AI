VitalAI: Scalable Healthcare Prediction API

VitalAI is a scalable RESTful API for real-time healthcare prediction tasks including Diabetes, Heart Disease, and Blood Sugar risk assessment. Built using FastAPI and robust machine learning pipelines (XGBoost, SVM, CatBoost), it enables sensitive and interpretable predictive analytics for healthcare platforms, with a focus on transparency and easy digital integration.

 Supported Predictions

| Endpoint             | ML Algorithm | Evaluation  | Explainability | Feature Engineering                |
|----------------------|--------------|-------------|---------------|-------------------------------------|
| /predict/diabetes    | XGBoost      | ROC-AUC     | SHAP          | BMI, pulse ratios, polynomial and interaction terms |
| /predict/heart       | SVM          | ROC-AUC     | -             | Clinical & derived features        |
| /predict/bloodsugar  | CatBoost     | -           | -             | Advanced ratios, interaction terms |

 Features

- Multiple specialized models for diabetes, heart disease, and blood sugar prediction.
- Advanced feature engineering for all endpoints:
- BMI, pulse ratios, clinical interactions, and polynomial features.
- Exploratory Data Analysis (EDA) and correlation heatmaps to ensure medically relevant features.
- Model evaluation with ROC-AUC metrics for classification tasks.
- Interpretability SHAP values for diabetes model, enabling transparency for clinicians.
- Modular preprocessing pipelines: Imputation, scaling, and encoding, serialized via joblib for reusable, robust deployment.
- REST API with FastAPI: Real-time, scalable access to all models.
- Easy integration into digital health platforms—input/output via JSON.

Technical Workflow

- EDA: Analyze and visualize data to select features.
- Modeling: Train and tune models on historical healthcare data.
- Diabetes: XGBoost, with SHAP for explanation.
- Heart Disease: SVM classifier.
- Blood Sugar: CatBoost regressor or classifier (as appropriate).
- Preprocessing: Modular pipelines for reproducible feature transformation.
- Deployment: All models served with FastAPI, ready for production.

