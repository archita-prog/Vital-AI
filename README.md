VitalAI: Scalable Healthcare Prediction API

VitalAI is a scalable RESTful API designed for real-time healthcare prediction tasks, supporting Diabetes, Heart Disease, and Blood Sugar risk assessments. Built with FastAPI and robust machine learning pipelines leveraging XGBoost, SVM, and CatBoost, VitalAI delivers sensitive and interpretable predictive analytics for healthcare platforms, prioritizing transparency and seamless digital integration.

Supported Predictions

| Endpoint           | ML Algorithm | Evaluation | Explainability | Feature Engineering                         |
|--------------------|--------------|------------|----------------|--------------------------------------------|
| `/predict/diabetes` | XGBoost      | ROC-AUC    | SHAP           | BMI, pulse ratios, polynomial & interaction terms |
| `/predict/heart`    | SVM          | ROC-AUC    | -              | Clinical & derived features                 |
| `/predict/bloodsugar`| CatBoost    | -          | -              | Advanced ratios, interaction terms          |

Features

- Specialized predictive models for diabetes, heart disease, and blood sugar risk assessment.
- Advanced feature engineering including BMI, pulse pressure ratios, clinical feature interactions, and polynomial terms, improving model effectiveness.
- Comprehensive Exploratory Data Analysis (EDA) and correlation heatmaps to ensure selection of medically relevant features.
- Model evaluation using ROC-AUC metrics achieving up to 0.88 for classification tasks.
- Interpretability provided via SHAP values for diabetes prediction, enabling clinician trust and transparency.
- Modular preprocessing pipelines (imputation, scaling, encoding) serialized with joblib for reproducible and scalable deployments.
- Real-time predictions served through a REST API built with FastAPI, facilitating easy integration with enterprise digital health platforms.
- JSON-based input/output API design for seamless digital workflows.

Technical Workflow

- Data analysis and visualization through EDA to identify key predictive features.
- Model training and hyperparameter tuning on historical healthcare datasets.
- Utilization of XGBoost for Diabetes prediction with LIME and SHAP explainability.
- Implementation of SVM classifier for Heart Disease risk prediction.
- Deployment of CatBoost regressor/classifier for Blood Sugar assessments.
- Construction of modular ML pipelines ensuring consistent preprocessing from training to inference.
- Deployment as scalable FastAPI endpoints enabling enterprise-grade, real-time access to predictions.

Achievements

- Engineered end-to-end predictive models achieving ROC-AUC scores up to 0.88.
- Increased F1 scores by 15-20% over baseline through advanced feature engineering and exploratory analysis.
- Developed scalable, reusable ML pipelines serialized via joblib, enhancing development efficiency.
- Successfully launched models as production-ready RESTful APIs for integration into digital healthcare ecosystems.

  Working
  <img width="1873" height="603" alt="image" src="https://github.com/user-attachments/assets/0611be8e-d790-4515-9cc1-abc2bd59948a" />
  <img width="1824" height="676" alt="image" src="https://github.com/user-attachments/assets/d0c88e3f-666b-414b-b391-9f2e3895d013" />
  <img width="1848" height="786" alt="image" src="https://github.com/user-attachments/assets/68a153b4-60e0-42c2-9c2f-dbf72c15a19d" />
  <img width="1876" height="817" alt="image" src="https://github.com/user-attachments/assets/821c8ed3-8900-4533-b319-24e30d6b0cf7" />
  <img width="1874" height="570" alt="image" src="https://github.com/user-attachments/assets/6782f24b-474e-4e07-99e9-ed2868ae1939" />




