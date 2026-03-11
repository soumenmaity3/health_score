---
language: en
license: mit
tags:
- health
- medical
- cardiovascular
- risk-prediction
- scikit-learn
- xgboost
- healthcare
- heart-disease
metrics:
- accuracy
- f1
- roc_auc
---

# ❤️ Cardiovascular Health Risk Predictor

This model predicts the risk of cardiovascular issues based on personal and clinical bio-metrics. It was trained on the Cardiovascular Disease dataset and is intended for informational and research purposes.

## 🚀 Model Details
- **Task**: Binary Classification (Risk / No Risk)
- **Framework**: Scikit-Learn / Joblib
- **Features**: 
    - `age`: Age in years
    - `gender`: Gender (mapped)
    - `height`: Height in cm
    - `weight`: Weight in kg
    - `systolic_bp`: Systolic Blood Pressure
    - `diastolic_bp`: Diastolic Blood Pressure
    - `cholesterol`: Cholesterol Level (mapped)
    - `gluc`: Glucose Level (mapped)
    - `smoke`: Smoker status
    - `alco`: Alcohol consumption
    - `active`: Physical activity
    - `bmi`: Body Mass Index (calculated)
    - `pulse_pressure`: Difference between Systolic and Diastolic BP

## 📊 Performance
The model has been optimized for high recall and ROC-AUC to ensure potential risks are not missed. (Detailed metrics available in training logs).

## 🛠️ Usage
You can load the model and scaler using `pickle` or `joblib`:

```python
import pickle
import pandas as pd

# Load resources
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Sample prediction
data = pd.DataFrame({...}) # Match feature order
data[scaler_cols] = scaler.transform(data[scaler_cols])
prediction = model.predict(data)
```

## ⚠️ Disclaimer
**This tool uses a machine learning model for informational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any questions regarding a medical condition.**
