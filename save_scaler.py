import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Set paths
data_path = r"C:\Users\sm803\OneDrive\Desktop\ALL ML PROJECTS\MedicalAi\HealthScore\Data\cardio_train.csv"
model_dir = r"C:\Users\sm803\OneDrive\Desktop\ALL ML PROJECTS\MedicalAi\HealthScore\NoteBook"

# Load data
df = pd.read_csv(data_path, sep=";")

# Basic preprocessing as per notebook
df.age = (df.age / 365).astype(int)
df.drop('id', axis=1, inplace=True)
df = df[df["ap_lo"] >= 40]
df.gender = df.gender.replace({1: 0, 2: 1})
df = df.rename(columns={"ap_lo": "diastolic_bp", "ap_hi": "systolic_bp"})
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
df["pulse_pressure"] = df.systolic_bp - df.diastolic_bp

# Define numeric columns for scaling
scl_col = ['age', 'height', 'weight', 'systolic_bp', 'diastolic_bp', 'bmi', 'pulse_pressure']

# Fit scaler
scaler = StandardScaler()
scaler.fit(df[scl_col])

# Save scaler
scaler_path = os.path.join(model_dir, 'scaler.joblib')
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")
