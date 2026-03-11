import pandas as pd
import pickle
from huggingface_hub import hf_hub_download

# Define constants
REPO_ID = "sm89/health_score"

def load_resources_from_hf():
    """
    Downloads and loads the model and scaler from Hugging Face Hub.
    """
    print(f"Fetching resources from {REPO_ID}...")
    try:
        # Download files from Hugging Face Hub
        model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pkl")
        scaler_path = hf_hub_download(repo_id=REPO_ID, filename="scaler.pkl")

        # Load into memory
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        
        print("Resources loaded successfully!")
        return model, scaler
    except Exception as e:
        print(f"Error loading resources: {e}")
        return None, None

def run_sample_prediction():
    model, scaler = load_resources_from_hf()
    
    if model and scaler:
        # Define a sample input based on the expected schema
        # Features: age, gender, height, weight, systolic_bp, diastolic_bp, 
        #           cholesterol, gluc, smoke, alco, active, bmi, pulse_pressure
        sample_input = {
            'age': 45,
            'gender': 1,  # 1: Male, 0: Female
            'height': 170,
            'weight': 75.0,
            'systolic_bp': 140, # Slightly high BP
            'diastolic_bp': 90,
            'cholesterol': 2,   # Above Normal
            'gluc': 1,          # Normal
            'smoke': 1,         # Smoker
            'alco': 1,
            'active': 0,        # Inactive
            'bmi': 25.95,       # Calculated: 75 / (1.7^2)
            'pulse_pressure': 50 # 140 - 90
        }
        
        df = pd.DataFrame([sample_input])
        
        # Preprocessing: Scale the numerical columns as per training logic
        scl_col = ['age', 'height', 'weight', 'systolic_bp', 'diastolic_bp', 'bmi', 'pulse_pressure']
        df[scl_col] = scaler.transform(df[scl_col])
        
        # Prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        print("\n--- Prediction Results ---")
        print(f"Blood Pressure: {sample_input['systolic_bp']}/{sample_input['diastolic_bp']} mmHg")
        print(f"Risk Probability: {probability:.2%}")
        if prediction == 1:
            print("Status: ⚠️ HIGH RISK IDENTIFIED")
        else:
            print("Status: ✅ LOW RISK")

if __name__ == "__main__":
    run_sample_prediction()
