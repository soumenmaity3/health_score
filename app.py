from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os
from flask_cors import CORS
from huggingface_hub import hf_hub_download

app = Flask(__name__)
CORS(app)

# Configuration
REPO_ID = "sm89/health_score"

# Global Variables to store model and scaler
model = None
scaler = None

def load_resources():
    global model, scaler
    try:
        print("Fetching models from Hugging Face Hub...")
        # Download files from Hugging Face Hub
        model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pkl")
        scaler_path = hf_hub_download(repo_id=REPO_ID, filename="scaler.pkl")
        
        # Load into memory
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        print("Resources loaded successfully!")
    except Exception as e:
        print(f"Error loading model from Hub: {e}")

# Load model/scaler when the app starts
load_resources()

def preprocess_input(form_data):
    """
    Preprocess raw form data into a DataFrame suitable for the model.
    """
    # Mapping
    gender_mapped = 1 if form_data['gender'] == "Male" else 0
    cholesterol_mapped = int(form_data['cholesterol'])
    gluc_mapped = int(form_data['gluc'])
    
    age = int(form_data['age'])
    height = float(form_data['height'])
    weight = float(form_data['weight'])
    systolic_bp = int(form_data['systolic_bp'])
    diastolic_bp = int(form_data['diastolic_bp'])
    smoke = 1 if form_data.get('smoke') == "on" else 0
    alco = 1 if form_data.get('alco') == "on" else 0
    active = 1 if form_data.get('active') == "on" else 0
    
    # Feature Engineering
    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = systolic_bp - diastolic_bp
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': [age],
        'gender': [gender_mapped],
        'height': [height],
        'weight': [weight],
        'systolic_bp': [systolic_bp],
        'diastolic_bp': [diastolic_bp],
        'cholesterol': [cholesterol_mapped],
        'gluc': [gluc_mapped],
        'smoke': [smoke],
        'alco': [alco],
        'active': [active],
        'bmi': [bmi],
        'pulse_pressure': [pulse_pressure]
    })
    
    # Matching column order used during training
    cols_order = ['age', 'gender', 'height', 'weight', 'systolic_bp', 'diastolic_bp', 
                  'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'pulse_pressure']
    data = data[cols_order]
    
    # Scaling numerical columns
    scl_col = ['age', 'height', 'weight', 'systolic_bp', 'diastolic_bp', 'bmi', 'pulse_pressure']
    data[scl_col] = scaler.transform(data[scl_col])
    
    return data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded. Contact administrator.'}), 500
    
    try:
        # Preprocess input from POST request
        processed_data = preprocess_input(request.form)
        
        # Predict
        prediction = int(model.predict(processed_data)[0])
        probability = float(model.predict_proba(processed_data)[0][1])
        
        return jsonify({
            'prediction': prediction,
            'probability': probability,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
