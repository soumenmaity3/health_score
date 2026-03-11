import streamlit as st
import pandas as pd
import pickle
import os
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
REPO_ID = "sm89/health_score"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Cardiovascular AI Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING ---
st.markdown("""
<style>
    /* Remove hardcoded background for .main to support system themes */
    .stApp { background-color: transparent; }
    
    /* Improve metric card visibility */
    [data-testid="stMetricValue"] { color: #2c3e50 !important; }
    
    .risk-box { 
        padding: 20px; 
        border-radius: 15px; 
        margin: 10px 0; 
        font-weight: bold; 
        text-align: center; 
        font-size: 1.2rem;
    }
    
    .risk-high { 
        background-color: #ffeae1; 
        color: #e74c3c; 
        border: 2px solid #e74c3c; 
    }
    
    .risk-low { 
        background-color: #e8f8f5; 
        color: #27ae60; 
        border: 2px solid #27ae60; 
    }

    /* Target headers and text for better contrast if needed */
    h1, h2, h3 { color: inherit !important; }
</style>
""", unsafe_allow_html=True)

# --- GLOBAL MODEL STORAGE ---
@st.cache_resource
def load_resources():
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pkl")
        scaler_path = hf_hub_download(repo_id=REPO_ID, filename="scaler.pkl")
        with open(model_path, 'rb') as f:
            m = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            s = pickle.load(f)
        return m, s
    except Exception as e:
        st.error(f"Failed to load AI models: {e}")
        return None, None

model, scaler = load_resources()

# --- PREPROCESSING LOGIC ---
def predict_risk(data_dict):
    """Core prediction logic shared by Streamlit and Flask."""
    if not model or not scaler:
        return None, "Model not loaded"
    
    # Mapping
    gender_map = 1 if str(data_dict.get('gender')).lower() in ['male', '1'] else 0
    chol_map = int(data_dict.get('cholesterol', 1))
    gluc_map = int(data_dict.get('gluc', 1))
    
    # Feature calculation
    height = float(data_dict['height'])
    weight = float(data_dict['weight'])
    bmi = weight / ((height / 100) ** 2)
    pp = int(data_dict['systolic_bp']) - int(data_dict['diastolic_bp'])
    
    df = pd.DataFrame({
        'age': [int(data_dict['age'])],
        'gender': [gender_map],
        'height': [height],
        'weight': [weight],
        'systolic_bp': [int(data_dict['systolic_bp'])],
        'diastolic_bp': [int(data_dict['diastolic_bp'])],
        'cholesterol': [chol_map],
        'gluc': [gluc_map],
        'smoke': [1 if data_dict.get('smoke') in [True, 'on', '1', 1] else 0],
        'alco': [1 if data_dict.get('alco') in [True, 'on', '1', 1] else 0],
        'active': [1 if data_dict.get('active', True) in [True, 'on', '1', 1] else 0],
        'bmi': [bmi],
        'pulse_pressure': [pp]
    })
    
    cols = ['age', 'gender', 'height', 'weight', 'systolic_bp', 'diastolic_bp', 
            'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'pulse_pressure']
    df = df[cols]
    
    scl_cols = ['age', 'height', 'weight', 'systolic_bp', 'diastolic_bp', 'bmi', 'pulse_pressure']
    df[scl_cols] = scaler.transform(df[scl_cols])
    
    prob = model.predict_proba(df)[0][1]
    pred = int(model.predict(df)[0])
    return {'prediction': pred, 'probability': float(prob), 'bmi': round(bmi, 1), 'pp': pp}, None

# --- FLASK API SERVER ---
server = Flask(__name__)
CORS(server)

@server.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    result, error = predict_risk(data)
    if error:
        return jsonify({'error': error}), 500
    return jsonify(result)

def run_flask():
    # Run Flask on a separate port so it doesn't clash with Streamlit
    server.run(port=5001, debug=False, use_reloader=False)

# Start Flask in a background thread
if 'flask_started' not in st.session_state:
    thread = threading.Thread(target=run_flask, daemon=True)
    thread.start()
    st.session_state['flask_started'] = True

# --- STREAMLIT UI ---
st.title("❤️ Advanced Cardiovascular Health AI")
st.markdown("---")

with st.sidebar:
    st.header("🔗 API Integration")
    st.success("API is active at:")
    st.code("http://localhost:5001/api/predict")
    st.info("Send a JSON POST request with patient data to integrate this model into your other apps.")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📊 Patient Bio-Metrics")
    with st.container(border=True):
        age = st.slider("Age", 1, 100, 45)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        height = st.number_input("Height (cm)", 100, 250, 170)
        weight = st.number_input("Weight (kg)", 30, 250, 75)
        sbp = st.number_input("Systolic BP", 80, 220, 120)
        dbp = st.number_input("Diastolic BP", 50, 150, 80)
        
    st.subheader("🧬 Clinical & Lifestyle")
    with st.container(border=True):
        chol = st.select_slider("Cholesterol", options=[1, 2, 3], value=1, help="1: Normal, 2: Above, 3: High")
        gluc = st.select_slider("Glucose", options=[1, 2, 3], value=1)
        c1, c2, c3 = st.columns(3)
        smoke = c1.checkbox("Smoker")
        alco = c2.checkbox("Alcohol")
        active = c3.checkbox("Active", value=True)

with col2:
    st.subheader("🔍 Real-time Analysis")
    
    input_data = {
        'age': age, 'gender': gender, 'height': height, 'weight': weight,
        'systolic_bp': sbp, 'diastolic_bp': dbp, 'cholesterol': chol, 'gluc': gluc,
        'smoke': smoke, 'alco': alco, 'active': active
    }
    
    result, err = predict_risk(input_data)
    
    if result:
        prob = result['probability']
        
        # Big Metric Display
        st.markdown(f"### Score: {prob*100:.1f}% Risk")
        st.progress(prob)
        
        if result['prediction'] == 1:
            st.markdown("<div class='risk-box risk-high'>⚠️ HIGH CARDIOVASCULAR RISK IDENTIFIED</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='risk-box risk-low'>✅ LOW CARDIOVASCULAR RISK</div>", unsafe_allow_html=True)
            
        m1, m2 = st.columns(2)
        m1.metric("Body Mass Index", f"{result['bmi']}", delta="Normal" if 18.5 < result['bmi'] < 25 else "Check Weight")
        m2.metric("Pulse Pressure", f"{result['pp']} mmHg", delta_color="inverse")
        
        with st.expander("Technical Data Details (JSON)"):
            st.json(result)
            
st.markdown("---")
st.caption("Disclaimer: This AI tool is for research guidance and does not replace professional medical diagnosis.")
    