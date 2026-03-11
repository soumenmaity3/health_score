# ❤️ Cardiovascular Health Risk Predictor 

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/sm89/health_score)

A full-stack Machine Learning application designed to predict the risk of cardiovascular diseases based on patient bio-metrics, clinical measurements, and lifestyle factors. 

This project provides multiple ways to interact with the trained AI model:
1. **Streamlit UI**: A fast, data-centric web dashboard.
2. **Flask API**: A backend server tailored for integration with external applications (e.g., Android, React, iOS).
3. **Dual-Service Mode**: Run both the web dashboard and API concurrently.

---

## 🎯 Features
- **High-Accuracy ML Model**: Built using Scikit-Learn (XGBoost/Random Forest).
- **Automated Cloud Sync**: Fetches the latest `.pkl` model directly from the Hugging Face Hub on startup.
- **RESTful API**: Accepts JSON payloads and returns structured prediction formatting.
- **Premium UI**: The Flask templates feature a glassmorphism design with responsive CSS, and the Streamlit app automatically adjusts to Light/Dark system themes.

---

## 🚀 Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/soumenmaity3/health_score.git
   cd health_score
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dual-Service App (Streamlit + Background API):**
   ```bash
   streamlit run app_hf.py
   ```
   *(UI opens on port `8501`, API listens on `http://localhost:5001/api/predict`)*

4. **Run the Standalone Flask Web App:**
   ```bash
   python flask_app.py
   ```
   *(Opens on `http://127.0.0.1:5000`)*

---

## 📡 API Reference

When the Flask server is running, external apps can make POST requests to get health score predictions.

**Endpoint:** `/api/predict` (for `app_hf.py`) or `/predict` (for `flask_app.py`)  
**Method:** `POST`  
**Payload Example (JSON):**

```json
{
    "age": 45,
    "gender": "Male",
    "height": 170,
    "weight": 75,
    "systolic_bp": 140,
    "diastolic_bp": 90,
    "cholesterol": 2,
    "gluc": 1,
    "smoke": true,
    "alco": false,
    "active": true
}
```

**Response Example:**
```json
{
  "bmi": 26.0,
  "pp": 50,
  "prediction": 1,
  "probability": 0.84,
  "status": "success"
}
```

---

## ☁️ Deployment Guide: Hosting the API on Render.com

Hugging Face Spaces is excellent for graphical interfaces but restricts external API connections. If you want your Android, iOS, or Web app to securely connect to this AI model over the internet, deploy the Flask API to **Render.com** (Free Tier).

### Step-by-Step Render Deployment:

1. **Commit and Push to GitHub**
   Ensure all your latest files (especially `flask_app.py` and `requirements.txt`) are pushed to your GitHub repository.
   - *Note: `gunicorn` must be in your `requirements.txt` (which is already included!)*

2. **Create a Render Account**
   Sign up at [Render.com](https://render.com/) and connect your GitHub account.

3. **Create a New Web Service**
   - Click **"New +"** and select **"Web Service"**.
   - Choose **"Build and deploy from a Git repository"**.
   - Select your repository: `soumenmaity3/health_score`.

4. **Configure the Service**
   Fill in the form with the following EXACT settings:
   - **Name**: `healthscore-api` (or any name you prefer)
   - **Language**: `Python 3`
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn flask_app:app`

5. **Deploy!**
   - Scroll down and click **Create Web Service**.
   - Wait ~3 to 5 minutes for Render to download the libraries and start your server.

6. **Connect Your External App!**
   Render will provide you with a live URL (e.g., `https://healthscore-api.onrender.com`). 
   Update your mobile/external app to send its POST request data to:
   `https://healthscore-api.onrender.com/predict`

---
## ⚠️ Disclaimer
**This tool uses a machine learning model for informational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment.**
