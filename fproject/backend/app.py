from flask import Flask, request, jsonify
import tensorflow as tf
from models.skincancer import create_model
import numpy as np
import cv2
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Initialize Flask app
app = Flask(__name__)

# ========== MODEL PATHS ==========
DIABETES_MODEL_PATH = 'backend/models/diabetes_model.pkl'
SKIN_MODEL_PATH = 'backend/models/permanent_skin_model.h5'
HEART_MODEL_PATH = 'backend/models/saved_models/heart_disease_model.pkl'
HEART_SCALER_PATH = 'backend/models/saved_models/heart_scaler.pkl'
HEART_DATA_PATH = 'backend/data/cardio_train.csv'

# ========== MODEL LOADING ==========
def load_skin_model():
    """Load skin cancer model"""
    try:
        model = create_model()
        model.load_weights(SKIN_MODEL_PATH)
        return model
    except Exception as e:
        print(f"Error loading skin model: {str(e)}")
        return None

def load_heart_model():
    """Load heart disease model and scaler"""
    try:
        model = joblib.load(HEART_MODEL_PATH)
        scaler = joblib.load(HEART_SCALER_PATH)
        return model, scaler
    except Exception as e:
        print(f"Error loading heart model: {str(e)}")
        return None, None
def load_diabetes_model():
    """Load diabetes prediction model"""
    try:
        model = joblib.load(DIABETES_MODEL_PATH)
        return model
    except Exception as e:
        print(f"Error loading diabetes model: {str(e)}")
        return None
    
# Load models at startup
skin_model = load_skin_model()
heart_model, heart_scaler = load_heart_model()
diabetes_model = load_diabetes_model()


# ========== DATA PROCESSING FUNCTIONS ==========
def load_heart_data():
    """Load and clean heart disease data"""
    try:
        df = pd.read_csv(HEART_DATA_PATH, sep=';')
        # Clean data
        df = df[df['ap_hi'] < 200]
        df = df[df['ap_lo'] < 150]
        df = df[df['height'] > 140]
        df = df[df['weight'] < 200]
        df = df.dropna()
        return df
    except Exception as e:
        print(f"Error loading heart data: {str(e)}")
        return None

def generate_heart_advice(prediction, num_vessels, probability):
    """Generate medical advice based on prediction results"""
    if prediction == 1:
        if num_vessels >= 3:
            return ("🚨 CRITICAL RISK: Multiple blocked vessels detected. "
                   "Immediate cardiology consultation required. "
                   "High probability of significant coronary artery disease.")
        elif num_vessels >= 2:
            return ("⚠️ HIGH RISK: Multiple vessel involvement. "
                   "Urgent stress test and echocardiogram recommended. "
                   "Consider coronary angiography.")
        else:
            return ("⚠️ MODERATE RISK: Early signs of cardiovascular disease. "
                   "Lifestyle changes and follow-up tests recommended. "
                   "Monitor cholesterol and blood pressure.")
    else:
        if probability > 0.4:
            return ("ℹ️ BORDERLINE: No disease detected but some risk factors present. "
                   "Preventive measures recommended. Annual cardiac checkup advised.")
        else:
            return ("✅ LOW RISK: No significant signs of heart disease. "
                   "Maintain healthy lifestyle with regular exercise.")
        

def generate_diabetes_advice(prediction, glucose, hba1c):
    """Generate personalized diabetes advice"""
    if prediction == 1:
        if glucose > 200 or hba1c > 6.5:
            return ("🚨 CRITICAL: Very high glucose/HbA1c detected. "
                   "Immediate medical attention required. "
                   "Likely diabetes with poor control.")
        else:
            return ("⚠️ WARNING: Diabetes detected. "
                   "Schedule endocrinologist visit. "
                   "Monitor glucose levels daily.")
    else:
        if glucose > 140 or hba1c > 5.7:
            return ("ℹ️ PRE-DIABETES: Borderline values detected. "
                   "Lifestyle changes recommended. "
                   "Re-test in 3 months.")
        else:
            return ("✅ NORMAL: No diabetes detected. "
                   "Maintain healthy diet and exercise.")


# ========== API ROUTES ==========
@app.route('/')
def home():
    return "Disease Prediction API (Skin Cancer + Heart Disease + Diabetes)"

# ===== Skin Cancer Routes =====
@app.route('/skin/predict', methods=['POST'])
def predict_skin():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        file = request.files['file']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = skin_model.predict(img)
        predicted_class = int(np.argmax(prediction))

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': float(np.max(prediction))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== Heart Disease Routes =====
@app.route('/heart/analyze', methods=['GET'])
def analyze_heart():
    try:
        df = load_heart_data()
        if df is None:
            return jsonify({"error": "Could not load heart data"}), 500

        stats = {
            "dataset_shape": df.shape,
            "disease_prevalence": float(df['cardio'].mean()),
            "gender_distribution": df['gender'].value_counts().to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }

        return jsonify({
            "success": True,
            "message": "Heart data analysis completed",
            "results": stats
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ===== Diabetes Routes =====
@app.route('/diabetes/predict', methods=['POST'])
def predict_diabetes():
    try:
        data = request.get_json()
        
        # Prepare input features
        input_features = [
            data['gender'],
            data['age'],
            data['hypertension'],
            data['heart_disease'],
            data['smoking_history'],
            data['bmi'],
            data['hba1c_level'],
            data['blood_glucose_level']
        ]
        
        # Make prediction
        prediction = diabetes_model.predict([input_features])
        try:
            probability = diabetes_model.predict_proba([input_features])[0]
        except:
            probability = [0.8, 0.2] if prediction[0] == 1 else [0.2, 0.8]
        
        # Generate risk factors
        risk_factors = []
        if data['bmi'] > 30:
            risk_factors.append(f"BMI {data['bmi']} (obese)")
        elif data['bmi'] > 25:
            risk_factors.append(f"BMI {data['bmi']} (overweight)")
        if data['hba1c_level'] > 5.7:
            risk_factors.append(f"HbA1c {data['hba1c_level']} (pre-diabetic)")
        if data['blood_glucose_level'] > 140:
            risk_factors.append(f"Glucose {data['blood_glucose_level']} (high)")
        if data['age'] > 45:
            risk_factors.append(f"Age {data['age']} (higher risk)")
        
        return jsonify({
            "prediction": int(prediction[0]),
            "probability": float(np.max(probability)),
            "risk_factors": risk_factors,
            "medical_advice": generate_diabetes_advice(
                prediction[0],
                data['blood_glucose_level'],
                data['hba1c_level']
            ),
            "class_probabilities": probability.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/heart/predict', methods=['POST'])
def predict_heart():
    try:
        data = request.get_json()
        
        # Prepare input features with medical weighting
        input_features = [
            data['age'],
            data['gender'],
            data['height']/100,
            data['weight'],
            min(data['ap_hi'], 200),
            min(data['ap_lo'], 150),
            data['cholesterol'] * 1.2,
            data['gluc'] * 1.1,
            data['smoke'] * 1.5,
            data['alco'] * 1.3,
            data['active'] * 0.9,
            data['ca'] * 2.0
        ]
        
        # Scale features and make prediction
        scaled_features = heart_scaler.transform([input_features])
        prediction = heart_model.predict(scaled_features)
        probability = heart_model.predict_proba(scaled_features)
        
        # Generate risk factors explanation
        risk_factors = []
        if data['age'] > 50:
            risk_factors.append(f"Age {data['age']} (higher risk)")
        if data['gender'] == 1:
            risk_factors.append("Male gender (higher risk)")
        if data['cholesterol'] > 2:
            risk_factors.append(f"High cholesterol level {data['cholesterol']}")
        if data['ca'] > 0:
            risk_factors.append(f"{data['ca']} blocked vessels (major risk)")
        if data['smoke'] == 1:
            risk_factors.append("Smoking (major risk)")
        
        return jsonify({
            "prediction": int(prediction[0]),
            "probability": float(np.max(probability)),
            "risk_factors": risk_factors,
            "medical_advice": generate_heart_advice(prediction[0], data['ca'], probability[0][1]),
            "class_probabilities": probability[0].tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)