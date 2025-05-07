import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import matplotlib.pyplot as plt
import cv2
import random
from datetime import datetime
import pandas as pd
import joblib
import os
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report



# Constants
DIABETES_MODEL_PATH = r"C:\Users\lenovo\Desktop\fproject\backend\models\diabetes_model.pkl"
SKIN_MODEL_PATH = r"C:\Users\lenovo\Desktop\fproject\backend\models\permanent_skin_model.h5"
HEART_MODEL_PATH = "../backend/models/saved_models/heart_disease_model.pkl"
HEART_SCALER_PATH = "../backend/models/saved_models/heart_scaler.pkl"
HEART_DATA_PATH = "../backend/models/data/heart.csv"
MIN_CONFIDENCE = 0.85  # 85% minimum confidence for serious lesions
MIN_AREA = 200  # Minimum lesion area in pixels
MAX_AGE_DAYS = 30  # Maximum cache age for model reload
# Constants
PNEUMONIA_MODEL_PATH = r"C:\Users\lenovo\Desktop\fproject\backend\models\pneumonia_classifier_model.h5"  # Add this with other paths

# Configure page (move this to the top)
st.set_page_config(
    page_title="MedScan AI - Comprehensive Health Diagnostics", 
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://medical-ai-support.com',
        'Report a bug': "https://medscan-issues.com",
        'About': "### MedScan AI v4.0\nAdvanced health diagnostics platform"
    }
)



@st.cache_resource(ttl=86400)
def load_pneumonia_model():
    """Load and cache the pneumonia classification model"""
    try:
        model = tf.keras.models.load_model(PNEUMONIA_MODEL_PATH)
        st.success("✅ Pneumonia Detection Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Pneumonia Model loading failed: {str(e)}")
        return None







# Load models with caching
@st.cache_resource(ttl=86400)  # Cache for 1 day
def load_diabetes_model():
    """Load and cache the diabetes prediction model"""
    try:
        model = joblib.load(DIABETES_MODEL_PATH)
        st.success("✅ Diabetes Prediction Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Diabetes Model loading failed: {str(e)}")
        return None
    
@st.cache_resource(ttl=86400)  # Cache for 1 day
def load_skin_model():
    """Load and cache the TensorFlow skin model with custom objects"""
    try:
        model = tf.keras.models.load_model(
            SKIN_MODEL_PATH,
            custom_objects={
                'dice_coefficient': dice_coefficient,
                'binary_crossentropy': tf.keras.losses.binary_crossentropy
            }
        )
        st.success("✅ Skin Cancer Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Skin Model loading failed: {str(e)}")
        return None

@st.cache_data
def load_heart_model():
    """Load heart disease model and scaler"""
    try:
        model = joblib.load(HEART_MODEL_PATH)
        scaler = joblib.load(HEART_SCALER_PATH)
        st.success("✅ Heart Disease Model loaded successfully")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Heart Model loading failed: {str(e)}")
        return None, None

@st.cache_data
def load_heart_data():
    """Load heart disease dataset"""
    try:
        return pd.read_csv(HEART_DATA_PATH)
    except Exception as e:
        st.error(f"❌ Heart Data loading failed: {str(e)}")
        return None

def dice_coefficient(y_true, y_pred, smooth=1):
    """Dice coefficient metric for model evaluation"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

class LesionAnalyzer:
    """Advanced lesion analysis with medical-grade criteria"""
    
    def __init__(self, image):
        self.original_img = np.array(image)
        self.image_size = image.size
        self.results = {
            'mask': None,
            'prob': 0.0,
            'is_cancer': False,
            'lesions': [],
            'advice': "",
            'confidence': 0.0
        }
        
    def preprocess_image(self):
        """Prepare image for model input"""
        img_array = cv2.resize(self.original_img, (256, 256)) / 255.0
        return np.expand_dims(img_array, axis=0)
    
    def is_serious_lesion(self, contour, area):
        """Medical-grade criteria for serious lesions"""
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        convexity = cv2.matchShapes(contour, cv2.convexHull(contour), 1, 0.0)
        
        # Advanced shape analysis
        aspect_ratio = self._get_aspect_ratio(contour)
        solidity = self._get_solidity(contour)
        
        return all([
            area >= MIN_AREA,
            circularity < 0.7,
            convexity > 0.15,
            aspect_ratio > 1.5,
            solidity < 0.9
        ])
    
    def _get_aspect_ratio(self, contour):
        """Calculate aspect ratio of contour bounding box"""
        _, _, w, h = cv2.boundingRect(contour)
        return max(w, h) / min(w, h) if min(w, h) > 0 else 0
    
    def _get_solidity(self, contour):
        """Calculate solidity (area/convex hull area)"""
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        return contour_area / hull_area if hull_area > 0 else 0
    
    def analyze(self, model):
        """Run complete analysis pipeline"""
        if model is None:
            return None
            
        # Model prediction
        img_array = self.preprocess_image()
        pred = model.predict(img_array)[0,...,0]
        self.results['prob'] = float(np.max(pred)) * 100
        self.results['confidence'] = np.max(pred)
        
        # Lesion detection
        binary_mask = (pred > 0.6).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        serious_lesions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.is_serious_lesion(cnt, area):
                serious_lesions.append(cnt)
        
        if serious_lesions and self.results['confidence'] >= MIN_CONFIDENCE:
            self.results['is_cancer'] = True
            mask = np.zeros_like(pred)
            for cnt in serious_lesions:
                cv2.drawContours(mask, [cnt], -1, 1, thickness=cv2.FILLED)
            self.results['mask'] = cv2.resize(mask, self.image_size)
            self.results['lesions'] = serious_lesions
            self.results['advice'] = self._generate_advice(serious=True)
        else:
            self.results['advice'] = self._generate_advice(serious=False)
        
        return self.results
    
    def _generate_advice(self, serious):
        """Generate personalized medical advice"""
        if serious:
            return (
                "🚨 URGENT: Detected serious lesion with high malignancy probability. "
                "Immediate consultation with a dermatologist is strongly recommended. "
                "Consider biopsy for definitive diagnosis."
            )
        elif self.results['prob'] > 70:
            return (
                "⚠️ CAUTION: Suspicious features detected but below serious lesion threshold. "
                "Schedule a dermatology appointment within 2 weeks. "
                "Monitor for changes in size, shape or color."
            )
        elif self.results['prob'] > 40:
            return (
                "ℹ️ NOTE: Some atypical features present. "
                "Recommend clinical evaluation if lesion is new, changing, or concerning. "
                "Annual skin check advised."
            )
        else:
            return (
                "✅ REASSURANCE: No serious lesions detected. "
                "Continue regular self-exams and annual dermatology visits "
                "if you have risk factors."
            )

def display_skin_results(image, results):
    """Display skin analysis results with medical visualization"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, use_column_width=True, caption="Original Image")
    
    with col2:
        # Probability meter with color coding
        prob = results['prob']
        if prob > 85:
            color = "#e74c3c"  # Red
        elif prob > 70:
            color = "#f39c12"  # Orange
        else:
            color = "#2ecc71"  # Green
            
        st.markdown(
            f"""
<div style="margin-bottom: 1.5rem;">
    <h3 style="margin-bottom: 0.5rem;">Malignancy Probability</h3>
    <div style="background: #ecf0f1; border-radius: 20px; height: 30px;">
        <div style="background: {color}; width: {prob:.1f}%; border-radius: 20px; height: 30px; 
            display: flex; align-items: center; padding-left: 10px; color: white; font-weight: bold;">
            {prob:.1f}%
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
        
        # Mask display if serious lesions found
        if results['is_cancer'] and results['mask'] is not None:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(np.array(image))
            ax.imshow(results['mask'], alpha=0.4, cmap='Reds')
            ax.set_title("Detected Serious Lesions", fontsize=14, fontweight='bold')
            ax.axis('off')
            st.pyplot(fig)
            
            # Lesion statistics
            st.markdown("**Lesion Characteristics:**")
            stats = []
            for i, cnt in enumerate(results['lesions']):
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                stats.append(f"""
                - Lesion {i+1}: Area={area:.1f} px² | Perimeter={perimeter:.1f} px
                """)
            st.markdown("\n".join(stats))
        
        # Medical advice
        st.markdown("""
    <div style="background: #002b5c; padding: 1.5rem; border-radius: 12px; margin-top: 1.5rem; color: #ffffff;">
        <h3 style="color: #ffffff; margin-top: 0;">Medical Recommendation</h3>
        <p style="font-size: 1.1rem; margin-bottom: 0;">{}</p>
    </div>
""".format(results['advice']), unsafe_allow_html=True)

# Constants
DIABETES_MODEL_PATH = r"C:\Users\lenovo\Desktop\fproject\backend\models\diabetes_model.pkl"
SKIN_MODEL_PATH = r"C:\Users\lenovo\Desktop\fproject\backend\models\permanent_skin_model.h5"
HEART_MODEL_PATH = "../backend/models/saved_models/heart_disease_model.pkl"
HEART_SCALER_PATH = "../backend/models/saved_models/heart_scaler.pkl"
HEART_DATA_PATH = "../backend/models/data/heart.csv"
MIN_CONFIDENCE = 0.85  # 85% minimum confidence for serious lesions
MIN_AREA = 200  # Minimum lesion area in pixels
MAX_AGE_DAYS = 30  # Maximum cache age for model reload
# Constants
PNEUMONIA_MODEL_PATH = r"C:\Users\lenovo\Desktop\fproject\backend\models\pneumonia_classifier_model.h5"  # Add this with other paths

@st.cache_resource(ttl=86400)
def load_pneumonia_model():
    """Load and cache the pneumonia classification model"""
    try:
        model = tf.keras.models.load_model(PNEUMONIA_MODEL_PATH)
        st.success("✅ Pneumonia Detection Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Pneumonia Model loading failed: {str(e)}")
        return None

# Load models with caching
@st.cache_resource(ttl=86400)  # Cache for 1 day
def load_diabetes_model():
    """Load and cache the diabetes prediction model"""
    try:
        model = joblib.load(DIABETES_MODEL_PATH)
        st.success("✅ Diabetes Prediction Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Diabetes Model loading failed: {str(e)}")
        return None
    
@st.cache_resource(ttl=86400)  # Cache for 1 day
def load_skin_model():
    """Load and cache the TensorFlow skin model with custom objects"""
    try:
        model = tf.keras.models.load_model(
            SKIN_MODEL_PATH,
            custom_objects={
                'dice_coefficient': dice_coefficient,
                'binary_crossentropy': tf.keras.losses.binary_crossentropy
            }
        )
        st.success("✅ Skin Cancer Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Skin Model loading failed: {str(e)}")
        return None

@st.cache_data
def load_heart_model():
    """Load heart disease model and scaler"""
    try:
        model = joblib.load(HEART_MODEL_PATH)
        scaler = joblib.load(HEART_SCALER_PATH)
        st.success("✅ Heart Disease Model loaded successfully")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Heart Model loading failed: {str(e)}")
        return None, None

@st.cache_data
def load_heart_data():
    """Load heart disease dataset"""
    try:
        return pd.read_csv(HEART_DATA_PATH)
    except Exception as e:
        st.error(f"❌ Heart Data loading failed: {str(e)}")
        return None

def dice_coefficient(y_true, y_pred, smooth=1):
    """Dice coefficient metric for model evaluation"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

class LesionAnalyzer:
    """Advanced lesion analysis with medical-grade criteria"""
    
    def __init__(self, image):
        self.original_img = np.array(image)
        self.image_size = image.size
        self.results = {
            'mask': None,
            'prob': 0.0,
            'is_cancer': False,
            'lesions': [],
            'advice': "",
            'confidence': 0.0
        }
        
    def preprocess_image(self):
        """Prepare image for model input"""
        img_array = cv2.resize(self.original_img, (256, 256)) / 255.0
        return np.expand_dims(img_array, axis=0)
    
    def is_serious_lesion(self, contour, area):
        """Medical-grade criteria for serious lesions"""
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        convexity = cv2.matchShapes(contour, cv2.convexHull(contour), 1, 0.0)
        
        # Advanced shape analysis
        aspect_ratio = self._get_aspect_ratio(contour)
        solidity = self._get_solidity(contour)
        
        return all([
            area >= MIN_AREA,
            circularity < 0.7,
            convexity > 0.15,
            aspect_ratio > 1.5,
            solidity < 0.9
        ])
    
    def _get_aspect_ratio(self, contour):
        """Calculate aspect ratio of contour bounding box"""
        _, _, w, h = cv2.boundingRect(contour)
        return max(w, h) / min(w, h) if min(w, h) > 0 else 0
    
    def _get_solidity(self, contour):
        """Calculate solidity (area/convex hull area)"""
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        return contour_area / hull_area if hull_area > 0 else 0
    
    def analyze(self, model):
        """Run complete analysis pipeline"""
        if model is None:
            return None
            
        # Model prediction
        img_array = self.preprocess_image()
        pred = model.predict(img_array)[0,...,0]
        self.results['prob'] = float(np.max(pred)) * 100
        self.results['confidence'] = np.max(pred)
        
        # Lesion detection
        binary_mask = (pred > 0.6).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        serious_lesions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.is_serious_lesion(cnt, area):
                serious_lesions.append(cnt)
        
        if serious_lesions and self.results['confidence'] >= MIN_CONFIDENCE:
            self.results['is_cancer'] = True
            mask = np.zeros_like(pred)
            for cnt in serious_lesions:
                cv2.drawContours(mask, [cnt], -1, 1, thickness=cv2.FILLED)
            self.results['mask'] = cv2.resize(mask, self.image_size)
            self.results['lesions'] = serious_lesions
            self.results['advice'] = self._generate_advice(serious=True)
        else:
            self.results['advice'] = self._generate_advice(serious=False)
        
        return self.results
    
    def _generate_advice(self, serious):
        """Generate personalized medical advice"""
        if serious:
            return (
                "🚨 URGENT: Detected serious lesion with high malignancy probability. "
                "Immediate consultation with a dermatologist is strongly recommended. "
                "Consider biopsy for definitive diagnosis."
            )
        elif self.results['prob'] > 70:
            return (
                "⚠️ CAUTION: Suspicious features detected but below serious lesion threshold. "
                "Schedule a dermatology appointment within 2 weeks. "
                "Monitor for changes in size, shape or color."
            )
        elif self.results['prob'] > 40:
            return (
                "ℹ️ NOTE: Some atypical features present. "
                "Recommend clinical evaluation if lesion is new, changing, or concerning. "
                "Annual skin check advised."
            )
        else:
            return (
                "✅ REASSURANCE: No serious lesions detected. "
                "Continue regular self-exams and annual dermatology visits "
                "if you have risk factors."
            )

def display_skin_results(image, results):
    """Display skin analysis results with medical visualization"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, use_column_width=True, caption="Original Image")
    
    with col2:
        # Probability meter with color coding
        prob = results['prob']
        if prob > 85:
            color = "#e74c3c"  # Red
        elif prob > 70:
            color = "#f39c12"  # Orange
        else:
            color = "#2ecc71"  # Green
            
        st.markdown(
            f"""
<div style="margin-bottom: 1.5rem;">
    <h3 style="margin-bottom: 0.5rem;">Malignancy Probability</h3>
    <div style="background: #ecf0f1; border-radius: 20px; height: 30px;">
        <div style="background: {color}; width: {prob:.1f}%; border-radius: 20px; height: 30px; 
            display: flex; align-items: center; padding-left: 10px; color: white; font-weight: bold;">
            {prob:.1f}%
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
        
        # Mask display if serious lesions found
        if results['is_cancer'] and results['mask'] is not None:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(np.array(image))
            ax.imshow(results['mask'], alpha=0.4, cmap='Reds')
            ax.set_title("Detected Serious Lesions", fontsize=14, fontweight='bold')
            ax.axis('off')
            st.pyplot(fig)
            
            # Lesion statistics
            st.markdown("**Lesion Characteristics:**")
            stats = []
            for i, cnt in enumerate(results['lesions']):
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                stats.append(f"""
                - Lesion {i+1}: Area={area:.1f} px² | Perimeter={perimeter:.1f} px
                """)
            st.markdown("\n".join(stats))
        
        # Medical advice
        st.markdown("""
    <div style="background: #002b5c; padding: 1.5rem; border-radius: 12px; margin-top: 1.5rem; color: #ffffff;">
        <h3 style="color: #ffffff; margin-top: 0;">Medical Recommendation</h3>
        <p style="font-size: 1.1rem; margin-bottom: 0;">{}</p>
    </div>
""".format(results['advice']), unsafe_allow_html=True)

def pneumonia_detection_page(model):
    """Pneumonia detection from chest X-rays"""
    st.title("🫁 Pneumonia Detection")
    st.markdown("""
    <div class="guideline-box">
        <h3>📸 Chest X-ray Imaging Guidelines</h3>
        <ul style="font-size: 1.05rem;">
            <li><b>Position:</b> Posterior-Anterior (PA) view preferred</li>
            <li><b>Inclusion:</b> Entire lung fields, including costophrenic angles</li>
            <li><b>Quality:</b> Adequate penetration (vertebral bodies just visible through heart)</li>
            <li><b>Artifacts:</b> Remove jewelry, buttons, or other obscuring objects</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload chest X-ray image", 
        type=["jpg", "jpeg", "png"],
        help="PA view chest X-ray in DICOM or JPEG format"
    )
    
    if uploaded_file and model:
        try:
            # Remove .convert('L') to keep original channels
            image = Image.open(uploaded_file)  
            img_array = np.array(image)
            
            with st.spinner("🔍 Analyzing lung patterns..."):
                # Preprocess image
                img = cv2.resize(img_array, (150, 150))
                img = img / 255.0
                
                # Convert grayscale to RGB if needed
                if len(img.shape) == 2:  # If grayscale
                    img = np.stack((img,)*3, axis=-1)  # Convert to 3 channels
                
                img = np.expand_dims(img, axis=0)  # Add batch dimension
                
                # Make prediction
                raw_pred = model.predict(img)[0][0]
                prediction = 1 - raw_pred  # THIS IS THE CRITICAL FIX
                result = "Pneumonia" if prediction > 0.5 else "Normal"
                confidence = float(prediction if result == "Pneumonia" else 1 - prediction)
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, use_column_width=True, caption="Chest X-ray")
                
                with col2:
                    # Probability meter
                    color = "#e74c3c" if result == "Pneumonia" else "#2ecc71"
                    st.markdown(f"""
                    <div style="margin-bottom: 1.5rem;">
                        <h3>Pneumonia Probability</h3>
                        <div style="background: #ecf0f1; border-radius: 20px; height: 30px;">
                            <div style="background: {color}; width: {confidence*100:.1f}%; 
                                border-radius: 20px; height: 30px; display: flex; 
                                align-items: center; padding-left: 10px; color: white; 
                                font-weight: bold;">
                                {confidence*100:.1f}%
                            </div>
                        </div>
                        <p style="font-size: 1.2rem; font-weight: bold; color: {color};">
                            {result}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Generate medical advice
                    def generate_pneumonia_advice(result, confidence):
                        """Generate medical advice based on pneumonia detection results."""
                        if result == "Pneumonia":
                            return (
                                "🚨 URGENT: Pneumonia detected with high confidence. "
                                "Immediate consultation with a pulmonologist is strongly recommended. "
                                "Consider chest CT scan and blood tests for further evaluation."
                            )
                        else:
                            return (
                                "✅ REASSURANCE: No signs of pneumonia detected. "
                                "Maintain a healthy lifestyle and monitor for any respiratory symptoms. "
                                "Consult a doctor if symptoms persist or worsen."
                            )
                    
                    advice = generate_pneumonia_advice(result, confidence)
                    st.markdown(f"""
                    <div style="background: #002b5c; padding: 1.5rem; border-radius: 12px; 
                        margin-top: 1.5rem; color: #ffffff;">
                        <h3 style="color: #ffffff; margin-top: 0;">Medical Recommendation</h3>
                        <p style="font-size: 1.1rem; margin-bottom: 0;">{advice}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")




def heart_prediction_page(heart_model, heart_scaler):
    """Heart disease prediction page"""
    st.title("❤️ Heart Disease Prediction")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 20, 100, 50)
        sex = st.radio("Sex", ["Female", "Male"])
        sex_val = 0 if sex == "Female" else 1
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
        cp_val = ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"].index(cp)
        trestbps = st.number_input("Resting Blood Pressure (mmHg)", 90, 200, 120)

    with col2:
        chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dL?", ["No", "Yes"])
        fbs_val = 0 if fbs == "No" else 1
        restecg = st.selectbox("Resting ECG Results", ["Normal", "Abnormal ST-T", "LVH"])
        restecg_val = ["Normal", "Abnormal ST-T", "LVH"].index(restecg)
        thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)

    with col3:
        exang = st.radio("Exercise-induced Angina", ["No", "Yes"])
        exang_val = 0 if exang == "No" else 1
        oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
        slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)
        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", list(range(4)))
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
        thal_val = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 1

    if st.button("🔍 Predict Heart Disease"):
        input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val,
                             thalach, exang_val, oldpeak, slope_val, ca, thal_val]])
        
        with st.spinner("Analyzing cardiovascular risk..."):
            input_scaled = heart_scaler.transform(input_data)
            pred = heart_model.predict(input_scaled)[0]
            proba = heart_model.predict_proba(input_scaled)[0]

        st.subheader("🔎 Prediction Result")
        if pred == 1:
            st.error(f"⚠️ Heart Disease Likely (Confidence: {proba[1]*100:.1f}%)")
            st.markdown("""
            <div class="risk-alert">
                <h4>🚨 Recommended Actions:</h4>
                <ul>
                    <li>Schedule a cardiology consultation immediately</li>
                    <li>Consider additional tests (stress test, echocardiogram)</li>
                    <li>Monitor for chest pain, shortness of breath, or fatigue</li>
                    <li>Review cardiovascular risk factors with your doctor</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success(f"✅ No Heart Disease Detected (Confidence: {proba[0]*100:.1f}%)")
            st.markdown("""
            <div class="safe-alert">
                <h4>💚 Heart Health Tips:</h4>
                <ul>
                    <li>Maintain regular physical activity</li>
                    <li>Follow a heart-healthy diet (Mediterranean recommended)</li>
                    <li>Monitor blood pressure and cholesterol regularly</li>
                    <li>Annual check-ups if over 40 or with risk factors</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🔢 Probability Breakdown")
        st.bar_chart(pd.DataFrame({"Class": ["No Disease", "Disease Present"], "Probability": proba}).set_index("Class"))

def heart_data_exploration_page(heart_data):
    """Heart disease data exploration page"""
    st.title("📈 Heart Disease Data Exploration")
    
    tab1, tab2, tab3 = st.tabs(["📄 Dataset", "📊 Visualizations", "📚 Analysis"])

    with tab1:
        st.subheader("📋 Dataset Overview")
        st.dataframe(heart_data.head())
        st.text(f"Rows: {heart_data.shape[0]}, Columns: {heart_data.shape[1]}")
        st.text("Columns: " + ", ".join(heart_data.columns))
        st.text("Missing Values:")
        st.write(heart_data.isnull().sum())

    with tab2:
        st.subheader("📊 Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            sns.countplot(x='target', data=heart_data, palette='coolwarm')
            plt.title('Heart Disease Distribution')
            st.pyplot(fig)

            fig, ax = plt.subplots()
            sns.histplot(heart_data, x='age', hue='target', kde=True, bins=20, palette='coolwarm')
            plt.title('Age vs Heart Disease')
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            sns.countplot(x='sex', hue='target', data=heart_data, palette='viridis')
            plt.title('Gender vs Heart Disease')
            st.pyplot(fig)

            fig, ax = plt.subplots()
            sns.histplot(heart_data, x='chol', hue='target', kde=True, bins=20, palette='magma')
            plt.title('Cholesterol vs Heart Disease')
            st.pyplot(fig)

    with tab3:
        st.subheader("📚 Statistical Analysis")
        st.dataframe(heart_data.describe().T)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(heart_data.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        st.pyplot(fig)

def diabetes_prediction_page(diabetes_model):
    """Diabetes prediction page"""
    st.title("🩸 Diabetes Prediction")
    
    st.markdown("""
    <div class="guideline-box">
        <h3>📋 Clinical Parameters</h3>
        <p>Please provide accurate health metrics for reliable prediction:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # Column 1: Core medical features
    with col1:
        age = st.slider("Age", 1, 100, 30)
        hypertension = st.radio("Hypertension", ["No", "Yes"])
        hypertension_val = 0 if hypertension == "No" else 1
        heart_disease = st.radio("Heart Disease", ["No", "Yes"])
        heart_disease_val = 0 if heart_disease == "No" else 1
        bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    
    # Column 2: Diabetes-specific markers
    with col2:
        hba1c_level = st.slider("HbA1c Level", 3.5, 9.0, 5.0)
        blood_glucose_level = st.slider("Blood Glucose Level", 80, 300, 100)
        gender = st.radio("Gender", ["Female", "Male"])
        
        # Gender encoding (creates 2 binary features)
        gender_female = 1 if gender == "Female" else 0
        gender_male = 1 if gender == "Male" else 0
    
    # Column 3: Smoking history (must match training categories exactly)
    with col3:
        smoking = st.selectbox("Smoking History", [
            "never", 
            "No Info", 
            "current", 
            "former", 
            "ever", 
            "not current"
        ])
        
        # Smoking encoding (creates 6 binary features)
        smoking_never = 1 if smoking == "never" else 0
        smoking_noinfo = 1 if smoking == "No Info" else 0
        smoking_current = 1 if smoking == "current" else 0
        smoking_former = 1 if smoking == "former" else 0
        smoking_ever = 1 if smoking == "ever" else 0
        smoking_notcurrent = 1 if smoking == "not current" else 0

    if st.button("🔍 Predict Diabetes Risk"):
        # Assemble features in EXACT order used during training
        input_data = np.array([[
            age, hypertension_val, heart_disease_val, bmi, 
            hba1c_level, blood_glucose_level,
            gender_female, gender_male,
            smoking_noinfo, smoking_current, 
            smoking_ever, smoking_former,
            smoking_never, smoking_notcurrent
        ]])
        
        # Load the scaler (make sure to load this at the start of your app)
        scaler = joblib.load(r"C:\Users\lenovo\Desktop\fproject\backend\models\scaler.pkl")
        
        with st.spinner("Analyzing diabetes risk..."):
            try:
                # Scale the first 6 numeric features
                input_data_scaled = input_data.copy()
                input_data_scaled[:, :6] = scaler.transform(input_data[:, :6])
                
                pred = diabetes_model.predict(input_data_scaled)[0]
                proba = diabetes_model.predict_proba(input_data_scaled)[0]
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                return

        st.subheader("🔎 Prediction Result")
        if pred == 1:
            st.error(f"⚠️ Diabetes Likely (Confidence: {proba[1]*100:.1f}%)")
            st.markdown("""
            <div class="risk-alert">
                <h4>🚨 Recommended Actions:</h4>
                <ul>
                    <li>Consult with an endocrinologist or primary care physician</li>
                    <li>Get fasting blood glucose and HbA1c tests confirmed</li>
                    <li>Begin lifestyle modifications (diet and exercise)</li>
                    <li>Monitor blood sugar levels regularly</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success(f"✅ No Diabetes Detected (Confidence: {proba[0]*100:.1f}%)")
            st.markdown("""
            <div class="safe-alert">
                <h4>💚 Prevention Tips:</h4>
                <ul>
                    <li>Maintain healthy weight (BMI under 25)</li>
                    <li>Exercise at least 150 minutes per week</li>
                    <li>Limit processed sugars and refined carbs</li>
                    <li>Annual check-ups if over 40 or with risk factors</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Probability visualization
        st.markdown("### 🔢 Risk Probability Breakdown")
        st.bar_chart(pd.DataFrame({
            "Class": ["No Diabetes", "Diabetes"], 
            "Probability": proba
        }).set_index("Class"))




def skin_cancer_page(skin_model):
    """Skin cancer detection page"""
    st.title("🧬 Advanced Skin Cancer Detection")
    st.markdown("""
    <div class="guideline-box">
        <h3>📸 Clinical Imaging Guidelines</h3>
        <ul style="font-size: 1.05rem;">
            <li><b>Distance:</b> 6-12 inches from skin surface (15-30cm)</li>
            <li><b>Lighting:</b> Bright, diffuse illumination (avoid shadows)</li>
            <li><b>Focus:</b> Sharp focus on the lesion center</li>
            <li><b>Angles:</b> Include perpendicular and oblique views</li>
            <li><b>Scale:</b> Include ruler/coin for size reference when possible</li>
            <li><b>Background:</b> Solid neutral color (preferably blue or black)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced upload section
    with st.container():
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload dermatological image", 
            type=["jpg","jpeg","png"],
            accept_multiple_files=False,
            help="High-quality image of skin lesion"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file and skin_model:
        try:
            image = Image.open(uploaded_file)
            analyzer = LesionAnalyzer(image)
            
            with st.spinner("🔬 Analyzing lesion with medical AI..."):
                results = analyzer.analyze(skin_model)
            
            if results:
                display_skin_results(image, results)
                
                # Additional medical information
                with st.expander("📚 Clinical Interpretation Guide"):
                    st.markdown("""
                    **Probability Interpretation:**
                    - <85%: Low suspicion
                    - 85-94%: Moderate suspicion
                    - ≥95%: High suspicion
                    
                    **ABCDE Rule for Melanoma:**
                    - **A**symmetry: Irregular shape
                    - **B**order: Uneven or scalloped edges
                    - **C**olor: Multiple colors or uneven distribution
                    - **D**iameter: >6mm (pencil eraser size)
                    - **E**volving: Changing in size, shape or color
                    
                    **When to Seek Immediate Care:**
                    - Rapidly growing lesion
                    - Bleeding or ulceration
                    - Pain or itching
                    - Personal/family history of melanoma
                    """)
                    
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            st.info("Please try another image or contact support")

def home_page():
    """Home page with overview"""
    st.title("🏥 MedScan AI - Comprehensive Health Diagnostics")
    st.image(r"C:\Users\lenovo\Desktop\fproject\backend\models\data\image for front.png",
         width=700,
         caption="MedScan AI")
    
    st.markdown("""
    <div class="module-card">
        <h2>Welcome to MedScan AI</h2>
        <p>An advanced diagnostic platform combining dermatology and cardiology AI models for comprehensive health assessment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="module-card">
            <h3>🧬 Skin Cancer Detection</h3>
            <p>Advanced AI for early detection of melanoma and other skin cancers with medical-grade accuracy.</p>
            <ul>
                <li>Lesion shape analysis</li>
                <li>Malignancy probability scoring</li>
                <li>Clinical recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="module-card">
            <h3>❤️ Heart Disease Prediction</h3>
            <p>Cardiovascular risk assessment using clinical parameters and machine learning.</p>
            <ul>
                <li>Risk factor analysis</li>
                <li>Probability estimation</li>
                <li>Preventive recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-alert">
        <h3>⚠️ Important Disclaimer</h3>
        <p>This tool is for informational purposes only and does not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical concerns.</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application with enhanced medical UI"""
    # Custom medical-grade styling
    st.markdown("""
    <style>
        /* Professional medical UI */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2c3e50 0%, #1a252f 100%) !important;
            border-right: 1px solid #34495e !important;
        }
        .sidebar-title {
            color: #ecf0f1 !important;
            font-size: 24px !important;
            font-weight: 700 !important;
            text-align: center;
            padding: 1rem 0;
            border-bottom: 1px solid #34495e;
        }
        .risk-alert {
            border-left: 5px solid #e74c3c;
            padding: 1.5rem;
            background: linear-gradient(90deg, #fadbd8 0%, #f9ebea 100%);
            margin: 1.5rem 0;
            border-radius: 0 12px 12px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .safe-alert {
            border-left: 5px solid #27ae60;
            padding: 1.5rem;
            background: linear-gradient(90deg, #d5f5e3 0%, #e8f8f5 100%);
            margin: 1.5rem 0;
            border-radius: 0 12px 12px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .warning-alert {
            border-left: 5px solid #001f3f;
            padding: 1.5rem;
            background: #002b5c;
            color: white;
            margin: 1.5rem 0;
            border-radius: 0 12px 12px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .guideline-box {
            border: 2px solid #001f3f;
            padding: 1.75rem;
            border-radius: 12px;
            background-color: #002b5c;
            color: #ffffff;
            margin: 1.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        .module-card {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            border-radius: 12px;
            padding: 1.75rem;
            color: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            margin: 1rem 0;
            transition: transform 0.3s ease;
        }
        .module-card:hover {
            transform: translateY(-5px);
        }
        .upload-box {
            border: 2px dashed #3498db;
            border-radius: 12px;
            padding: 3rem 2rem;
            text-align: center;
            background: rgba(52, 152, 219, 0.05);
            margin: 1.5rem 0;
        }
        .stButton>button {
            background: linear-gradient(90deg, #3498db 0%, #2980b9 100%);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(52, 152, 219, 0.3);
        }
    </style>
    """, unsafe_allow_html=True)

    # Load all models and data
    with st.spinner("🔍 Loading AI Diagnostics Engines..."):
        skin_model = load_skin_model()
        heart_model, heart_scaler = load_heart_model()
        heart_data = load_heart_data()
        diabetes_model = load_diabetes_model()
        pneumonia_model = load_pneumonia_model()  # Load pneumonia model

    # Sidebar Navigation
    with st.sidebar:
        st.markdown('<p class="sidebar-title">MedScan AI</p>', unsafe_allow_html=True)
        selected = st.radio(
            "Navigation",
            ["🏠 Home", 
             "🩺 Skin Cancer Detection", 
             "❤️ Heart Disease Prediction",
             "🫁 Pneumonia Detection",
             "🩸 Diabetes Prediction",
             "📊 Heart Disease Data",
             "ℹ️ About"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("""
        <div style="padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 8px;">
            <h4 style="color: #ecf0f1; margin-bottom: 0.5rem;">System Status</h4>
            <p style="color: #bdc3c7; margin-bottom: 0.2rem;">Skin Model: Dermatology v4.2</p>
            <p style="color: #bdc3c7; margin-bottom: 0.2rem;">Heart Model: Cardiology v3.1</p>
            <p style="color: #bdc3c7; margin-bottom: 0.2rem;">Last Updated: {}</p>
            <p style="color: #2ecc71; font-weight: bold;">All Systems Operational</p>
        </div>
        """.format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)

    # Page routing
    if selected == "🏠 Home":
        home_page()
    elif selected == "🩺 Skin Cancer Detection":
        skin_cancer_page(skin_model)
    elif selected == "❤️ Heart Disease Prediction":
        heart_prediction_page(heart_model, heart_scaler)

    elif selected == "🫁 Pneumonia Detection":
        if pneumonia_model is not None:
            pneumonia_detection_page(pneumonia_model)
        else:
            st.error("Pneumonia model failed to load")
    
    elif selected == "📊 Heart Disease Data":
        if heart_data is not None:
            heart_data_exploration_page(heart_data)
        else:
            st.error("Heart disease data not available")

    elif selected == "🩸 Diabetes Prediction":  # <-- ADD THIS NEW CONDITION
       if diabetes_model is not None:
        diabetes_prediction_page(diabetes_model)
       else:
        st.error("Diabetes model failed to load")
    elif selected == "ℹ️ About":
        st.title("ℹ️ About MedScan AI")
        st.markdown("""
        <div class="module-card">
            <h3>Comprehensive Health Diagnostics Platform</h3>
            <p>MedScan AI combines multiple medical AI models into a unified diagnostic platform.</p>
            
            <h4>Key Features:</h4>
            <ul>
                <li>🧬 Skin Cancer Detection with lesion analysis</li>
                <li>❤️ Cardiovascular Risk Assessment</li>
                <li>📊 Clinical Data Visualization</li>
                <li>⚕️ Medical-grade recommendations</li>
            </ul>
            
            <h4>Technology Stack:</h4>
            <ul>
                <li>TensorFlow for skin lesion analysis</li>
                <li>Scikit-learn for heart disease prediction</li>
                <li>Streamlit for interactive interface</li>
            </ul>
            
            <h4>Disclaimer:</h4>
            <p>This application is for informational purposes only and does not constitute medical advice. 
            Always consult qualified healthcare professionals for medical diagnosis and treatment.</p>
            
            <p>Developed with ❤️ for better healthcare</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()