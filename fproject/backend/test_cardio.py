import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2
import random
from datetime import datetime

# Constants
MODEL_PATH = r"C:\Users\lenovo\Desktop\fproject\backend\models\permanent_skin_model.h5"
MIN_CONFIDENCE = 0.85  # 85% minimum confidence for serious lesions
MIN_AREA = 200  # Minimum lesion area in pixels
MAX_AGE_DAYS = 30  # Maximum cache age for model reload

@st.cache_resource(ttl=86400)  # Cache for 1 day
def load_model():
    """Load and cache the TensorFlow model with custom objects"""
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                'dice_coefficient': dice_coefficient,
                'binary_crossentropy': tf.keras.losses.binary_crossentropy
            }
        )
        st.success("✅ AI Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
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

def display_results(image, results):
    """Display analysis results with medical visualization"""
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
            
        st.markdown(f"""
        <div style="margin-bottom: 1.5rem;">
            <h3 style="margin-bottom: 0.5rem;">Malignancy Probability</h3>
            <div style="background: #ecf0f1; border-radius: 20px; height: 30px;">
                <div style="background: {color}; width: {prob}%; border-radius: 20px; height: 30px; 
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


def main():
    """Main application with enhanced medical UI"""
    # Configure page
    st.set_page_config(
        page_title="MedScan AI - Dermatology Diagnostics", 
        page_icon="⚕️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://medical-ai-support.com',
            'Report a bug': "https://medscan-issues.com",
            'About': "### MedScan AI v4.0\nAdvanced dermatology diagnostics"
        }
    )
    
    # Load model with progress indicator
    with st.spinner("🔍 Loading AI Diagnostics Engine..."):
        model = load_model()
    
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
    border-left: 5px solid #001f3f; /* Dark navy border */
    padding: 1.5rem;
    background: #002b5c;            /* Solid dark blue background */
    color: white;                  /* White text for visibility */
    margin: 1.5rem 0;
    border-radius: 0 12px 12px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

       .guideline-box {
    border: 2px solid #001f3f; /* Dark navy blue border */
    padding: 1.75rem;
    border-radius: 12px;
    background-color: #002b5c; /* Solid dark blue background */
    color: #ffffff; /* White text */
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

    # Sidebar Navigation
    with st.sidebar:
        ('<p class="sidebar-title">MedScan AI</p>', unsafe_allow_html=True)
        selected = st.radio(
            "Navigation",
            ["🏠 Home", "🩺 Skin Cancer", "🫁 Lung Health", "❤️ Heart Health", "👀 Eye Care", "🧠 Neurology"],
            index=1 if 'skin_cancer' in st.session_state else 0
        )
        
        st.markdown("---")
        st.markdown("""
        <div style="padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 8px;">
            <h4 style="color: #ecf0f1; margin-bottom: 0.5rem;">System Status</h4>
            <p style="color: #bdc3c7; margin-bottom: 0.2rem;">Model: Dermatology v4.2</p>
            <p style="color: #bdc3c7; margin-bottom: 0.2rem;">Last Updated: {}</p>
            <p style="color: #2ecc71; font-weight: bold;">Operational</p>
        </div>
        """.format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)

    # Skin Cancer Page
    if selected == "🩺 Skin Cancer":
        st.session_state.skin_cancer = True
        
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
        
        if uploaded_file and model:
            try:
                image = Image.open(uploaded_file)
                analyzer = LesionAnalyzer(image)
                
                with st.spinner("🔬 Analyzing lesion with medical AI..."):
                    results = analyzer.analyze(model)
                
                if results:
                    display_results(image, results)
                    
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

    # Other pages remain unchanged...
    # [Previous implementation of other pages goes here]

if __name__ == "__main__":
    main()