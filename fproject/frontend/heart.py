import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report

# Page config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data and model
@st.cache_data

def load_data():
    return pd.read_csv("../backend/models/data/heart.csv")

@st.cache_data

def load_model():
    model = joblib.load("../backend/models/saved_models/heart_disease_model.pkl")
    scaler = joblib.load("../backend/models/saved_models/heart_scaler.pkl")
    return model, scaler

# Sidebar navigation
st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Data Exploration", "🧠 Model Prediction", "ℹ️ About"])

# Load assets
df = load_data()
model, scaler = load_model()

# Home
if page.startswith("🏠"):
    st.title("❤️ Heart Disease Prediction System")
    st.image("https://www.heart.org/-/media/Images/Health-Topics/Heart-Attack/heart-attack-risk-factors.jpg", width=700)
    
    st.markdown("""
    Welcome to the **Heart Disease Prediction System**, a tool designed to assist healthcare professionals and patients in assessing heart disease risk using machine learning algorithms.
    
    🚀 **Features:**
    - Explore heart disease dataset with charts & stats
    - Predict based on input features
    - High performance Random Forest classifier
    
    👉 Use the sidebar to navigate through sections.
    """)

# Data Exploration
elif page.startswith("📊"):
    st.title("📈 Data Exploration Dashboard")
    tab1, tab2, tab3 = st.tabs(["📄 Dataset", "📊 Visualizations", "📚 Analysis"])

    with tab1:
        st.subheader("📋 Dataset Overview")
        st.dataframe(df.head())
        st.text(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.text("Columns: " + ", ".join(df.columns))
        st.text("Missing Values:")
        st.write(df.isnull().sum())

    with tab2:
        st.subheader("📊 Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            sns.countplot(x='target', data=df, palette='coolwarm')
            plt.title('Heart Disease Distribution')
            st.pyplot(fig)

            fig, ax = plt.subplots()
            sns.histplot(df, x='age', hue='target', kde=True, bins=20, palette='coolwarm')
            plt.title('Age vs Heart Disease')
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            sns.countplot(x='sex', hue='target', data=df, palette='viridis')
            plt.title('Gender vs Heart Disease')
            st.pyplot(fig)

            fig, ax = plt.subplots()
            sns.histplot(df, x='chol', hue='target', kde=True, bins=20, palette='magma')
            plt.title('Cholesterol vs Heart Disease')
            st.pyplot(fig)

    with tab3:
        st.subheader("📚 Statistical Analysis")
        st.dataframe(df.describe().T)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        st.pyplot(fig)

# Model Prediction
elif page.startswith("🧠"):
    st.title("💡 Predict Heart Disease")
    st.markdown("Fill in the patient data to predict the likelihood of heart disease.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 20, 100, 50)
        sex = st.radio("Sex", ["Female", "Male"])
        sex_val = 0 if sex == "Female" else 1
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
        cp_val = ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"].index(cp)
        trestbps = st.number_input("Resting Blood Pressure", 90, 200, 120)

    with col2:
        chol = st.number_input("Cholesterol", 100, 600, 200)
        fbs = st.radio("Fasting Blood Sugar > 120?", ["No", "Yes"])
        fbs_val = 0 if fbs == "No" else 1
        restecg = st.selectbox("ECG Results", ["Normal", "Abnormal ST-T", "LVH"])
        restecg_val = ["Normal", "Abnormal ST-T", "LVH"].index(restecg)
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)

    with col3:
        exang = st.radio("Exercise-induced Angina", ["No", "Yes"])
        exang_val = 0 if exang == "No" else 1
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"])
        slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)
        ca = st.selectbox("Major Vessels Colored", list(range(4)))
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
        thal_val = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 1

    input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val,
                             thalach, exang_val, oldpeak, slope_val, ca, thal_val]])

    if st.button("🔍 Predict"):
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        st.subheader("🔎 Prediction Result")
        if pred == 1:
            st.error(f"⚠️ Heart Disease Likely (Confidence: {proba[1]*100:.1f}%)")
        else:
            st.success(f"✅ No Heart Disease Detected (Confidence: {proba[0]*100:.1f}%)")

        st.markdown("### 🔢 Probability Breakdown")
        st.bar_chart(pd.DataFrame({"Class": ["No", "Yes"], "Probability": proba}).set_index("Class"))

# About Page
elif page.startswith("ℹ️"):
    st.title("ℹ️ About This App")
    st.markdown("""
    ### Heart Disease ML App
    Built using `Streamlit`, this app applies a `Random Forest` model trained on the **UCI Heart Disease dataset**.

    **Key Points:**
    - Clinical dataset of 303 records
    - Optimized for healthcare interpretation
    - Emphasis on reducing false negatives

    **Disclaimer:** Not a substitute for medical diagnosis. Always consult professionals.
    
    **Developer:** Your Name | [GitHub](https://github.com/yourname)
    """)

if __name__ == '__main__':
    pass
