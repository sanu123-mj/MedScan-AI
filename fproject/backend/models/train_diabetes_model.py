# diabetes_model_training.py
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.combine import SMOTETomek

# =============================================
# FIXED FILE LOADING WITH ERROR HANDLING
# =============================================
def load_diabetes_data():
    """Robust data loading function"""
    data_path = r"C:\Users\lenovo\Desktop\fproject\backend\models\data\diabetes_prediction_dataset.csv"
    
    # Verify file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found at: {data_path}")
    
    # Try multiple reading methods
    try:
        return pd.read_csv(data_path)
    except PermissionError:
        try:
            # Alternative read method
            with open(data_path, 'rb') as f:
                return pd.read_csv(io.BytesIO(f.read()))
        except Exception as e:
            raise PermissionError(f"Could not read file. Please close Excel/other programs using it.\nError: {str(e)}")

# Load the data
try:
    DiabetesData = load_diabetes_data()
except Exception as e:
    print(f"ERROR LOADING DATA: {str(e)}")
    exit()

# =============================================
# YOUR ORIGINAL CODE (UNCHANGED)
# =============================================
DiabetesData.head()

# Check for missing values
print(DiabetesData.isnull().sum())

# Check for duplicates
duplicates = DiabetesData[DiabetesData.duplicated()]
print("Number of duplicate rows: ", duplicates.shape)
DiabetesData = DiabetesData.drop_duplicates()

# Describe the data
DiabetesData.describe()

# Visualizations (your original plots)
sns.countplot(x="diabetes", data=DiabetesData)
plt.title("Diabetes outcome distribution")
plt.show()

sns.countplot(x="heart_disease", data=DiabetesData)
plt.title("Heart disease distribution")
plt.show()

sns.countplot(x="hypertension", data=DiabetesData)
plt.title("Hypertension distribution")
plt.show()

sns.countplot(x='smoking_history', data=DiabetesData)
plt.title('Smoking History Distribution')
plt.show()

plt.hist(x='age', data=DiabetesData, edgecolor="black")
plt.title('Age Distribution')
plt.show()

sns.countplot(x='gender', hue='diabetes', data=DiabetesData)
plt.title('Gender vs Diabetes')
plt.show()

DiabetesData = DiabetesData[DiabetesData['gender'] != 'Other']
sns.countplot(x='gender', hue='diabetes', data=DiabetesData)
plt.title('Gender vs Diabetes')
plt.show()

sns.boxplot(x='diabetes', y='blood_glucose_level', data=DiabetesData)
plt.title('Blood Glucose Level vs Diabetes')
plt.show()

sns.boxplot(x='diabetes', y='HbA1c_level', data=DiabetesData)
plt.title('HbA1c level vs Diabetes')
plt.show()

sns.boxplot(x='diabetes', y='bmi', data=DiabetesData)
plt.title('BMI vs Diabetes')
plt.show()

sns.boxplot(x='diabetes', y='age', data=DiabetesData)
plt.title('Age vs Diabetes')
plt.show()

# Feature engineering
encoded_gender = pd.get_dummies(DiabetesData['gender'], prefix='gender_encoded')
encoded_smoking = pd.get_dummies(DiabetesData['smoking_history'], prefix='smoking_history_encoded')
DiabetesData = pd.concat([DiabetesData.drop('gender', axis=1), encoded_gender], axis=1)
DiabetesData = pd.concat([DiabetesData.drop('smoking_history', axis=1), encoded_smoking], axis=1)

# Correlation matrices
plt.figure(figsize=(16,6))
sns.heatmap(DiabetesData.corr(), vmin=-1, vmax=1, annot=True)
plt.title("Correlation matrix")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(DiabetesData.corr()[['diabetes']].sort_values(by='diabetes', ascending=False), 
            vmin=-1, vmax=1, annot=True)
plt.title("Features correlating with diabetes")
plt.show()

# Prepare data
X = DiabetesData.drop('diabetes', axis=1)
y = DiabetesData['diabetes']

# Handle imbalance
smt = SMOTETomek(random_state=42)
X, y = smt.fit_resample(X, y)

# Visualize after SMOTE
resampled_data = pd.DataFrame(y, columns=['diabetes'])
sns.countplot(x="diabetes", data=resampled_data)
plt.title("Diabetes outcome distribution after SMOTETomek")
plt.show()

# Scale numeric features
numeric = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical = [col for col in DiabetesData.columns if 'gender_encoded' in col or 'smoking_history_encoded' in col]
scaler = StandardScaler()
X_numeric_std = pd.DataFrame(scaler.fit_transform(X[numeric]), columns=numeric)
X_std = pd.concat([X_numeric_std, X[categorical]], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

# Model training
param_grid = {
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

DTModel = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=DTModel, 
    param_grid=param_grid, 
    cv=5, 
    n_jobs=-1, 
    verbose=2, 
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

# Get best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluation
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print('Best Parameters:', best_params)
print(f'Best Accuracy: {accuracy_best}')

y_pred = grid_search.predict(X_test)
print("Model Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# =============================================
# FIXED MODEL SAVING TO CORRECT FOLDER
# =============================================
model_save_dir = r"C:\Users\lenovo\Desktop\fproject\backend\models"
os.makedirs(model_save_dir, exist_ok=True)

# Save model
model_path = os.path.join(model_save_dir, "diabetes_model.pkl")
joblib.dump(best_model, model_path)
print(f"Model saved to: {model_path}")

# Save scaler
scaler_path = os.path.join(model_save_dir, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to: {scaler_path}")



# After all your code, add:
plt.show()  # This keeps graphs open