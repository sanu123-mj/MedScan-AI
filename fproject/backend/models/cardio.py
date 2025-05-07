# Import necessary libraries
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import joblib


def clean_data(df):
    df = df.dropna()  # remove rows with missing values
    df = df[df['age'] > 0]  # example rule: age must be positive
    df = df.drop_duplicates()
    return df  # return the cleaned dataframe


from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model  # return the trained model

# Set your dataset path
dataset_path = r"C:\Users\lenovo\Desktop\fproject\backend\models\data\heart.csv"

# Check if file exists and print the path
if os.path.exists(dataset_path):
    print(f"Found dataset at: {dataset_path}")
else:
    print(f"Dataset not found at: {dataset_path}")

# If you want to walk through directories (modified from Kaggle version)
# This will scan your data directory and print all files
data_directory = os.path.dirname(dataset_path)
print("\nFiles in data directory:")
for filename in os.listdir(data_directory):
    print(os.path.join(data_directory, filename))



import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import uniform

import warnings
warnings.filterwarnings('ignore')

# Set your dataset path
dataset_path = r"C:\Users\lenovo\Desktop\fproject\backend\models\data\heart.csv"

# Check if file exists and print the path
if os.path.exists(dataset_path):
    print(f"Found dataset at: {dataset_path}")
else:
    print(f"Dataset not found at: {dataset_path}")

# List files in your data directory
data_directory = os.path.dirname(dataset_path)
print("\nFiles in data directory:")
for filename in os.listdir(data_directory):
    print(os.path.join(data_directory, filename))




# Core data processing and visualization
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Model selection and evaluation
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    GridSearchCV, 
    RandomizedSearchCV
)

# Preprocessing and metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    precision_score, 
    recall_score
)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")



# Load dataset from your local path
df = pd.read_csv(r"C:\Users\lenovo\Desktop\fproject\backend\models\data\heart.csv")

# Display 10 random rows from the dataset
print("10 random rows from the dataset:")
print(df.sample(10))


# Optional: Show basic dataset info
print("\nDataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())



# 1. Count of gender values (sex)
print("Gender value counts (0 = female, 1 = male):")
print(df['sex'].value_counts())
print("\n")

# 2. Print dataset dimensions
print(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.\n")

# 3. Check for missing values
print("Missing values per column:")
print(df.isnull().sum())
print("\n")

# 4. Descriptive statistics (excluding sex and target columns)
print("Descriptive statistics (excluding sex and target columns):")
print(df.drop(columns=['sex', 'target']).describe().round(2))



# Alternative if not using Jupyter display:
# print(df.drop(columns=['sex', 'target']).describe().round(2).to_string())



# 1. Identify rows with invalid values (ca > 3 or thal == 0)
print("Rows with potential invalid values (ca > 3 or thal == 0):")
invalid_rows = df[(df['ca'] > 3) | (df['thal'] == 0)]
print(invalid_rows)  # or print(invalid_rows.to_string()) for regular Python

# 2. Remove invalid rows and create cleaned copy of dataframe
print("\nRemoving invalid rows...")
df_clean = df[~((df['ca'] > 3) | (df['thal'] == 0))].copy()

# 3. Show first 5 rows of cleaned dataset
print("\nFirst 5 rows of cleaned dataset:")
print(df_clean.head())  # or print(df_clean.head().to_string())

# 4. Show new dimensions of cleaned dataset
print("\nCleaned dataset shape (rows, columns):")
print(df_clean.shape)

# Optional: Compare original and cleaned sizes
print(f"\nRemoved {len(df) - len(df_clean)} rows ({((len(df)-len(df_clean))/len(df))*100:.2f}% of data)")




# Set the style for better visuals
sns.set_style("whitegrid")
plt.figure(figsize=(12, 5))

# 1. Distribution of target variable
plt.subplot(1, 2, 1)
ax = sns.countplot(x='target', data=df, palette="viridis")
plt.title('Distribution of Heart Disease Cases', pad=20, fontsize=14)
plt.xlabel('Diagnosis (0 = No Disease, 1 = Disease)', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Add percentage annotations
total = len(df)
for p in ax.patches:
    percentage = f'{100 * p.get_height()/total:.1f}%'
    ax.annotate(percentage, 
                (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 10), 
                textcoords='offset points',
                fontsize=11)

# 2. Distribution by sex
plt.subplot(1, 2, 2)
ax = sns.countplot(x='sex', hue='target', data=df, palette="coolwarm")
plt.title('Heart Disease Distribution by Gender', pad=20, fontsize=14)
plt.xlabel('Gender (0 = Female, 1 = Male)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Diagnosis', labels=['No Disease', 'Disease'], 
           bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and display
plt.tight_layout()
plt.show()

# Additional statistics
print("\nDisease Prevalence Statistics:")
print(f"Total cases: {len(df)}")
print(f"Disease prevalence: {df['target'].mean():.2%}")
print("\nGender-wise distribution:")
print(pd.crosstab(df['sex'], df['target'], 
      margins=True, 
      margins_name="Total",
      normalize='index').style.format("{:.2%}"))



# Set up the figure
plt.figure(figsize=(10, 6))

# Create the countplot with improved styling
ax = sns.countplot(x='cp', hue='target', data=df, 
                   palette="RdYlBu_r", 
                   edgecolor=".2",
                   linewidth=1.5)

# Add proper labels and title
plt.title('Heart Disease by Chest Pain Type', fontsize=14, pad=20)
plt.xlabel('Chest Pain Type', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Improve the x-axis labels with actual pain descriptions
chest_pain_types = {
    0: 'Typical Angina',
    1: 'Atypical Angina',
    2: 'Non-anginal Pain',
    3: 'Asymptomatic'
}
ax.set_xticklabels([chest_pain_types[x] for x in sorted(df['cp'].unique())])

# Enhance the legend
plt.legend(title='Diagnosis', 
           labels=['No Disease', 'Disease'],
           frameon=True,
           shadow=True,
           facecolor='white')

# Add percentage annotations
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 3,
            f'{height}',
            ha="center", 
            fontsize=10)

# Add statistical insight text
total_counts = df['cp'].value_counts().sort_index()
plt.text(0.02, 0.95, 
         f"Total cases by type:\n{total_counts.to_string()}",
         transform=ax.transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Show the plot
plt.tight_layout()
plt.show()

# Additional statistical analysis
print("\nDetailed Crosstab Analysis:")
cross_tab = pd.crosstab(df['cp'], df['target'], 
                       margins=True, 
                       margins_name="Total",
                       normalize='index')


print("Crosstab shape:", cross_tab.shape)
print("Crosstab:\n", cross_tab)

print("cross_tab columns before renaming:", cross_tab.columns)

# Fixing column renaming based on actual count
if cross_tab.shape[1] == 3:
    cross_tab.columns = ['No Disease', 'Disease', 'Total']
elif cross_tab.shape[1] == 2:
    cross_tab.columns = ['No Disease', 'Disease']
else:
    raise ValueError(f"Unexpected number of columns in cross_tab: {cross_tab.shape[1]}")


print(cross_tab.style.format("{:.2%}"))




import matplotlib.pyplot as plt
import seaborn as sns

# Set the style and palette
sns.set_style("whitegrid")
palette = {0: "#2ecc71", 1: "#e74c3c"}  # Green for no disease, red for disease

# 1. Chest Pain Type Visualization
plt.figure(figsize=(12, 6))
ax1 = sns.countplot(x='cp', hue='target', data=df, palette=palette)

# Add proper chest pain labels
chest_pain_labels = ['Typical Angina', 'Atypical Angina', 'Non-anginal', 'Asymptomatic']
ax1.set_xticklabels(chest_pain_labels)

plt.title('Heart Disease by Chest Pain Type', fontsize=14, pad=20)
plt.xlabel('Chest Pain Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Diagnosis', labels=['No Disease', 'Disease'])
plt.tight_layout()
plt.show()

# 2. Age Distribution Visualization
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='age', hue='target', kde=True, 
             element='step', palette=palette, bins=15)
plt.title("Age Distribution by Heart Disease Status", fontsize=14, pad=20)
plt.xlabel('Age (years)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(title='Diagnosis', labels=['No Disease', 'Disease'])
plt.tight_layout()
plt.show()

# 3. Blood Pressure Visualization
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='trestbps', hue='target', kde=True,
             element='step', palette=palette, bins=15)
plt.title('Resting Blood Pressure by Heart Disease Status', fontsize=14, pad=20)
plt.xlabel('Resting Blood Pressure (mm Hg)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(title='Diagnosis', labels=['No Disease', 'Disease'])
plt.tight_layout()
plt.show()

# 4. Cholesterol Visualization
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='chol', hue='target', kde=True,
             element='step', palette=palette, bins=15)
plt.title('Serum Cholesterol by Heart Disease Status', fontsize=14, pad=20)
plt.xlabel('Cholesterol (mg/dl)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(title='Diagnosis', labels=['No Disease', 'Disease'])
plt.tight_layout()
plt.show()

# Additional Statistical Summary
print("\n" + "="*50)
print("Key Statistical Summary".center(50))
print("="*50)
print(f"\nAverage age for heart disease: {df[df['target']==1]['age'].mean():.1f} years")
print(f"Average age without heart disease: {df[df['target']==0]['age'].mean():.1f} years")
print(f"\nAverage cholesterol for heart disease: {df[df['target']==1]['chol'].mean():.1f} mg/dl")
print(f"Average cholesterol without heart disease: {df[df['target']==0]['chol'].mean():.1f} mg/dl")
print(f"\nAverage blood pressure for heart disease: {df[df['target']==1]['trestbps'].mean():.1f} mm Hg")
print(f"Average blood pressure without heart disease: {df[df['target']==0]['trestbps'].mean():.1f} mm Hg")





# Set the style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# 1. Correlation Matrix
plt.figure(figsize=(14, 10))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
            center=0, fmt=".2f", linewidths=.5,
            annot_kws={"size": 10}, cbar_kws={"shrink": .8})
plt.title("Feature Correlation Matrix", fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 2. Age vs Maximum Heart Rate
plt.figure(figsize=(12, 7))
scatter = sns.scatterplot(data=df, x='age', y='thalach', hue='target',
                         palette=palette, alpha=0.8, s=100)
plt.title("Age vs Maximum Heart Rate by Heart Disease Status", fontsize=14, pad=20)
plt.xlabel('Age (years)', fontsize=12)
plt.ylabel('Maximum Heart Rate (thalach)', fontsize=12)
plt.legend(title='Diagnosis', labels=['No Disease', 'Disease'])
plt.grid(True, linestyle='--', alpha=0.3)

# Add regression lines
sns.regplot(data=df[df['target']==0], x='age', y='thalach', 
            scatter=False, color=palette[0], label='No Disease Trend')
sns.regplot(data=df[df['target']==1], x='age', y='thalach', 
            scatter=False, color=palette[1], label='Disease Trend')
plt.legend()
plt.tight_layout()
plt.show()

# 3. Fasting Blood Sugar Distribution
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='fbs', hue='target', data=df, palette=palette)
plt.title('Heart Disease by Fasting Blood Sugar Status', fontsize=14, pad=20)
plt.xlabel('Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Diagnosis', labels=['No Disease', 'Disease'])

# Add percentage labels
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height+5,
            f'{height}\n({height/len(df)*100:.1f}%)',
            ha="center", 
            fontsize=10)
plt.tight_layout()
plt.show()

# 4. Combined Features Plot
fig, ax = plt.subplots(1, 3, figsize=(20, 6))

# Slope plot
sns.countplot(x='slope', hue='target', data=df, ax=ax[0], palette=palette)
ax[0].set_title('Peak Exercise ST Segment Slope', fontsize=14)
ax[0].set_xlabel('Slope (0: upsloping, 1: flat, 2: downsloping)', fontsize=12)
ax[0].set_ylabel('Count', fontsize=12)
ax[0].legend(title='Diagnosis', labels=['No Disease', 'Disease'])

# Major Vessels plot
sns.countplot(x='ca', hue='target', data=df, ax=ax[1], palette=palette)
ax[1].set_title('Number of Major Vessels', fontsize=14)
ax[1].set_xlabel('Major Vessels Colored by Fluoroscopy', fontsize=12)
ax[1].set_ylabel('')
ax[1].legend().set_visible(False)

# Thalassemia plot
sns.countplot(x='thal', hue='target', data=df, ax=ax[2], palette=palette)
ax[2].set_title('Thalassemia Types', fontsize=14)
ax[2].set_xlabel('Thalassemia (1: normal, 2: fixed defect, 3: reversible defect)', fontsize=12)
ax[2].set_ylabel('')
ax[2].legend().set_visible(False)

plt.suptitle("Heart Disease Across Key Clinical Features", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# Print key correlations
print("\nTop Correlations with Heart Disease:")
print(corr_matrix['target'].sort_values(ascending=False)[1:6])
print("\nTop Negative Correlations with Heart Disease:")
print(corr_matrix['target'].sort_values()[:5])




from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 1. Data Preparation
X = df.drop('target', axis=1)
y = df['target']

# Split data with stratification to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=10,
    stratify=y  # Important for imbalanced datasets
)

print(f"\nTrain set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"\nClass distribution in training set:\n{y_train.value_counts(normalize=True)}")
print(f"\nClass distribution in test set:\n{y_test.value_counts(normalize=True)}")

# 2. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
X_test_scaled = scaler.transform(X_test)       # Transform test data using training parameters

print("\nFeature scaling completed:")
print(f"Example scaled mean: {X_train_scaled[:, 0].mean():.2f}")
print(f"Example scaled std: {X_train_scaled[:, 0].std():.2f}")

# 3. Model Initialization
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=10),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(probability=True, random_state=10),
    "Random Forest": RandomForestClassifier(random_state=10)
}

# 4. Model Training
print("\nTraining models...")
for name, model in models.items():
    # Use scaled data for KNN, original for others
    train_data = X_train_scaled if name == "K-Nearest Neighbors" else X_train
    model.fit(train_data, y_train)
    print(f"{name} trained successfully")

# 5. Verify training
print("\nTraining verification:")
for name, model in models.items():
    train_data = X_train_scaled if name == "K-Nearest Neighbors" else X_train
    score = model.score(train_data, y_train)
    print(f"{name} training accuracy: {score:.4f}")






from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score


log_reg = LogisticRegression(solver='liblinear', C=0.1, penalty='l2')
log_reg.fit(X_train, y_train)

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Standardize the features
scaler_knn = StandardScaler()
X_train_stand = scaler_knn.fit_transform(X_train)
X_test_stand = scaler_knn.transform(X_test)

# Create and train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_stand, y_train)


scaler_svc = StandardScaler()
X_train_svc = scaler_svc.fit_transform(X_train)
X_test_svc = scaler_svc.transform(X_test)

# Define and train the SVC model
svc = SVC(kernel='rbf', C=1.0, gamma='scale')  # You can adjust parameters as needed
svc.fit(X_train_svc, y_train)


# Predictions for different models
y_pred_lr = log_reg.predict(X_test)
y_pred_knn = knn.predict(X_test_stand)
y_pred_svc = svc.predict(X_test)

print(f"Accuracy of Logistic Regression : {accuracy_score(y_test, y_pred_lr) * 100}%")
print(f"Accuracy of KNN : {accuracy_score(y_test, y_pred_knn) * 100}%")
print(f"Accuracy of SVC : {accuracy_score(y_test, y_pred_svc) * 100}%")

# Accuracy scores using 5-fold cross-validation
for model in [log_reg, knn, svc]:
    print(f"Accuracy of {model.__class__.__name__} for 5 folds is : {np.mean(cross_val_score(model, X, y, scoring='accuracy', cv=5)) * 100}%")

# Recall scores using 5-fold cross-validation
for model in [log_reg, knn, svc]:
    print(f"The recall score of {model.__class__.__name__} for 5 folds is : {np.mean(cross_val_score(model, X, y, scoring='recall', cv=5)) * 100}%")

# Random forest is a tree-based model, which doesn't require scaling.
from sklearn.ensemble import RandomForestClassifier

# Define the model
rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)


rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"Accuracy Score of {rf.__class__.__name__} is {accuracy_score(y_test, y_pred) * 100}%")
print(f"Recall Score of {rf.__class__.__name__} is {recall_score(y_test, y_pred) * 100}%")
print(f"Precision Score of {rf.__class__.__name__} is {precision_score(y_test, y_pred) * 100}%")

# Performing 5 Fold Cross Validation
print(f"Recall Score of {rf.__class__.__name__} is {np.mean(cross_val_score(rf, X, y, cv=5, scoring='recall')) * 100}%")

# for logistic regression
logistic_regression = LogisticRegression(solver='liblinear')  # Use 'liblinear' for small datasets

# Parameter grid for Logistic Regression
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}

# Grid Search for Logistic Regression, scoring = recall
grid_search_lr = GridSearchCV(estimator=logistic_regression, param_grid=param_grid_lr, cv=5, scoring='recall')
grid_search_lr.fit(X_train, y_train)  # X_train and y_train are the training data

# Best parameters and score
print("Best parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best score for Logistic Regression:", grid_search_lr.best_score_)

random_forest = RandomForestClassifier()

# Parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Randomized Search for Random Forest
random_search_rf = RandomizedSearchCV(estimator=random_forest, param_distributions=param_grid_rf, n_iter=20, cv=5, scoring='recall', random_state=42)
random_search_rf.fit(X_train, y_train)

# Best parameters and score
print("Best parameters for Random Forest:", random_search_rf.best_params_)
print("Best score for Random Forest:", random_search_rf.best_score_)

log_reg = LogisticRegression(solver='liblinear', C=0.1, penalty='l2')
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

print(f"Accuracy Score of {log_reg.__class__.__name__} is {accuracy_score(y_test, y_pred) * 100}%")
print(f"Recall Score of {log_reg.__class__.__name__} is {recall_score(y_test, y_pred) * 100}%")
print(f"Precision Score of {log_reg.__class__.__name__} is {precision_score(y_test, y_pred) * 100}%")

# but we want high recall score, which is 
print(f"Recall Score of {log_reg.__class__.__name__} is {recall_score(y_test, y_pred) * 100}%")




















# At the bottom of cardio.py
if __name__ == "__main__":
    # This will run when you execute cardio.py directly
    print("Running analysis...")
    print(f"Dataset shape: {df.shape}")
    print(f"Models trained: {list(models.keys())}")

final_model = random_search_rf.best_estimator_  # Using best model from randomized search
scaler = StandardScaler().fit(X_train)  # Save scaler used for KNN if needed

# Create a model directory if it doesn't exist
model_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
os.makedirs(model_dir, exist_ok=True)

# Save components
joblib.dump(final_model, os.path.join(model_dir, 'heart_disease_model.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'heart_scaler.pkl'))
print("\nModel and scaler saved successfully in directory:", model_dir)
