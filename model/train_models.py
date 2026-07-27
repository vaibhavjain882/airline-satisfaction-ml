"""
Airline Passenger Satisfaction Classification Model Training
============================================================
This script trains 5 ML models and calculates 6 evaluation metrics for each.

Dataset: Airline Passenger Satisfaction
Classes: Satisfied / Neutral or Dissatisfied
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# Import ML models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# ==============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ==============================================================================

print("=" * 70)
print("STEP 1: Loading Data")
print("=" * 70)

# Load train and test data from Kaggle
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Combine both datasets
df = pd.concat([train_df, test_df], ignore_index=True)

print(f"✓ Dataset loaded successfully!")
print(f"  - Train shape: {train_df.shape}")
print(f"  - Test shape: {test_df.shape}")
print(f"  - Combined shape: {df.shape} (rows, columns)")
print(f"  - Features: {df.columns.tolist()}")

# ==============================================================================
# STEP 2: DATA PREPROCESSING
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 2: Data Preprocessing")
print("=" * 70)

# Remove rows with missing values
df = df.dropna()
print(f"✓ Missing values removed")

# Handle categorical variables (convert to numeric)
label_encoders = {}
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Remove 'satisfaction' from categorical columns (it's our target)
if 'satisfaction' in categorical_columns:
    categorical_columns.remove('satisfaction')

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"✓ Encoded {col}")

# Separate features (X) and target (y)
X = df.drop('satisfaction', axis=1)
y = df['satisfaction']

# Encode target variable
le_target = LabelEncoder()
y = le_target.fit_transform(y)

print(f"✓ Data shape: X={X.shape}, y={y.shape}")
print(f"✓ Class distribution: {np.bincount(y)}")

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Train-Test Split: Train={X_train.shape}, Test={X_test.shape}")

# Scale the features (important for distance-based models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"✓ Features scaled using StandardScaler")

# Save scaler for later use in Streamlit app
joblib.dump(scaler, 'model/scaler.pkl')
print(f"✓ Scaler saved")

# ==============================================================================
# STEP 3: TRAIN MODELS AND CALCULATE METRICS
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 3: Training Models and Calculating Metrics")
print("=" * 70)

# Dictionary to store results
results = {}

# ==============================================================================
# MODEL 1: LOGISTIC REGRESSION
# ==============================================================================

print("\n[1/5] Training Logistic Regression...")
model_lr = LogisticRegression(random_state=42, max_iter=1000)
model_lr.fit(X_train_scaled, y_train)
y_pred_lr = model_lr.predict(X_test_scaled)
y_pred_proba_lr = model_lr.predict_proba(X_test_scaled)[:, 1]

results['Logistic Regression'] = {
    'Accuracy': accuracy_score(y_test, y_pred_lr),
    'AUC': roc_auc_score(y_test, y_pred_proba_lr),
    'Precision': precision_score(y_test, y_pred_lr, average='weighted'),
    'Recall': recall_score(y_test, y_pred_lr, average='weighted'),
    'F1': f1_score(y_test, y_pred_lr, average='weighted'),
    'MCC': matthews_corrcoef(y_test, y_pred_lr),
    'Confusion_Matrix': confusion_matrix(y_test, y_pred_lr)
}

joblib.dump(model_lr, 'model/logistic_regression.pkl')
print(f"  ✓ Accuracy: {results['Logistic Regression']['Accuracy']:.4f}")

# ==============================================================================
# MODEL 2: DECISION TREE CLASSIFIER
# ==============================================================================

print("\n[2/5] Training Decision Tree Classifier...")
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train_scaled, y_train)
y_pred_dt = model_dt.predict(X_test_scaled)
y_pred_proba_dt = model_dt.predict_proba(X_test_scaled)[:, 1]

results['Decision Tree'] = {
    'Accuracy': accuracy_score(y_test, y_pred_dt),
    'AUC': roc_auc_score(y_test, y_pred_proba_dt),
    'Precision': precision_score(y_test, y_pred_dt, average='weighted'),
    'Recall': recall_score(y_test, y_pred_dt, average='weighted'),
    'F1': f1_score(y_test, y_pred_dt, average='weighted'),
    'MCC': matthews_corrcoef(y_test, y_pred_dt),
    'Confusion_Matrix': confusion_matrix(y_test, y_pred_dt)
}

joblib.dump(model_dt, 'model/decision_tree.pkl')
print(f"  ✓ Accuracy: {results['Decision Tree']['Accuracy']:.4f}")

# ==============================================================================
# MODEL 3: K-NEAREST NEIGHBORS (KNN)
# ==============================================================================

print("\n[3/5] Training K-Nearest Neighbors (KNN)...")
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train_scaled, y_train)
y_pred_knn = model_knn.predict(X_test_scaled)
y_pred_proba_knn = model_knn.predict_proba(X_test_scaled)[:, 1]

results['K-Nearest Neighbor'] = {
    'Accuracy': accuracy_score(y_test, y_pred_knn),
    'AUC': roc_auc_score(y_test, y_pred_proba_knn),
    'Precision': precision_score(y_test, y_pred_knn, average='weighted'),
    'Recall': recall_score(y_test, y_pred_knn, average='weighted'),
    'F1': f1_score(y_test, y_pred_knn, average='weighted'),
    'MCC': matthews_corrcoef(y_test, y_pred_knn),
    'Confusion_Matrix': confusion_matrix(y_test, y_pred_knn)
}

joblib.dump(model_knn, 'model/knn.pkl')
print(f"  ✓ Accuracy: {results['K-Nearest Neighbor']['Accuracy']:.4f}")

# ==============================================================================
# MODEL 4: NAIVE BAYES CLASSIFIER
# ==============================================================================

print("\n[4/5] Training Naive Bayes Classifier...")
model_nb = GaussianNB()
model_nb.fit(X_train_scaled, y_train)
y_pred_nb = model_nb.predict(X_test_scaled)
y_pred_proba_nb = model_nb.predict_proba(X_test_scaled)[:, 1]

results['Naive Bayes'] = {
    'Accuracy': accuracy_score(y_test, y_pred_nb),
    'AUC': roc_auc_score(y_test, y_pred_proba_nb),
    'Precision': precision_score(y_test, y_pred_nb, average='weighted'),
    'Recall': recall_score(y_test, y_pred_nb, average='weighted'),
    'F1': f1_score(y_test, y_pred_nb, average='weighted'),
    'MCC': matthews_corrcoef(y_test, y_pred_nb),
    'Confusion_Matrix': confusion_matrix(y_test, y_pred_nb)
}

joblib.dump(model_nb, 'model/naive_bayes.pkl')
print(f"  ✓ Accuracy: {results['Naive Bayes']['Accuracy']:.4f}")

# ==============================================================================
# MODEL 5: RANDOM FOREST (ENSEMBLE)
# ==============================================================================

print("\n[5/5] Training Random Forest Classifier...")
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train_scaled, y_train)
y_pred_rf = model_rf.predict(X_test_scaled)
y_pred_proba_rf = model_rf.predict_proba(X_test_scaled)[:, 1]

results['Random Forest'] = {
    'Accuracy': accuracy_score(y_test, y_pred_rf),
    'AUC': roc_auc_score(y_test, y_pred_proba_rf),
    'Precision': precision_score(y_test, y_pred_rf, average='weighted'),
    'Recall': recall_score(y_test, y_pred_rf, average='weighted'),
    'F1': f1_score(y_test, y_pred_rf, average='weighted'),
    'MCC': matthews_corrcoef(y_test, y_pred_rf),
    'Confusion_Matrix': confusion_matrix(y_test, y_pred_rf)
}

joblib.dump(model_rf, 'model/random_forest.pkl')
print(f"  ✓ Accuracy: {results['Random Forest']['Accuracy']:.4f}")

# ==============================================================================
# STEP 4: DISPLAY RESULTS
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 4: Model Comparison Results")
print("=" * 70)

# Create results dataframe
results_df = pd.DataFrame({
    model_name: {metric: value for metric, value in metrics.items() if metric != 'Confusion_Matrix'}
    for model_name, metrics in results.items()
}).T

print("\n" + results_df.to_string())

# Save results to CSV
results_df.to_csv('model/metrics.csv')
print("\n✓ Metrics saved to model/metrics.csv")

# ==============================================================================
# STEP 5: SAVE TEST DATA AND LABELS
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 5: Saving Test Data for Streamlit App")
print("=" * 70)

# Create test data dataframe
test_df = X_test.copy()
test_df['satisfaction'] = y_test

# Save test data (smaller subset for Streamlit)
test_df_sample = test_df.sample(n=min(1000, len(test_df)), random_state=42)
test_df_sample.to_csv('test_data.csv', index=False)
print(f"✓ Test data saved: test_data.csv ({test_df_sample.shape[0]} samples)")

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print("\nNext Steps:")
print("1. All models saved in model/ folder")
print("2. Test data saved as test_data.csv")
print("3. Ready for Streamlit app deployment")
