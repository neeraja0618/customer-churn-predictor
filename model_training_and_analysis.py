"""
Customer Churn Prediction - Full Model Training & Analysis Pipeline
---------------------------------------------------------------------
This script covers the complete process: loading raw data, cleaning it,
exploring patterns, encoding features, training and comparing two models,
and saving the final chosen model for the Streamlit app.

Run this if you want to reproduce the model from scratch:
    python model_training_and_analysis.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------------------------------------------------
# STEP 1: Load raw data
# -----------------------------------------------------------------------
df = pd.read_csv('Telco-Customer-Churn.csv')
print(f"Loaded {len(df)} customer records with {df.shape[1]} columns.")
print(f"Churn rate: {100*(df['Churn']=='Yes').mean():.1f}%")

# -----------------------------------------------------------------------
# STEP 2: Clean data
# -----------------------------------------------------------------------
# TotalCharges has 11 blank values, all belonging to brand-new customers
# (tenure = 0). These customers haven't been billed yet, so we fill with 0
# rather than dropping real customer records.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# -----------------------------------------------------------------------
# STEP 3: Exploratory analysis (key patterns found)
# -----------------------------------------------------------------------
print("\n--- Churn rate by Contract type ---")
print(df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).round(1))

print("\n--- Average tenure: churned vs stayed ---")
print(df.groupby('Churn')['tenure'].mean().round(1))

print("\n--- Churn rate by Internet Service ---")
print(df.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).round(1))

# -----------------------------------------------------------------------
# STEP 4: Encode categorical features
# -----------------------------------------------------------------------
df = df.drop(columns=['customerID'])  # ID column carries no predictive signal

binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map(lambda x: 1 if x in ['Yes', 'Male'] else 0)

multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
              'Contract', 'PaymentMethod']
df_encoded = pd.get_dummies(df, columns=multi_cols)

# -----------------------------------------------------------------------
# STEP 5: Train/test split
# -----------------------------------------------------------------------
X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------------
# STEP 6: Train and compare two models
# -----------------------------------------------------------------------
print("\n=== Model 1: Logistic Regression (class_weight=balanced) ===")
log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, log_preds)*100:.1f}%")
print(classification_report(y_test, log_preds, target_names=['No Churn', 'Churn']))

print("\n=== Model 2: Random Forest (class_weight=balanced) ===")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, rf_preds)*100:.1f}%")
print(classification_report(y_test, rf_preds, target_names=['No Churn', 'Churn']))

# -----------------------------------------------------------------------
# STEP 7: Model selection
# -----------------------------------------------------------------------
# Logistic Regression is chosen as the final model despite slightly lower
# accuracy than Random Forest, because it achieves higher RECALL on the
# Churn class (79% vs 70%). In this business problem, failing to catch a
# real churner is more costly than a false alarm, so recall is prioritized
# over raw accuracy. Logistic Regression's coefficients are also directly
# interpretable, which powers the "why" explanation in the app.
final_model = log_model
print("\nFinal model selected: Logistic Regression (prioritizing churn recall)")

# -----------------------------------------------------------------------
# STEP 8: Save model artifacts for the Streamlit app
# -----------------------------------------------------------------------
joblib.dump(final_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(list(X_train.columns), 'columns.pkl')
print("\nSaved model.pkl, scaler.pkl, columns.pkl")
