#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# core/train_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import joblib
import os

# === Paths ===
data_path = "data/processed/combined_disaster_data.csv"
model_path = "models/xgb_disaster_model.joblib"
encoder_path = "models/label_encoder.joblib"

# === Load Dataset ===
data = pd.read_csv(data_path)

# === DEBUG: View columns and NaNs ===
print("[DEBUG] Loaded dataset with shape:", data.shape)
print("[DEBUG] Number of NaNs per column:\n", data.isnull().sum())

# === Drop rows without target ===
data.dropna(subset=["disaster_type"], inplace=True)

# === Encode target label ===
label_encoder = LabelEncoder()
data["label"] = label_encoder.fit_transform(data["disaster_type"])

# === Save encoder ===
joblib.dump(label_encoder, encoder_path)

# === Feature / Target Split ===
X = data.drop(["disaster_type", "label"], axis=1)
y = data["label"]

# === Impute Missing Values ===
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# === Balance Dataset using SMOTE ===
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_imputed, y)

# === Train XGBoost Model ===
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_balanced, y_balanced)

# === Save the Model ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, model_path)

# === Evaluate ===
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

