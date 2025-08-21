#!/usr/bin/env python
# coding: utf-8

# In[ ]:
# train model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/processed_dataset.csv")

# Drop empty index column if present
if df.columns[0] == "Unnamed: 0" or df[df.columns[0]].isnull().all():
    df = df.drop(df.columns[0], axis=1)

# ---------------------------
# Define target labels
# ---------------------------

# Flood (binary from disaster_type)
df["Flood"] = df["disaster_type"].apply(lambda x: 1 if str(x).strip().lower() == "flood" else 0)

# Earthquake (1 if magnitude ≥ 5)
df["Earthquake"] = df["magnitude"].apply(lambda x: 1 if x >= 5 else 0)

# Landslide (use whichever column has higher value)
if "landslide" in df.columns and "landslides" in df.columns:
    df["Landslide"] = df[["landslide", "landslides"]].max(axis=1)
elif "landslide" in df.columns:
    df["Landslide"] = df["landslide"]
elif "landslides" in df.columns:
    df["Landslide"] = df["landslides"]
else:
    raise ValueError("No landslide column found in dataset.")

# Cyclone (heuristic: warm SST + low shear)
df["Cyclone"] = df.apply(
    lambda row: 1 if (row["sea_surface_temperature"] > 26 and row["wind_shear"] < 20) else 0,
    axis=1
)

# ---------------------------
# Features and targets
# ---------------------------
Y = df[["Flood", "Earthquake", "Landslide", "Cyclone"]]

X = df.drop(
    ["disaster_type", "Flood", "Earthquake", "Landslide", "Cyclone"],
    axis=1,
    errors="ignore"
)

# ---------------------------
# Train-test split
# ---------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ---------------------------
# Train model
# ---------------------------
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
model.fit(X_train, Y_train)

# ---------------------------
# Evaluate
# ---------------------------
Y_pred = model.predict(X_test)
print("\nClassification Report (per disaster):\n")
print(classification_report(Y_test, Y_pred, target_names=Y.columns))

# ---------------------------
# Save model + metadata
# ---------------------------
joblib.dump(model, "models/multi_disaster_model.pkl")
joblib.dump(X.columns.tolist(), "models/feature_names.pkl")
joblib.dump(Y.columns.tolist(), "models/label_names.pkl")

print("\n✅ Model training complete. Saved to 'models/' directory.")
