#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# core/predict_model.py

"""
Prediction module. Loads trained ML model and predicts on features.
"""
import joblib
import numpy as np
import pandas as pd
import os

# Load model and metadata
MODEL_PATH = "models/multi_disaster_model.pkl"
FEATURES_PATH = "models/feature_names.pkl"
LABELS_PATH = "models/label_names.pkl"

if not (os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH) and os.path.exists(LABELS_PATH)):
    raise FileNotFoundError("❌ Trained model or metadata not found. Run train_model.py first.")

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)
label_names = joblib.load(LABELS_PATH)


def predict_disaster(input_data: dict):
    """
    Predicts multiple disasters from given input data.

    Parameters:
        input_data (dict): Dictionary of feature_name -> value

    Returns:
        dict: Predicted labels with 0/1 risk classification
    """

    # Ensure all required features exist, fill missing with 0
    features = []
    for col in feature_names:
        val = input_data.get(col, 0)  # default = 0 if not provided
        features.append(val)

    # Convert to dataframe (model expects 2D input)
    X = pd.DataFrame([features], columns=feature_names)

    # Predict
    y_pred = model.predict(X)[0]

    # Map predictions to labels
    results = {label: int(pred) for label, pred in zip(label_names, y_pred)}

    return results


# ---------------------------
# Example usage (manual test)
# ---------------------------
if __name__ == "__main__":
    # Example dummy input (replace with realtime_fetcher / simulate)
    sample_input = {
        "atmospheric_pressure": 1000,
        "humidity": 80,
        "rainfall_mm": 120,
        "magnitude": 6.2,
        "depth": 30,
        "sea_surface_temperature": 28,
        "wind_shear": 15,
        "vorticity": -2,
        "soil_saturation": 0.9,
        "urbanization": 6,
    }

    results = predict_disaster(sample_input)
    print("\n🌍 Disaster Risk Prediction:")
    for hazard, risk in results.items():
        print(f"- {hazard}: {'⚠️ RISK' if risk == 1 else '✅ Safe'}")
