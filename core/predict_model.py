#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# core/predict_model.py
import joblib
import numpy as np
import os

# Load the trained model
MODEL_PATH = os.path.join("models", "xgb_disaster_model.joblib")

try:
    model = joblib.load(MODEL_PATH)
    print("[✅] Model loaded successfully.")
except FileNotFoundError:
    print("[❌] Trained model not found at:", MODEL_PATH)
    model = None

# Define the list of features (must match training)
FEATURES = [
    'humidity', 'rainfall_mm', 'soil_saturation', 'vegetation_cover',
    'earthquake_activity', 'proximity_to_water', 'soil_type_gravel',
    'soil_type_sand', 'soil_type_silt', 'sea_surface_temperature',
    'atmospheric_pressure', 'wind_shear', 'vorticity', 'latitude',
    'ocean_depth', 'proximity_to_coastline', 'pre_existing_disturbance',
    'slope_angle', 'longitude', 'depth', 'magnitude'
]

def predict_disaster(input_data: dict):
    if not model:
        return "Model not available."

    try:
        # Fill missing values with 0
        input_features = [input_data.get(feat, 0) for feat in FEATURES]
        input_array = np.array(input_features).reshape(1, -1)

        prediction = model.predict(input_array)[0]
        return prediction

    except Exception as e:
        print("[❌] Prediction failed:", str(e))
        return "Error"

