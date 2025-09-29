# predict_model.py
import joblib
import os
import pandas as pd

MODEL_PATH = os.path.join("models", "LightGBM_model.pkl")

try:
    MODEL = joblib.load(MODEL_PATH)
    print("✅ LightGBM model loaded successfully.")
except Exception as e:
    print(f"⚠ Warning: Model load failed, using mock predictions. {e}")
    MODEL = None

LABEL_MAP = {0: "Flood", 1: "Earthquake", 2: "Landslide", 3: "Cyclone", 4: "None"}

def predict_disaster(data: pd.DataFrame):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a Pandas DataFrame")

    # -----------------------------
    # Use ML model if available
    # -----------------------------
    if MODEL is not None:
        try:
            prediction_raw = MODEL.predict(data)
            probabilities = MODEL.predict_proba(data)[0] if hasattr(MODEL, "predict_proba") else None
            disaster = LABEL_MAP.get(int(prediction_raw[0]), "Unknown")
            return disaster, probabilities
        except Exception as e:
            print(f"⚠ Prediction failed, using mock rules: {e}")

    # -----------------------------
    # Fallback mock rules
    # -----------------------------
    row = data.iloc[0]
    if row["rainfall"] > 100:
        return "Flood", None
    elif row["earthquake_mean_earth"] > 4.0:
        return "Earthquake", None
    elif row["slope_angle"] > 30:
        return "Landslide", None
    elif row["windspeed"] > 100:
        return "Cyclone", None
    else:
        return "None", None
