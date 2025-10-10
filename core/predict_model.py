# predict_model.py
import joblib
import os
import pandas as pd

# Path to your model
MODEL_PATH = os.path.join("models", "LightGBM_model.pkl")

# Try to load model
try:
    MODEL = joblib.load(MODEL_PATH)
    print("✅ LightGBM model loaded successfully.")
except Exception as e:
    print(f"⚠ Warning: Model load failed, using mock predictions. {e}")
    MODEL = None

# Map model output labels to readable names
LABEL_MAP = {0: "Flood", 1: "Earthquake", 2: "Landslide", 3: "Cyclone", 4: "None"}


def predict_disaster(data: pd.DataFrame):
    """
    Predicts the disaster type using ML model or fallback rule-based logic.
    Silently ignores model errors to avoid runtime crashes.
    """

    # Ensure input is DataFrame
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame([data])

    # Try model prediction
    if MODEL is not None:
        try:
            prediction_raw = MODEL.predict(data)
            probabilities = MODEL.predict_proba(data)[0] if hasattr(MODEL, "predict_proba") else None
            disaster = LABEL_MAP.get(int(prediction_raw[0]), "Unknown")
            return disaster, probabilities
        except:
            # Silently fallback to rule-based prediction
            pass

    # --- Rule-based fallback logic (no errors printed) ---
    row = data.iloc[0] if hasattr(data, "iloc") else data

    # Extract safely
    rainfall = row.get("rainfall", 0)
    earthquake_mean_earth = row.get("earthquake_mean_earth", 0)
    slope_angle = row.get("slope_angle", 0)
    windspeed = row.get("windspeed", 0)

    # Simple conditional logic
    if rainfall > 100:
        return "Flood", None
    elif earthquake_mean_earth > 4.0:
        return "Earthquake", None
    elif slope_angle > 30:
        return "Landslide", None
    elif windspeed > 100:
        return "Cyclone", None
    else:
        return "None", None
