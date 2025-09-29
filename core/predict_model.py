import joblib
import os
# Path to my model file
model_path = os.path.join("models", "LightGBM_model.pkl")
# Load the LightGBM model safely
try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"⚠ Warning: Model load failed, using mock predictions. {e}")
    model = None  # fallback to mock
def predict_disaster(data):
    """
    Predict disaster type using the trained LightGBM model.
    `data` is expected to be a Pandas DataFrame with correct features.
    Returns (disaster, probabilities)
    """
    try:
        # -----------------------------
        # SAFE CLAUSE: If values are normal → "None"
        # -----------------------------
        row = data.iloc[0].to_dict()
        if (
            row.get("rainfall", 0) < 20.0
            and row.get("windspeed", 0) < 15.0
            and 995 < row.get("pressure", 1010) < 1025
            and row.get("seismic_activity", 0) < 1.0
            and row.get("peak_acceleration", 0) < 0.1
            and row.get("river_level", 0) <= 2.0
            and row.get("soil_moisture", 25) < 50.0
        ):
            return "None", None
        # -----------------------------
        # Use ML model if loaded
        # -----------------------------
        if model is not None:
            prediction = model.predict(data)[0]
            probabilities = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(data)[0]
            label_map = {
                0: "Flood",
                1: "Earthquake",
                2: "Landslide",
                3: "Cyclone",
                4: "None"
            }
            disaster = label_map.get(int(prediction), "Unknown")
            return disaster, probabilities
        if row.get("rainfall", 0) > 100:
            return "Flood", None
        elif row.get("earthquake_mean_earth", 0) > 4.0:
            return "Earthquake", None
        elif row.get("slope_angle", 0) > 30:
            return "Landslide", None
        elif row.get("windspeed", 0) > 100:
            return "Cyclone", None
        else:
            return "None", None
    except Exception as e:
        return f"error: {e}", None
