import requests
import pandas as pd
import numpy as np
import geocoder

# ---------------------------
# API Keys
# ---------------------------
OPENWEATHER_API_KEY = "uApb4FYB4vxEiYC0AiEV1A2onC9lXHUU"

# ---------------------------
# Geolocation
# ---------------------------
def get_current_location():
    try:
        g = geocoder.ip("me")
        if g.ok:
            return g.latlng
    except Exception as e:
        print(f"[❌] Geolocation error: {e}")
    return [13.0878, 80.2785]  # fallback to Chennai


# ---------------------------
# Fetch Earthquake Data
# ---------------------------
def fetch_earthquake_data():
    try:
        url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson"
        res = requests.get(url, timeout=10)
        data = res.json()

        if "features" in data and len(data["features"]) > 0:
            eq = data["features"][0]  # latest quake
            coords = eq["geometry"]["coordinates"]
            props = eq["properties"]

            return {
                "magnitude": round(props.get("mag", 0) or 0, 2),
                "depth": round(coords[2], 2) if len(coords) > 2 else 0,
                "lat": round(coords[1], 6),
                "lon": round(coords[0], 6),
            }
    except Exception as e:
        print(f"[❌] Earthquake fetch error: {e}")

    return {"magnitude": 0, "depth": 0, "lat": 0, "lon": 0}


# ---------------------------
# Fetch Weather Data
# ---------------------------
def fetch_weather_data(lat, lon):
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/onecall?"
            f"lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        )
        res = requests.get(url, timeout=10)
        data = res.json()

        current = data.get("current", {})
        return {
            "temp": round(current.get("temp", 0), 2),
            "humidity": current.get("humidity", 0),
            "pressure": current.get("pressure", 0),
            "wind_speed": current.get("wind_speed", 0),
            "weather_id": current.get("weather", [{}])[0].get("id", 0),
        }
    except Exception as e:
        print(f"[❌] Weather fetch error: {e}")
        return {"temp": 0, "humidity": 0, "pressure": 0, "wind_speed": 0, "weather_id": 0}


# ---------------------------
# Feature Engineering
# ---------------------------
def get_features_for_prediction():
    lat, lon = get_current_location()
    eq_data = fetch_earthquake_data()
    weather_data = fetch_weather_data(lat, lon)

    features_dict = {
        "atmospheric_disaster_typ": weather_data["weather_id"],
        "urbanization": 1,
        "proximity_t_wetlandloss": 1.49,
        "magnitude": eq_data["magnitude"],
        "depth": eq_data["depth"],
        "temp": weather_data["temp"],
        "humidity": weather_data["humidity"],
        "pressure": weather_data["pressure"],
        "wind_speed": weather_data["wind_speed"],
        "lat": lat,
        "lon": lon,
        "climatechange": 1,
        "deforestation": 1,
        "landslides": 0.85,
        # add dummy placeholders for rest of your 21 features
        "feature_13": 0,
        "feature_14": 0,
        "feature_15": 0,
        "feature_16": 0,
        "feature_17": 0,
        "feature_18": 0,
        "feature_19": 0,
        "feature_20": 0,
        "feature_21": 0,
    }

    features_df = pd.DataFrame([features_dict])

    # 🔹 Flatten nested lists/arrays (convert [801] → 801)
    for col in features_df.columns:
        features_df[col] = features_df[col].apply(
            lambda x: x[0] if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0 else x
        )

    # 🔹 Force numeric (convert strings → numbers, fill NaN with 0)
    features_df = features_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    return features_df
