# apis/realtime_fetcher.py
import requests
import datetime
from core.predict_model import predict_disaster

# ---------------------------
# API KEYS (set yours here)
# ---------------------------
TOMORROW_API_KEY = "YOUR_TOMORROW_IO_API_KEY"
USGS_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson"

# ---------------------------
# Fetch Tomorrow.io Weather Data
# ---------------------------
def fetch_weather(lat=20.0, lon=77.0):
    url = f"https://api.tomorrow.io/v4/timelines"
    params = {
        "location": f"{lat},{lon}",
        "fields": ["temperature", "humidity", "pressureSeaLevel", "windSpeed", "precipitationIntensity"],
        "timesteps": "current",
        "apikey": TOMORROW_API_KEY,
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    values = data["data"]["timelines"][0]["intervals"][0]["values"]

    return {
        "atmospheric_pressure": values.get("pressureSeaLevel", 1013),
        "humidity": values.get("humidity", 60),
        "rainfall_mm": values.get("precipitationIntensity", 0) * 60,  # per hour
        "wind_shear": values.get("windSpeed", 5),
        "sea_surface_temperature": values.get("temperature", 26),
    }

# ---------------------------
# Fetch USGS Earthquake Data
# ---------------------------
def fetch_earthquake():
    resp = requests.get(USGS_URL)
    resp.raise_for_status()
    data = resp.json()

    if not data["features"]:
        return {"magnitude": 0, "depth": 0, "latitude": 0, "longitude": 0}

    latest_eq = data["features"][0]
    coords = latest_eq["geometry"]["coordinates"]

    return {
        "magnitude": latest_eq["properties"]["mag"] or 0,
        "depth": coords[2] if len(coords) > 2 else 0,
        "longitude": coords[0],
        "latitude": coords[1],
    }

# ---------------------------
# Merge into ML Input Format
# ---------------------------
def build_feature_dict(weather, earthquake):
    features = {
        # Weather-based
        "atmospheric_pressure": weather.get("atmospheric_pressure", 1013),
        "humidity": weather.get("humidity", 60),
        "rainfall_mm": weather.get("rainfall_mm", 0),
        "wind_shear": weather.get("wind_shear", 5),
        "sea_surface_temperature": weather.get("sea_surface_temperature", 26),

        # Earthquake-based
        "magnitude": earthquake.get("magnitude", 0),
        "depth": earthquake.get("depth", 0),
        "latitude": earthquake.get("latitude", 0),
        "longitude": earthquake.get("longitude", 0),

        # Safe defaults for dataset-required fields
        "soil_saturation": 0.5,
        "urbanization": 5,
        "drainagesystems": 3,
        "deterioratinginfrastructure": 0,
        "deforestation": 0,
        "vegetation_cover": 5,
        "slope_angle": 15,
        "ocean_depth": 4000,
        "vorticity": -1,
        "coastalvulnerability": 0,
    }

    return features

# ---------------------------
# Main Function
# ---------------------------
def fetch_and_predict():
    print("📡 Fetching weather data from Tomorrow.io...")
    weather = fetch_weather()

    print("🌎 Fetching earthquake data from USGS...")
    earthquake = fetch_earthquake()

    print("\n🔄 Building feature vector...")
    features = build_feature_dict(weather, earthquake)

    print("\n📊 Running ML prediction...")
    results = predict_disaster(features)

    print("\n🌍 Realtime Disaster Risk Results:")
    for hazard, risk in results.items():
        print(f"- {hazard}: {'⚠️ RISK' if risk == 1 else '✅ Safe'}")
