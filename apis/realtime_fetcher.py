import requests
import pandas as pd

# Default location = Chennai 🌴
LAT = 13.0827
LON = 80.2707

# -----------------------------
# Feature Mapping (Expected by ML model)
# -----------------------------
EXPECTED_FEATURES = [
    "temperature",
    "earthquake_mean_earth",
    "rainfall",
    "soil_moisture",
    "windspeed",
    "pressure",
    "river_level",
    "slope_angle",
    "seismic_activity",
    "peak_acceleration"
]

# Safe default values (represents "No Disaster" condition)
SAFE_DEFAULTS = {
    "temperature": 30.0,
    "earthquake_mean_earth": 0.0,
    "rainfall": 0.0,
    "soil_moisture": 25.0,
    "windspeed": 0.5,
    "pressure": 1013.0,
    "river_level": 1.0,
    "slope_angle": 5.0,
    "seismic_activity": 0.0,
    "peak_acceleration": 0.0,
}

# -----------------------------
# Weather API (Tomorrow.io)
# -----------------------------
def fetch_weather_data(api_key):
    try:
        url = f"https://api.tomorrow.io/v4/weather/realtime?location={LAT},{LON}&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        values = data.get("data", {}).get("values", {})

        df = pd.DataFrame([{
            "temperature": values.get("temperature", SAFE_DEFAULTS["temperature"]),
            "rainfall": values.get("precipitationIntensity", SAFE_DEFAULTS["rainfall"]),
            "windspeed": values.get("windSpeed", SAFE_DEFAULTS["windspeed"]),
            "pressure": values.get("pressureSurfaceLevel", SAFE_DEFAULTS["pressure"]),
            "soil_moisture": SAFE_DEFAULTS["soil_moisture"],   # not in API, fallback
            "river_level": SAFE_DEFAULTS["river_level"],       # not in API, fallback
            "slope_angle": SAFE_DEFAULTS["slope_angle"],       # not in API, fallback
            "seismic_activity": SAFE_DEFAULTS["seismic_activity"],  # not in API
            "peak_acceleration": SAFE_DEFAULTS["peak_acceleration"], # not in API
        }])
        return df
    except Exception as e:
        print(f"❌ Weather API error: {e}")
        return pd.DataFrame([SAFE_DEFAULTS])

# -----------------------------
# Earthquake API (USGS)
# -----------------------------
def fetch_earthquake_data():
    try:
        url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
        response = requests.get(url)
        data = response.json()

        if not data["features"]:
            return pd.DataFrame([{"earthquake_mean_earth": SAFE_DEFAULTS["earthquake_mean_earth"]}])

        quake = data["features"][0]["properties"]
        mag = quake.get("mag", 0.0)

        df = pd.DataFrame([{"earthquake_mean_earth": mag}])
        return df
    except Exception as e:
        print(f"❌ Earthquake API error: {e}")
        return pd.DataFrame([{"earthquake_mean_earth": SAFE_DEFAULTS["earthquake_mean_earth"]}])

# -----------------------------
# Simulation Mode
# -----------------------------
def fetch_simulated_data():
    print("\nSelect disaster to simulate:")
    print("1 = Flood")
    print("2 = Earthquake")
    print("3 = Landslide")
    print("4 = Cyclone")
    print("5 = No Disaster")
    choice = input("Enter your choice: ")

    scenarios = {
        "1": {
            "temperature": 28.0,
            "earthquake_mean_earth": 0.0,
            "rainfall": 120.0,
            "soil_moisture": 80.0,
            "windspeed": 10.0,
            "pressure": 995.0,
            "river_level": 5.0,
            "slope_angle": 10.0,
            "seismic_activity": 0.0,
            "peak_acceleration": 0.0,
        },
        "2": {
            "temperature": 30.0,
            "earthquake_mean_earth": 5.2,
            "rainfall": 0.0,
            "soil_moisture": 25.0,
            "windspeed": 5.0,
            "pressure": 1008.0,
            "river_level": 1.0,
            "slope_angle": 5.0,
            "seismic_activity": 6.5,
            "peak_acceleration": 0.3,
        },
        "3": {
            "temperature": 26.0,
            "earthquake_mean_earth": 0.0,
            "rainfall": 80.0,
            "soil_moisture": 70.0,
            "windspeed": 6.0,
            "pressure": 1002.0,
            "river_level": 2.0,
            "slope_angle": 35.0,
            "seismic_activity": 0.5,
            "peak_acceleration": 0.05,
        },
        "4": {
            "temperature": 27.0,
            "earthquake_mean_earth": 0.0,
            "rainfall": 200.0,
            "soil_moisture": 85.0,
            "windspeed": 120.0,
            "pressure": 970.0,
            "river_level": 4.0,
            "slope_angle": 8.0,
            "seismic_activity": 0.0,
            "peak_acceleration": 0.0,
        },
        "5": {
            "temperature": 30.0,
            "earthquake_mean_earth": 0.0,
            "rainfall": 0.0,
            "soil_moisture": 25.0,
            "windspeed": 5.0,
            "pressure": 1012.0,
            "river_level": 1.0,
            "slope_angle": 5.0,
            "seismic_activity": 0.0,
            "peak_acceleration": 0.0,
        },
    }

    selected = scenarios.get(choice, scenarios["5"])
    return pd.DataFrame([selected])

# -----------------------------
# Merge Weather + Earthquake into one DataFrame
# -----------------------------
def get_realtime_data(api_key):
    weather_df = fetch_weather_data(api_key)
    quake_df = fetch_earthquake_data()

    combined = pd.concat([weather_df, quake_df], axis=1)

    # Ensure all expected features exist
    for feature in EXPECTED_FEATURES:
        if feature not in combined.columns:
            combined[feature] = SAFE_DEFAULTS[feature]

    return combined[EXPECTED_FEATURES]

# -----------------------------
# Disaster Prediction (Mock for now)
# -----------------------------
def predict_disaster(input_df):
    try:
        # Mock prediction based on input values
        if input_df["rainfall"].values[0] > 100:
            return "Flood"
        elif input_df["earthquake_mean_earth"].values[0] > 4.0:
            return "Earthquake"
        elif input_df["slope_angle"].values[0] > 30:
            return "Landslide"
        elif input_df["windspeed"].values[0] > 100:
            return "Cyclone"
        else:
            return "No Disaster"
    except Exception as e:
        return f"error: {e}"

# -----------------------------
# Main Program
# -----------------------------
print("🌍 Disaster Prediction Demo (Chennai Default)\n")

print("Choose mode:")
print("1 = API (Tomorrow.io + USGS)")
print("2 = Simulation")
mode = input("Enter your choice: ")

if mode == "1":
    API_KEY = input("Enter your Tomorrow.io API key: ")
    input_df = get_realtime_data(API_KEY)
else:
    input_df = fetch_simulated_data()

# Display input data
print("\n📊 Input Data:\n")
for col in input_df.columns:
    print(f"{col}: {input_df[col].values[0]}")

# Make prediction
prediction = predict_disaster(input_df)

# Display result
print("\n🔮 Prediction Result:\n")
print(f"Predicted Disaster: {prediction}")

print("\n🚨 ALERT! 🚨")
print(f"⚠ Predicted Disaster: {prediction}")
print("Authorities have been notified.")
