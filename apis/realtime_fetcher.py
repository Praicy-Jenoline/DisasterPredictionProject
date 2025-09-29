import requests
import pandas as pd

LAT = 13.0827
LON = 80.2707

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

def fetch_weather_data(api_key):
    try:
        url = f"https://api.tomorrow.io/v4/weather/realtime?location={LAT},{LON}&apikey={api_key}"
        resp = requests.get(url).json()
        values = resp.get("data", {}).get("values", {})
        return pd.DataFrame([{
            "temperature": values.get("temperature", SAFE_DEFAULTS["temperature"]),
            "rainfall": values.get("precipitationIntensity", SAFE_DEFAULTS["rainfall"]),
            "windspeed": values.get("windSpeed", SAFE_DEFAULTS["windspeed"]),
            "pressure": values.get("pressureSurfaceLevel", SAFE_DEFAULTS["pressure"]),
            "soil_moisture": SAFE_DEFAULTS["soil_moisture"],
            "river_level": SAFE_DEFAULTS["river_level"],
            "slope_angle": SAFE_DEFAULTS["slope_angle"],
            "seismic_activity": SAFE_DEFAULTS["seismic_activity"],
            "peak_acceleration": SAFE_DEFAULTS["peak_acceleration"],
        }])
    except Exception as e:
        print(f"❌ Weather API error: {e}")
        return pd.DataFrame([SAFE_DEFAULTS])

def fetch_earthquake_data():
    try:
        url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
        resp = requests.get(url).json()
        if not resp["features"]:
            return pd.DataFrame([{"earthquake_mean_earth": SAFE_DEFAULTS["earthquake_mean_earth"]}])
        quake = resp["features"][0]["properties"]
        return pd.DataFrame([{"earthquake_mean_earth": quake.get("mag", 0.0)}])
    except Exception as e:
        print(f"❌ Earthquake API error: {e}")
        return pd.DataFrame([{"earthquake_mean_earth": SAFE_DEFAULTS["earthquake_mean_earth"]}])

def get_realtime_data(api_key):
    weather_df = fetch_weather_data(api_key)
    quake_df = fetch_earthquake_data()
    df = pd.concat([weather_df, quake_df], axis=1)
    return df

def fetch_simulated_data():
    print("\nSelect disaster to simulate:")
    print("1 = Flood\n2 = Earthquake\n3 = Landslide\n4 = Cyclone\n5 = No Disaster")
    choice = input("Enter your choice: ")
    scenarios = {
        "1": {"temperature":28,"earthquake_mean_earth":0,"rainfall":120,"soil_moisture":80,"windspeed":10,"pressure":995,"river_level":5,"slope_angle":10,"seismic_activity":0,"peak_acceleration":0},
        "2": {"temperature":30,"earthquake_mean_earth":5.2,"rainfall":0,"soil_moisture":25,"windspeed":5,"pressure":1008,"river_level":1,"slope_angle":5,"seismic_activity":6.5,"peak_acceleration":0.3},
        "3": {"temperature":26,"earthquake_mean_earth":0,"rainfall":80,"soil_moisture":70,"windspeed":6,"pressure":1002,"river_level":2,"slope_angle":35,"seismic_activity":0.5,"peak_acceleration":0.05},
        "4": {"temperature":27,"earthquake_mean_earth":0,"rainfall":200,"soil_moisture":85,"windspeed":120,"pressure":970,"river_level":4,"slope_angle":8,"seismic_activity":0,"peak_acceleration":0},
        "5": SAFE_DEFAULTS
    }
    return pd.DataFrame([scenarios.get(choice, SAFE_DEFAULTS)])
