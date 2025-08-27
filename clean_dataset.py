#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# -------------------------------
# Config
# -------------------------------
SAVE_PATH = r"C:\Users\Lenovo\DisasterPredictionProject\data\processed\combined_disaster_data.csv"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# -------------------------------
# Generate 30 days of dates
# -------------------------------
dates = [datetime.today().date() - timedelta(days=i) for i in range(30)]
dates.reverse()  # oldest first

# -------------------------------
# Generate synthetic + earthquake-like data
# -------------------------------
np.random.seed(42)

data = pd.DataFrame({
    "date": dates,
    "earthquake_event_count": np.random.poisson(2, 30),   # # of quakes/day
    "mean_earthquake_magnitude": np.random.uniform(3.0, 6.5, 30),  # Richter scale
    "rainfall": np.random.uniform(0, 200, 30),            # mm
    "soil_moisture": np.random.uniform(10, 90, 30),       # %
    "windspeed": np.random.uniform(5, 200, 30),           # km/h
    "pressure": np.random.uniform(950, 1025, 30),         # hPa
    "river_level": np.random.uniform(0, 10, 30),          # m
    "slope_angle": np.random.uniform(0, 45, 30)           # degrees
})

# Seismic activity proxy
data["seismic_activity_index"] = np.clip(
    (data["earthquake_event_count"] / 10) * (data["mean_earthquake_magnitude"] / 5),
    0, 1
)
data["peak_accel_proxy"] = np.clip(data["mean_earthquake_magnitude"] / 3, 0, 2)

# -------------------------------
# Labels (rules)
# -------------------------------
data["is_earthquake"] = ((data["seismic_activity_index"] > 0.2) | (data["mean_earthquake_magnitude"] >= 5)).astype(int)
data["is_flood"] = ((data["rainfall"] >= 150) | (data["river_level"] > 5)).astype(int)
data["is_landslide"] = ((data["slope_angle"] >= 20) & (data["rainfall"] > 100) & (data["soil_moisture"] > 65)).astype(int)
data["is_cyclone"] = ((data["windspeed"] >= 120) & (data["pressure"] < 990)).astype(int)

# Disaster type with priority
def assign_disaster(row):
    if row["is_earthquake"]:
        return "earthquake"
    elif row["is_cyclone"]:
        return "cyclone"
    elif row["is_flood"]:
        return "flood"
    elif row["is_landslide"]:
        return "landslide"
    else:
        return "none"

data["disaster_type"] = data.apply(assign_disaster, axis=1)

# -------------------------------
# Save CSV
# -------------------------------
data.to_csv(SAVE_PATH, index=False)
print(f"✅ Dataset saved to {SAVE_PATH}")
print(data.head())
