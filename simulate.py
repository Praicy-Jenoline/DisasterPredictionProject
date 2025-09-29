#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# simulate.py

"""
Simulate predefined disaster scenarios using the trained ML model.
"""
import pandas as pd
from core.predict_model import predict_disaster
import joblib
import os

def simulate():
    model_path = os.path.join("models", "trained_model.pkl")
    model = joblib.load(model_path)

    simulated_cases = [
        {"temperature": 26, "humidity": 90, "pressure": 995, "wind_speed": 3, "rainfall": 50, "magnitude": 0, "depth": 0},  # Flood
        {"temperature": 28, "humidity": 60, "pressure": 1005, "wind_speed": 2, "rainfall": 0, "magnitude": 6.2, "depth": 10}, # Earthquake
        {"temperature": 22, "humidity": 85, "pressure": 1000, "wind_speed": 5, "rainfall": 80, "magnitude": 0, "depth": 0},  # Landslide
        {"temperature": 29, "humidity": 88, "pressure": 980, "wind_speed": 120, "rainfall": 100, "magnitude": 0, "depth": 0} # Cyclone
    ]

    print("\nSelect disaster to simulate:")
    print("1 = Flood\n2 = Earthquake\n3 = Landslide\n4 = Cyclone")
    choice = input("Enter your choice: ")

    try:
        case = simulated_cases[int(choice)-1]
    except:
        print("Invalid choice, defaulting to Flood case.")
        case = simulated_cases[0]

    df = pd.DataFrame([case])

    print("\n📊 Simulated Input Data:\n")
    for col in df.columns:
        print(f"{col}: {df[col].values[0]}")

    result = predict_disaster(model, df)

    print("\n🔮 Prediction Result:\n")
    for key, value in result.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    simulate()
