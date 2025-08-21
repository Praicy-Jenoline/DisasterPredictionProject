#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# simulate.py
from core.predict_model import predict_disaster
from notifications.notifier import send_alert

# Sample simulated input for each disaster type
SIMULATED_INPUTS = {
    "Flood": {
        "rainfall_mm": 230,
        "soil_saturation": 0.85,
        "proximity_to_water": 0.2,
        "humidity": 90,
        "vegetation_cover": 0.3,
        "monsoonintensity": 0.9,
        "urbanization": 0.7
    },
    "Cyclone": {
        "sea_surface_temperature": 30,
        "atmospheric_pressure": 990,
        "wind_shear": 5,
        "humidity": 88,
        "vorticity": 0.7,
        "latitude": 15,
        "ocean_depth": 4000
    },
    "Earthquake": {
        "magnitude": 6.8,
        "depth": 15,
        "earthquake_activity": 1,
        "latitude": 26.5,
        "longitude": 87.4
    },
    "Landslide": {
        "rainfall_mm": 200,
        "slope_angle": 45,
        "soil_saturation": 0.9,
        "vegetation_cover": 0.2,
        "earthquake_activity": 1
    }
}

def simulate_disaster(disaster_type):
    if disaster_type not in SIMULATED_INPUTS:
        print("❌ Invalid disaster type. Choose from:", list(SIMULATED_INPUTS.keys()))
        return

    input_features = SIMULATED_INPUTS[disaster_type]
    print(f"\n🧪 Simulating disaster: {disaster_type}")
    predicted_disaster = predict_disaster(input_features)

    if predicted_disaster != "NoDisaster":
        message = f"🚨 Simulation: {predicted_disaster} would trigger an alert!"
        print(message)
        send_alert(message)
    else:
        print("✅ Simulation did not trigger any alert.")

if __name__ == "__main__":
    print("\nSelect a disaster to simulate:")
    for i, key in enumerate(SIMULATED_INPUTS.keys()):
        print(f"{i + 1}. {key}")

    choice = int(input("\nEnter choice (1-4): "))
    selected = list(SIMULATED_INPUTS.keys())[choice - 1]
    simulate_disaster(selected)

