#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# simulate.py

"""
Simulate predefined disaster scenarios using the trained ML model.
"""
import sys
from core.predict_model import predict_disaster

# ---------------------------
# Predefined simulation scenarios
# ---------------------------
SIMULATIONS = {
    "Flood": {
        "atmospheric_pressure": 995,
        "humidity": 90,
        "rainfall_mm": 200,
        "soil_saturation": 0.95,
        "urbanization": 7,
        "drainagesystems": 2,
        "latitude": 23,
        "longitude": 90,
    },
    "Earthquake": {
        "magnitude": 6.5,
        "depth": 15,
        "soil_type_sand": 1,
        "deterioratinginfrastructure": 1,
        "urbanization": 5,
        "latitude": 35,
        "longitude": 78,
    },
    "Landslide": {
        "slope_angle": 35,
        "rainfall_mm": 150,
        "soil_saturation": 0.8,
        "deforestation": 1,
        "landslide": 1,
        "vegetation_cover": 2,
        "urbanization": 6,
    },
    "Cyclone": {
        "sea_surface_temperature": 29,
        "wind_shear": 12,
        "vorticity": -3,
        "humidity": 85,
        "ocean_depth": 4000,
        "coastalvulnerability": 1,
        "latitude": 18,
        "longitude": 88,
    },
}

# ---------------------------
# Simulation runner
# ---------------------------
def run_simulation():
    print("\n🌍 Disaster Simulation Mode")
    print("Choose a disaster to simulate:")
    for i, disaster in enumerate(SIMULATIONS.keys(), 1):
        print(f"{i}. {disaster}")

    try:
        choice = int(input("\nEnter choice (1-4): ").strip())
        disaster = list(SIMULATIONS.keys())[choice - 1]
    except (ValueError, IndexError):
        print("❌ Invalid choice. Exiting simulation.")
        sys.exit(1)

    print(f"\n🔬 Running simulation for: {disaster}...\n")

    # Run prediction with predefined features
    results = predict_disaster(SIMULATIONS[disaster])

    # Print results
    print("📊 Prediction Results:")
    for hazard, risk in results.items():
        print(f"- {hazard}: {'⚠️ RISK' if risk == 1 else '✅ Safe'}")


if __name__ == "__main__":
    run_simulation()
