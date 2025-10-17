#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
from core.predict_model import predict_disaster
from apis.realtime_fetcher import get_realtime_data, SAFE_DEFAULTS
import os

app = Flask(__name__)
CORS(app)  # allow frontend to access backend

# Serve index.html from root folder
@app.route('/')
def home():
    return send_from_directory(os.getcwd(), 'index.html')

# Helper: Simulated data
def fetch_simulated_data_choice(choice):
    scenarios = {
        "1": {"temperature": 28.0, "earthquake_mean_earth": 0.0, "rainfall": 120.0, "soil_moisture": 80.0,
              "windspeed": 10.0, "pressure": 995.0, "river_level": 5.0, "slope_angle": 10.0,
              "seismic_activity": 0.0, "peak_acceleration": 0.0},
        "2": {"temperature": 30.0, "earthquake_mean_earth": 5.2, "rainfall": 0.0, "soil_moisture": 25.0,
              "windspeed": 5.0, "pressure": 1008.0, "river_level": 1.0, "slope_angle": 5.0,
              "seismic_activity": 6.5, "peak_acceleration": 0.3},
        "3": {"temperature": 26.0, "earthquake_mean_earth": 0.0, "rainfall": 80.0, "soil_moisture": 70.0,
              "windspeed": 6.0, "pressure": 1002.0, "river_level": 2.0, "slope_angle": 35.0,
              "seismic_activity": 0.5, "peak_acceleration": 0.05},
        "4": {"temperature": 27.0, "earthquake_mean_earth": 0.0, "rainfall": 200.0, "soil_moisture": 85.0,
              "windspeed": 120.0, "pressure": 970.0, "river_level": 4.0, "slope_angle": 8.0,
              "seismic_activity": 0.0, "peak_acceleration": 0.0},
        "5": SAFE_DEFAULTS
    }
    return pd.DataFrame([scenarios.get(choice, SAFE_DEFAULTS)])

@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.get_json()
    disaster_type = data.get("disaster", "5")
    df = fetch_simulated_data_choice(disaster_type)
    prediction, _ = predict_disaster(df)
    return jsonify({
        "mode": "simulation",
        "predicted_disaster": prediction,
        "input_data": df.to_dict(orient='records')[0]
    })

@app.route('/realtime', methods=['POST'])
def realtime():
    data = request.get_json()
    api_key = data.get("api_key")
    df = get_realtime_data(api_key)
    prediction, _ = predict_disaster(df)
    return jsonify({
        "mode": "realtime",
        "predicted_disaster": prediction,
        "input_data": df.to_dict(orient='records')[0]
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
