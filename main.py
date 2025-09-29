# main.py
from apis.realtime_fetcher import get_realtime_data, fetch_simulated_data
from core.predict_model import predict_disaster
from notifications.notifier import send_alert
import os
def main():
    print("\n🌍 Disaster Prediction Demo (Chennai Default)\n")
    print("Choose mode:")
    print("1 = API (Tomorrow.io + USGS)")
    print("2 = Simulation")
    choice = input("Enter your choice: ")

    if choice == "1":
        api_key = input("Enter Tomorrow.io API Key: ")
        data = get_realtime_data(api_key)
    else:
        data = fetch_simulated_data()

    print("\n📊 Input Data:\n")
    for k, v in data.iloc[0].items():
        print(f"{k}: {v}")

    prediction, probabilities = predict_disaster(data)

    print("\n🔮 Prediction Result:\n")
    print(f"Predicted Disaster: {prediction}")
    if probabilities is not None:
        print(f"Probability: {probabilities}")

    # ✅ Fix: wrap prediction in dict for notifier
    if prediction.lower() != "none":
        print("\n🚨 ALERT! 🚨")
        send_alert({"Predicted Disaster": prediction})


if __name__ == "__main__":
    main()
