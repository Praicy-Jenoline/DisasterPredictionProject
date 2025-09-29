# main.py
# main.py

from notifications.notifier import send_alert
from apis.realtime_fetcher import get_realtime_data, fetch_simulated_data
from core.predict_model import predict_disaster
import pandas as pd

def display_input_data(df: pd.DataFrame):
    print("\n📊 Input Data (Simulation/Realtime):\n")
    for col in df.columns:
        print(f"{col}: {df[col].values[0]}")

def main():
    print("🌍 Disaster Prediction Demo (Chennai Default)\n")
    print("Choose mode:\n1 = Real-Time API\n2 = Simulation Mode\n3 = Exit")
    mode = input("Enter your choice (1/2/3): ").strip()
    if mode == "3":
        print("Exiting program.")
        return

    # ------------------ Fetch data ------------------
    if mode == "2":
        data_df = fetch_simulated_data()
    elif mode == "1":
        api_key = input("Enter your Tomorrow.io API key: ").strip()
        data_df = get_realtime_data(api_key)
    else:
        print("Invalid choice. Exiting.")
        return

    # Display the input data
    display_input_data(data_df)

    # ------------------ Predict disaster ------------------
    disaster, probabilities = predict_disaster(data_df)

    print("\n🔮 Prediction Result:\n")
    print(f"Predicted Disaster: {disaster}")
    if probabilities is not None:
        print(f"Probabilities: {probabilities}")

    # ------------------ Send alert ------------------
    if disaster not in ["None", "No Disaster", "Unknown"]:
        send_alert("🚨 ALERT! 🚨", f"Predicted Disaster: {disaster}")
    else:
        print("No significant disaster predicted. No alert sent.")

if __name__ == "__main__":
    main()
