from apis.realtime_fetcher import get_features_for_prediction
from core.predict_model import predict_disaster
from notifications.notifier import send_alert
import time

def main():
    print("🌍 Real-Time Disaster Monitoring Started")
    print("Press Ctrl + C to stop.\n")

    while True:
        try:
            # Step 1: Fetch combined features (uses geolocation + weather + USGS + defaults)
            features = get_features_for_prediction()

            if features is None or features.empty:
                print("[⚠️] No features generated. Skipping prediction.")
                time.sleep(3600)
                continue

            # Step 2: Predict using the ML model
            prediction = predict_disaster(features)

            # Step 3: Notify if needed
            if prediction != "NoDisaster":
                message = f"⚠️ ALERT: Possible {prediction.upper()} risk detected!"
                print(message)
                send_alert(message)
            else:
                print("✅ No disaster detected.\n")

        except Exception as e:
            print("[❌] An error occurred during monitoring:", str(e))

        print("⏳ Waiting for next cycle...\n")
        time.sleep(3600)  # Run once every hour

if __name__ == "__main__":
    main()
