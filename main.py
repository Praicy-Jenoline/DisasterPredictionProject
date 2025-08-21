# main.py
import sys
from apis.realtime_fetcher import fetch_and_predict
from simulate import run_simulation

def main():
    print("\n🌍 Disaster Prediction System")
    print("Choose an option:")
    print("1. Check Realtime Disaster Risk (APIs)")
    print("2. Run a Disaster Simulation")

    try:
        choice = int(input("\nEnter choice (1-2): ").strip())
    except ValueError:
        print("❌ Invalid input. Exiting.")
        sys.exit(1)

    try:
        if choice == 1:
            print("\n📡 Fetching realtime data and predicting risk...\n")
            fetch_and_predict()
        elif choice == 2:
            run_simulation()
        else:
            print("❌ Invalid choice. Exiting.")
            sys.exit(1)
    except Exception as e:
        print(f"\n🚨 Error occurred: {e}")
        stop = input("Do you want to stop the program? (y/n): ").strip().lower()
        if stop == "y":
            print("🛑 Program stopped by user.")
            sys.exit(1)
        else:
            print("⚠️ Continuing execution...")

if __name__ == "__main__":
    main()
