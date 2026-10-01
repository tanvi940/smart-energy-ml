
import pandas as pd
import requests
import time

# Load historical readings
df = pd.read_csv("features_energy.csv")

# Flask API
API_URL = "http://127.0.0.1:5000/predict"

# Time between simulated readings
INTERVAL = 5

print("Live Energy Simulator Started!")

try:
    while True:
        for _, row in df.iterrows():

            data = {
                "Hour": int(row["Hour"]),
                "Day": int(row["Day"]),
                "Month": int(row["Month"]),
                "DayOfWeek": int(row["DayOfWeek"]),
                "Lag_1": float(row["Lag_1"]),
                "Lag_24": float(row["Lag_24"]),
                "Global_active_power": float(
                    row["Global_active_power"]
                )
            }

            try:
                response = requests.post(
                    API_URL,
                    json=data,
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()

                    print(
                        "Reading:",
                        data["Global_active_power"],
                        "| Prediction:",
                        round(result["predicted_energy"], 2),
                        "| Status:",
                        result["status"]
                    )
                else:
                    print("API Error:", response.status_code)

            except requests.exceptions.RequestException as e:
                print("Connection error:", e)
                break

            time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("Live Simulator Stopped.")