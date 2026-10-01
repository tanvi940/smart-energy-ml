from flask import Flask, request, jsonify
import joblib
from database import create_table, save_prediction

app = Flask(__name__)

# Load ML models
energy_model = joblib.load("energy_model.pkl")
anomaly_model = joblib.load("anomaly_model.pkl")

# Create database table
create_table()


@app.route("/")
def home():
    return "Smart Energy ML API is running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    # Features for energy prediction
    energy_features = [[
        data["Hour"],
        data["Day"],
        data["Month"],
        data["DayOfWeek"],
        data["Lag_1"],
        data["Lag_24"]
    ]]

    # Predict energy consumption
    predicted_energy = energy_model.predict(
        energy_features
    )[0]

    # Features for anomaly detection
    anomaly_features = [[
        data["Global_active_power"],
        data["Hour"],
        data["DayOfWeek"],
        data["Lag_1"],
        data["Lag_24"]
    ]]

    # Detect anomaly
    anomaly_prediction = anomaly_model.predict(
        anomaly_features
    )[0]

    if anomaly_prediction == -1:
        status = "Anomaly"
    else:
        status = "Normal"

    # Actual energy from incoming reading
    actual_energy = float(
        data["Global_active_power"]
    )

    # Save prediction to database
    save_prediction(
        actual_energy,
        float(predicted_energy),
        status
    )

    # Send result back
    return jsonify({
        "actual_energy": actual_energy,
        "predicted_energy": float(predicted_energy),
        "status": status
    })


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )