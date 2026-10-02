
from flask import Flask, request, jsonify
import joblib
import os

from database import (
    create_table,
    save_prediction,
    get_predictions
)

app = Flask(__name__)

energy_model = joblib.load("energy_model.pkl")
anomaly_model = joblib.load("anomaly_model.pkl")

create_table()


@app.route("/")
def home():
    return "Smart Energy ML API is running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    energy_features = [[
        data["Hour"],
        data["Day"],
        data["Month"],
        data["DayOfWeek"],
        data["Lag_1"],
        data["Lag_24"]
    ]]

    predicted_energy = energy_model.predict(
        energy_features
    )[0]

    anomaly_features = [[
        data["Global_active_power"],
        data["Hour"],
        data["DayOfWeek"],
        data["Lag_1"],
        data["Lag_24"]
    ]]

    anomaly_prediction = anomaly_model.predict(
        anomaly_features
    )[0]

    if anomaly_prediction == -1:
        status = "Anomaly"
    else:
        status = "Normal"

    actual_energy = float(
        data["Global_active_power"]
    )

    save_prediction(
        actual_energy,
        float(predicted_energy),
        status
    )

    return jsonify({
        "actual_energy": actual_energy,
        "predicted_energy": float(predicted_energy),
        "status": status
    })


@app.route("/history", methods=["GET"])
def history():

    data = get_predictions()

    return jsonify([
        {
            "id": row[0],
            "timestamp": row[1],
            "actual_energy": row[2],
            "predicted_energy": row[3],
            "status": row[4]
        }
        for row in data
    ])


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )

