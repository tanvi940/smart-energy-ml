import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

df = pd.read_csv("../features_energy.csv")

features = [
    "Global_active_power",
    "Hour",
    "DayOfWeek",
    "Lag_1",
    "Lag_24"
]

X = df[features]

model = IsolationForest(
    n_estimators=100,
    contamination=0.01,
    random_state=42
)

df["Anomaly"] = model.fit_predict(X)

df["Anomaly"] = df["Anomaly"].map({
    1: 0,
    -1: 1
})

joblib.dump(model, "../anomaly_model.pkl")

df.to_csv("../anomaly_results.csv", index=False)

print("Anomaly detection completed!")
print("Normal:", (df["Anomaly"] == 0).sum())
print("Anomalies:", (df["Anomaly"] == 1).sum())
print("Anomaly model saved successfully!")