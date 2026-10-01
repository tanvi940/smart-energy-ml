import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

df = pd.read_csv("../features_energy.csv")

features = [
    "Hour",
    "Day",
    "Month",
    "DayOfWeek",
    "Lag_1",
    "Lag_24"
]

X = df[features]
y = df["Global_active_power"]

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("Model Training Completed!")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

joblib.dump(model, "../energy_model.pkl")

print("Model saved successfully!")