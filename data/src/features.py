import pandas as pd

df = pd.read_csv("../cleaned_energy.csv")

df["Datetime"] = pd.to_datetime(df["Datetime"])

# Convert minute-level data to hourly data
df = df.set_index("Datetime")

df = df.resample("1h").mean(numeric_only=True)

df = df.reset_index()

# Time features
df["Hour"] = df["Datetime"].dt.hour
df["Day"] = df["Datetime"].dt.day
df["Month"] = df["Datetime"].dt.month
df["DayOfWeek"] = df["Datetime"].dt.dayofweek

# Previous hour
df["Lag_1"] = df["Global_active_power"].shift(1)

# Previous 24 hours
df["Lag_24"] = df["Global_active_power"].shift(24)

# Remove missing values
df = df.dropna()

# Save features
df.to_csv("../features_energy.csv", index=False)

print("Feature engineering completed!")
print()
print(df.head())
print()
print("Shape:", df.shape)